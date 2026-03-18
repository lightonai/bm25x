use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write as IoWrite};
use std::path::Path;

use memmap2::Mmap;

use crate::index::BM25Index;
use crate::scoring::Method;

const MAGIC: u64 = 0x424D32355253; // "BM25RS" in hex
const VERSION: u32 = 1;

/// On-disk posting entry: (doc_id, tf)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PostingEntry {
    pub doc_id: u32,
    pub tf: u32,
}

/// On-disk term offset: (byte_offset_in_postings, count)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TermOffset {
    pub offset: u64,
    pub count: u32,
    _pad: u32,
}

/// Header stored on disk.
#[repr(C)]
struct Header {
    magic: u64,
    version: u32,
    num_terms: u32,
    next_doc_id: u32,
    num_docs: u32,
    total_tokens: u64,
    k1_bits: u32,
    b_bits: u32,
    delta_bits: u32,
    method_id: u8,
    _pad: [u8; 3],
}

/// Memory-mapped data backing an index.
pub struct MmapData {
    doc_lens_mmap: Mmap,
    postings_mmap: Mmap,
    offsets_mmap: Mmap,
    num_terms: u32,
    _max_doc_id: u32,
}

impl MmapData {
    pub fn get_doc_length(&self, doc_id: u32) -> u32 {
        let idx = doc_id as usize;
        let slice = unsafe {
            std::slice::from_raw_parts(
                self.doc_lens_mmap.as_ptr() as *const u32,
                self.doc_lens_mmap.len() / 4,
            )
        };
        *slice.get(idx).unwrap_or(&0)
    }

    pub fn all_doc_lengths(&self) -> Vec<u32> {
        let slice = unsafe {
            std::slice::from_raw_parts(
                self.doc_lens_mmap.as_ptr() as *const u32,
                self.doc_lens_mmap.len() / 4,
            )
        };
        slice.to_vec()
    }

    pub fn posting_count(&self, term_id: u32) -> u32 {
        if term_id >= self.num_terms {
            return 0;
        }
        let offsets = unsafe {
            std::slice::from_raw_parts(
                self.offsets_mmap.as_ptr() as *const TermOffset,
                self.num_terms as usize,
            )
        };
        offsets[term_id as usize].count
    }

    pub fn for_each_posting<F: FnMut(u32, u32)>(&self, term_id: u32, f: &mut F) {
        if term_id >= self.num_terms {
            return;
        }
        let offsets = unsafe {
            std::slice::from_raw_parts(
                self.offsets_mmap.as_ptr() as *const TermOffset,
                self.num_terms as usize,
            )
        };
        let to = &offsets[term_id as usize];
        let entries = unsafe {
            let base = self.postings_mmap.as_ptr().add(to.offset as usize) as *const PostingEntry;
            std::slice::from_raw_parts(base, to.count as usize)
        };
        for entry in entries {
            f(entry.doc_id, entry.tf);
        }
    }
}

impl BM25Index {
    /// Save the index to a directory.
    pub fn save<P: AsRef<Path>>(&self, dir: P) -> io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;

        // Write header
        let header = Header {
            magic: MAGIC,
            version: VERSION,
            num_terms: self.get_vocab().len() as u32,
            next_doc_id: self.get_next_doc_id(),
            num_docs: self.len() as u32,
            total_tokens: self.get_total_tokens(),
            k1_bits: self.k1.to_bits(),
            b_bits: self.b.to_bits(),
            delta_bits: self.delta.to_bits(),
            method_id: self.method.to_id(),
            _pad: [0; 3],
        };
        write_bytes(dir.join("header.bin"), as_bytes(&header))?;

        // Write doc_lengths
        write_bytes(
            dir.join("doc_lens.bin"),
            as_slice_bytes(self.get_doc_lengths_slice()),
        )?;

        // Build flat postings and offsets
        let num_terms = self.get_vocab().len();
        let postings = self.get_postings();
        let mut flat_postings: Vec<PostingEntry> = Vec::new();
        let mut offsets: Vec<TermOffset> = Vec::with_capacity(num_terms);

        for entries in postings.iter().take(num_terms) {
            let byte_offset = flat_postings.len() * std::mem::size_of::<PostingEntry>();
            let count = entries.len() as u32;
            for &(doc_id, tf) in entries {
                flat_postings.push(PostingEntry { doc_id, tf });
            }
            offsets.push(TermOffset {
                offset: byte_offset as u64,
                count,
                _pad: 0,
            });
        }

        write_bytes(dir.join("postings.bin"), as_slice_bytes(&flat_postings))?;
        write_bytes(dir.join("offsets.bin"), as_slice_bytes(&offsets))?;

        // Write vocab and deleted as bincode
        let vocab_bytes = bincode::serialize(self.get_vocab()).map_err(io::Error::other)?;
        fs::write(dir.join("vocab.bin"), &vocab_bytes)?;

        let deleted_bytes = bincode::serialize(self.get_deleted()).map_err(io::Error::other)?;
        fs::write(dir.join("deleted.bin"), &deleted_bytes)?;

        Ok(())
    }

    /// Load an index from a directory. If `mmap` is true, postings and doc_lengths
    /// are memory-mapped instead of loaded into RAM.
    pub fn load<P: AsRef<Path>>(dir: P, mmap: bool) -> io::Result<Self> {
        let dir = dir.as_ref();

        // Read header
        let header_bytes = fs::read(dir.join("header.bin"))?;
        if header_bytes.len() < std::mem::size_of::<Header>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "header too small",
            ));
        }
        let header: Header = unsafe { std::ptr::read(header_bytes.as_ptr() as *const Header) };
        if header.magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad magic number",
            ));
        }

        let method = Method::from_id(header.method_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unknown method"))?;
        let k1 = f32::from_bits(header.k1_bits);
        let b = f32::from_bits(header.b_bits);
        let delta = f32::from_bits(header.delta_bits);

        // Read vocab and deleted
        let vocab_bytes = fs::read(dir.join("vocab.bin"))?;
        let vocab: HashMap<String, u32> =
            bincode::deserialize(&vocab_bytes).map_err(io::Error::other)?;

        let deleted_bytes = fs::read(dir.join("deleted.bin"))?;
        let deleted: HashSet<u32> =
            bincode::deserialize(&deleted_bytes).map_err(io::Error::other)?;

        let mut index = BM25Index::new(method, k1, b, delta, true);

        if mmap {
            let doc_lens_file = File::open(dir.join("doc_lens.bin"))?;
            let postings_file = File::open(dir.join("postings.bin"))?;
            let offsets_file = File::open(dir.join("offsets.bin"))?;

            let doc_lens_mmap = unsafe { Mmap::map(&doc_lens_file)? };
            let postings_mmap = unsafe { Mmap::map(&postings_file)? };
            let offsets_mmap = unsafe { Mmap::map(&offsets_file)? };

            let mmap_data = MmapData {
                doc_lens_mmap,
                postings_mmap,
                offsets_mmap,
                num_terms: header.num_terms,
                _max_doc_id: header.next_doc_id,
            };

            index.set_mmap_internals(
                vocab,
                deleted,
                header.total_tokens,
                header.num_docs,
                header.next_doc_id,
                mmap_data,
            );
        } else {
            // Read everything into memory
            let doc_lens_bytes = fs::read(dir.join("doc_lens.bin"))?;
            let doc_lengths: Vec<u32> = bytes_to_vec(&doc_lens_bytes);

            let postings_bytes = fs::read(dir.join("postings.bin"))?;
            let flat_postings: Vec<PostingEntry> = bytes_to_vec_posting(&postings_bytes);

            let offsets_bytes = fs::read(dir.join("offsets.bin"))?;
            let flat_offsets: Vec<TermOffset> = bytes_to_vec_offset(&offsets_bytes);

            // Reconstruct posting lists
            let num_terms = header.num_terms as usize;
            let mut postings = Vec::with_capacity(num_terms);
            for to in flat_offsets.iter().take(num_terms) {
                let start = to.offset as usize / std::mem::size_of::<PostingEntry>();
                let entries: Vec<(u32, u32)> = flat_postings[start..start + to.count as usize]
                    .iter()
                    .map(|e| (e.doc_id, e.tf))
                    .collect();
                postings.push(entries);
            }

            index.set_internals(
                vocab,
                deleted,
                doc_lengths,
                postings,
                header.total_tokens,
                header.num_docs,
                header.next_doc_id,
            );
        }

        Ok(index)
    }
}

// --- Helper functions ---

fn write_bytes<P: AsRef<Path>>(path: P, data: &[u8]) -> io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(data)?;
    f.flush()?;
    Ok(())
}

fn as_bytes<T>(val: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(val as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn as_slice_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

fn bytes_to_vec(bytes: &[u8]) -> Vec<u32> {
    let count = bytes.len() / 4;
    let mut v = Vec::with_capacity(count);
    for i in 0..count {
        let b = [
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ];
        v.push(u32::from_ne_bytes(b));
    }
    v
}

fn bytes_to_vec_posting(bytes: &[u8]) -> Vec<PostingEntry> {
    let size = std::mem::size_of::<PostingEntry>();
    let count = bytes.len() / size;
    let mut v = Vec::with_capacity(count);
    for i in 0..count {
        let offset = i * size;
        let doc_id = u32::from_ne_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        let tf = u32::from_ne_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        v.push(PostingEntry { doc_id, tf });
    }
    v
}

fn bytes_to_vec_offset(bytes: &[u8]) -> Vec<TermOffset> {
    let size = std::mem::size_of::<TermOffset>();
    let count = bytes.len() / size;
    let mut v = Vec::with_capacity(count);
    for i in 0..count {
        let offset = i * size;
        let o = u64::from_ne_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        let c = u32::from_ne_bytes([
            bytes[offset + 8],
            bytes[offset + 9],
            bytes[offset + 10],
            bytes[offset + 11],
        ]);
        v.push(TermOffset {
            offset: o,
            count: c,
            _pad: 0,
        });
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_save_and_load() {
        let mut index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&[
            "the quick brown fox",
            "the lazy dog",
            "a brown dog and a quick fox",
        ]);

        let dir = TempDir::new().unwrap();
        index.save(dir.path()).unwrap();

        let loaded = BM25Index::load(dir.path(), false).unwrap();
        assert_eq!(loaded.len(), 3);

        let results = loaded.search("quick fox", 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_save_and_load_mmap() {
        let mut index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&[
            "the quick brown fox",
            "the lazy dog",
            "a brown dog and a quick fox",
        ]);

        let dir = TempDir::new().unwrap();
        index.save(dir.path()).unwrap();

        let loaded = BM25Index::load(dir.path(), true).unwrap();
        assert_eq!(loaded.len(), 3);

        let results = loaded.search("quick fox", 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_save_load_with_deletions() {
        let mut index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello world", "foo bar", "hello foo"]);
        index.delete(&[1]);

        let dir = TempDir::new().unwrap();
        index.save(dir.path()).unwrap();

        let loaded = BM25Index::load(dir.path(), false).unwrap();
        assert_eq!(loaded.len(), 2);

        let results = loaded.search("foo", 10);
        assert!(results.iter().all(|r| r.index != 1));
    }
}
