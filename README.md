# bm25rs

A fast, streaming-friendly BM25 search engine written in Rust with memory-mapped (mmap) index support.

Inspired by [bm25s](https://github.com/xhluca/bm25s), but designed from the ground up for incremental indexing — add, delete, and update documents without rebuilding the entire index.

## Features

- **All 5 BM25 variants**: Lucene (default), Robertson, ATIRE, BM25L, BM25+
- **Streaming index**: Add documents incrementally, delete by ID, update in-place
- **Pre-filtered search**: Score only a subset of documents — up to 600x faster
- **Memory-mapped storage**: Load large indices with minimal RAM via mmap
- **Auto-persistence**: Point to a directory and the index saves/loads automatically
- **Python bindings**: Via [maturin](https://github.com/PyO3/maturin) / PyO3

## Python

### Install

```bash
cd python
uv venv && source .venv/bin/activate
uv pip install maturin
maturin develop --release
```

### Usage

```python
from bm25rs import BM25

# Create a persistent index (auto-saves on every mutation)
index = BM25(index="./my_index")
index.add(["the quick brown fox", "lazy dog on a mat", "fox and hound"])

# Search
results = index.search("quick fox", k=10)
for doc_id, score in results:
    print(f"doc {doc_id}: {score:.4f}")

# Pre-filtered search — only score a subset of documents
results = index.search("quick fox", k=10, subset=[0, 2])

# Streaming mutations (auto-saved to disk)
index.add(["a brand new document"])
index.delete([1])
index.update(0, "replaced text for doc zero")

# Reload later — just point to the same directory
index = BM25(index="./my_index")  # loads existing index, ready to search
```

### Constructor

```python
BM25(
    index=None,          # Path to persist the index (auto-save/load). None = in-memory only.
    method="lucene",     # "lucene", "robertson", "atire", "bm25l", "bm25+"
    k1=1.5,              # Term frequency saturation
    b=0.75,              # Document length normalization
    delta=0.5,           # Delta (BM25L/BM25+ only)
    use_stopwords=True,  # Remove English stopwords
)
```

### Methods

| Method | Description |
|---|---|
| `add(docs) -> list[int]` | Add documents (list of strings), returns assigned indices |
| `search(query, k, subset=None) -> list[tuple[int, float]]` | Top-k search. Pass `subset` to restrict scoring to specific doc IDs |
| `delete(doc_ids)` | Delete documents by their indices |
| `update(doc_id, text)` | Replace a document's text |
| `save(index)` | Explicit save (only needed for in-memory indices) |
| `load(index, mmap=False)` | Load from directory (static method) |
| `len(index)` | Number of active documents |

## Rust

### Add to your project

```toml
[dependencies]
bm25rs = "0.1"
```

### Usage

```rust
use bm25rs::{BM25, Method};

// Create an index
let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, true);

// Add documents (returns assigned indices)
let ids = index.add(&[
    "the quick brown fox jumps over the lazy dog",
    "a fast brown car drives over the bridge",
    "the fox is quick and clever",
]);

// Search — returns Vec<SearchResult> with .index and .score
let results = index.search("quick fox", 10);
for r in &results {
    println!("doc {}: {:.4}", r.index, r.score);
}

// Pre-filtered search — only score documents in the subset
let results = index.search_filtered("quick fox", 10, &[0, 2]);

// Streaming mutations
index.add(&["a brand new document"]);
index.delete(&[1]);
index.update(0, "updated text for doc zero");

// Save / load with mmap
index.save("./my_index").unwrap();
let index = BM25::load("./my_index", true).unwrap(); // mmap=true
```

### API

```rust
// Constructor
BM25::new(method: Method, k1: f32, b: f32, delta: f32, use_stopwords: bool) -> BM25

// Documents
fn add(&mut self, documents: &[&str]) -> Vec<usize>
fn delete(&mut self, doc_ids: &[usize])
fn update(&mut self, doc_id: usize, new_text: &str)

// Search
fn search(&self, query: &str, k: usize) -> Vec<SearchResult>
fn search_filtered(&self, query: &str, k: usize, subset: &[usize]) -> Vec<SearchResult>

// Persistence
fn save<P: AsRef<Path>>(&self, dir: P) -> io::Result<()>
fn load<P: AsRef<Path>>(dir: P, mmap: bool) -> io::Result<BM25>

// Info
fn len(&self) -> usize
fn is_empty(&self) -> bool
```

## Benchmarks

### BEIR SciFact — 5k documents, 300 queries

| Metric | bm25s | bm25rs |
|---|---|---|
| **NDCG@10** | 0.6617 | **0.6650** |
| Index time | 0.581s | **0.190s** (3.1x faster) |
| Search time | 0.031s | **0.011s** (2.8x faster) |

### BEIR MS MARCO — 8.8M documents, 6,980 queries

| Metric | bm25s | bm25rs |
|---|---|---|
| **NDCG@10** | 0.2124 | **0.2186** |
| Index time | 377.9s | **106.6s** (3.5x faster) |
| Index throughput | 23,395 d/s | **82,910 d/s** |
| Search throughput | 16 q/s | **65 q/s** (4x faster) |
| Mmap mem delta | 153 MB | **109 MB** |

### Pre-filtered search — 100k documents, 1k queries

| Filter size | Throughput | Speedup |
|---|---|---|
| No filter | 592 q/s | baseline |
| 1,000 docs | 4,286 q/s | 7x |
| 100 docs | 52,000 q/s | 88x |
| 10 docs | 366,000 q/s | 618x |

## Design

bm25s pre-computes all BM25 scores at index time (eager scoring). This makes queries fast but rebuilding the index is required to add or remove documents.

bm25rs uses **lazy scoring** — it stores raw term frequencies in an inverted index and computes BM25 scores at query time. This makes the index naturally streaming-friendly: add, delete, and update are cheap operations that don't require a full rebuild.

Pre-filtered search uses a **doc-centric** approach: instead of scanning posting lists, it iterates only the subset and binary-searches each document's term frequency. This makes it O(|subset| * |query_terms| * log n) instead of O(|posting_list|).

On-disk, the index uses a flat binary format with memory-mapped postings and document lengths, keeping RAM usage minimal for large indices.

## License

MIT
