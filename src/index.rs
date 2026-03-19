use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::scoring::{self, Method, ScoringParams};
use crate::storage::MmapData;
use crate::tokenizer::{Tokenizer, TokenizerMode};

/// Cap rayon parallelism at 12 threads max.
pub(crate) fn capped_pool() -> rayon::ThreadPool {
    let n = std::thread::available_parallelism()
        .map(|p| p.get().min(12))
        .unwrap_or(4);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("failed to build rayon pool")
}

/// A scored document result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The document index (0-based, contiguous).
    pub index: usize,
    /// The BM25 score.
    pub score: f32,
}

/// Wrapper for BinaryHeap min-heap (we want to keep top-k highest scores).
struct MinScored(f32, u32);

impl PartialEq for MinScored {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MinScored {}
impl PartialOrd for MinScored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinScored {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

/// The core BM25 index.
///
/// Document indices are contiguous 0..n. Deleting a document compacts the index:
/// all documents after the deleted one shift down by one.
pub struct BM25 {
    // Scoring parameters
    pub k1: f32,
    pub b: f32,
    pub delta: f32,
    pub method: Method,

    // Inverted index: term_id -> Vec<(doc_id, tf)> sorted by doc_id
    postings: Vec<Vec<(u32, u32)>>,

    // Cached document frequency per term
    doc_freqs: Vec<u32>,

    // Document metadata
    doc_lengths: Vec<u32>,
    total_tokens: u64,
    num_docs: u32,

    // Vocabulary: token string -> term_id
    vocab: HashMap<String, u32>,

    // Tokenizer
    tokenizer: Tokenizer,

    // Mmap backing (if loaded from disk)
    mmap_data: Option<MmapData>,

    // Auto-save path (if set, mutations persist to disk automatically)
    index_path: Option<PathBuf>,

    // If true, CUDA is required — operations fail instead of falling back to CPU.
    cuda_required: bool,
}

impl BM25 {
    /// Require CUDA: if set, operations will fail instead of falling back to CPU.
    /// Call after construction: `BM25::default().require_cuda()`
    pub fn require_cuda(mut self) -> Self {
        self.cuda_required = true;
        self
    }

    /// Returns true if this index requires CUDA.
    pub fn is_cuda_required(&self) -> bool {
        self.cuda_required
    }

    /// Create a new empty index.
    pub fn new(method: Method, k1: f32, b: f32, delta: f32, use_stopwords: bool) -> Self {
        Self::with_options(
            method,
            k1,
            b,
            delta,
            TokenizerMode::UnicodeStem,
            use_stopwords,
            false,
        )
    }

    /// Create a new empty index with a specific tokenizer mode.
    pub fn with_tokenizer(
        method: Method,
        k1: f32,
        b: f32,
        delta: f32,
        tokenizer_mode: TokenizerMode,
        use_stopwords: bool,
    ) -> Self {
        Self::with_options(method, k1, b, delta, tokenizer_mode, use_stopwords, false)
    }

    /// Create a new empty index with all options including CUDA.
    ///
    /// If `cuda` is true, operations will return errors instead of falling back to CPU.
    pub fn with_options(
        method: Method,
        k1: f32,
        b: f32,
        delta: f32,
        tokenizer_mode: TokenizerMode,
        use_stopwords: bool,
        cuda: bool,
    ) -> Self {
        BM25 {
            k1,
            b,
            delta,
            method,
            postings: Vec::new(),
            doc_freqs: Vec::new(),
            doc_lengths: Vec::new(),
            total_tokens: 0,
            num_docs: 0,
            vocab: HashMap::new(),
            tokenizer: Tokenizer::with_mode(tokenizer_mode, use_stopwords),
            mmap_data: None,
            index_path: None,
            cuda_required: cuda,
        }
    }

    /// Open a persistent index at the given path.
    ///
    /// - If the directory already contains a saved index, it is loaded (mmap).
    /// - Every mutation (`add`, `delete`, `update`) auto-saves to disk.
    /// - If the directory doesn't exist yet, a new empty index is created.
    /// - If `cuda` is true, operations will return errors instead of falling back to CPU.
    pub fn open<P: AsRef<Path>>(
        path: P,
        method: Method,
        k1: f32,
        b: f32,
        delta: f32,
        tokenizer_mode: TokenizerMode,
        use_stopwords: bool,
    ) -> io::Result<Self> {
        Self::open_with_cuda(
            path,
            method,
            k1,
            b,
            delta,
            tokenizer_mode,
            use_stopwords,
            false,
        )
    }

    /// Open a persistent index with explicit CUDA control.
    #[allow(clippy::too_many_arguments)]
    pub fn open_with_cuda<P: AsRef<Path>>(
        path: P,
        method: Method,
        k1: f32,
        b: f32,
        delta: f32,
        tokenizer_mode: TokenizerMode,
        use_stopwords: bool,
        cuda: bool,
    ) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if path.join("header.bin").exists() {
            let mut index = Self::load(&path, true)?;
            index.index_path = Some(path);
            index.cuda_required = cuda;
            Ok(index)
        } else {
            let mut index =
                Self::with_options(method, k1, b, delta, tokenizer_mode, use_stopwords, cuda);
            index.index_path = Some(path);
            Ok(index)
        }
    }

    /// Auto-save to disk if an index path is configured.
    fn auto_save(&self) -> io::Result<()> {
        if let Some(ref path) = self.index_path {
            self.save(path)?;
        }
        Ok(())
    }

    /// Add documents to the index. Returns the assigned document indices.
    ///
    /// When compiled with the `cuda` feature and a GPU is available, posting list
    /// construction is offloaded to CUDA for faster indexing. Tokenization always
    /// runs on CPU (rayon). Falls back to CPU automatically on GPU failure.
    pub fn add(&mut self, documents: &[&str]) -> io::Result<Vec<usize>> {
        if self.mmap_data.is_some() {
            self.materialize_mmap();
        }

        // Try GPU path
        #[cfg(feature = "cuda")]
        {
            if let Some(ctx) = crate::cuda::get_global_context() {
                match crate::cuda::add_documents_cuda(
                    &ctx,
                    &self.tokenizer,
                    documents,
                    &self.vocab,
                    self.num_docs,
                ) {
                    Ok(result) => return self.apply_cuda_result(documents.len(), result),
                    Err(e) => {
                        if self.cuda_required {
                            return Err(io::Error::other(format!(
                                "cuda=true but CUDA indexing failed: {}",
                                e
                            )));
                        }
                        eprintln!("[bm25x] CUDA indexing failed: {}, falling back to CPU", e);
                    }
                }
            } else if self.cuda_required {
                return Err(io::Error::other("cuda=true but no CUDA GPU is available"));
            }
        }
        #[cfg(not(feature = "cuda"))]
        if self.cuda_required {
            return Err(io::Error::other(
                "cuda=true but bm25x was not compiled with CUDA support",
            ));
        }

        self.add_cpu(documents)
    }

    /// CPU-only add implementation (original algorithm).
    fn add_cpu(&mut self, documents: &[&str]) -> io::Result<Vec<usize>> {
        let t_total = std::time::Instant::now();

        // Phase 1: tokenize + compute TF maps in parallel
        let t0 = std::time::Instant::now();
        let pool = capped_pool();
        let tokenized: Vec<(u32, HashMap<String, u32>)> = pool.install(|| {
            documents
                .par_iter()
                .map(|doc| {
                    let tokens = self.tokenizer.tokenize_owned(doc);
                    let doc_len = tokens.len() as u32;
                    let mut tf_map: HashMap<String, u32> = HashMap::new();
                    for token in tokens {
                        *tf_map.entry(token).or_insert(0) += 1;
                    }
                    (doc_len, tf_map)
                })
                .collect()
        });
        let t_tokenize = t0.elapsed();

        // Phase 2: merge into index sequentially (cheap — just HashMap lookups + Vec pushes)
        let t0 = std::time::Instant::now();
        let base_id = self.num_docs;
        let mut ids = Vec::with_capacity(documents.len());

        for (i, (doc_len, tf_map)) in tokenized.into_iter().enumerate() {
            let doc_id = base_id + i as u32;

            for (token, tf) in tf_map {
                let term_id = self.get_or_create_term(&token);
                self.postings[term_id as usize].push((doc_id, tf));
                self.doc_freqs[term_id as usize] += 1;
            }

            self.doc_lengths.push(doc_len);
            self.total_tokens += doc_len as u64;
            ids.push(doc_id as usize);
        }
        let t_merge = t0.elapsed();

        self.num_docs = base_id + documents.len() as u32;

        let t_total_elapsed = t_total.elapsed();
        if std::env::var("BM25X_PROFILE").is_ok() {
            eprintln!(
                "[bm25x-cpu] {:.3}s total | tokenize={:.3}s merge={:.3}s | {:.0} d/s",
                t_total_elapsed.as_secs_f64(),
                t_tokenize.as_secs_f64(),
                t_merge.as_secs_f64(),
                documents.len() as f64 / t_total_elapsed.as_secs_f64(),
            );
        }

        self.auto_save()?;
        Ok(ids)
    }

    /// Apply the result from CUDA-accelerated indexing into the BM25 struct.
    #[cfg(feature = "cuda")]
    fn apply_cuda_result(
        &mut self,
        num_new_docs: usize,
        result: crate::cuda::AddResult,
    ) -> io::Result<Vec<usize>> {
        let base_id = self.num_docs;

        if self.num_docs == 0 && self.postings.is_empty() {
            // Fresh index: move everything directly (O(1), no copying)
            self.doc_freqs = result.postings.iter().map(|p| p.len() as u32).collect();
            self.postings = result.postings;
            self.vocab = result.vocab;
        } else {
            // Existing index: merge
            let new_vocab_size = result.vocab.len();
            while self.postings.len() < new_vocab_size {
                self.postings.push(Vec::new());
                self.doc_freqs.push(0);
            }
            for (term_id, gpu_plist) in result.postings.into_iter().enumerate() {
                if !gpu_plist.is_empty() {
                    let count = gpu_plist.len() as u32;
                    self.postings[term_id].extend_from_slice(&gpu_plist);
                    self.doc_freqs[term_id] += count;
                }
            }
            if result.vocab.len() > self.vocab.len() {
                self.vocab = result.vocab;
            }
        }

        self.doc_lengths.extend_from_slice(&result.doc_lengths);
        self.total_tokens += result.total_tokens;
        self.num_docs = base_id + num_new_docs as u32;

        let ids: Vec<usize> = (base_id as usize..base_id as usize + num_new_docs).collect();
        self.auto_save()?;
        Ok(ids)
    }

    /// Search the index and return top-k results sorted by descending score.
    pub fn search(&self, query: &str, k: usize) -> Vec<SearchResult> {
        if self.num_docs == 0 {
            return Vec::new();
        }

        let query_tokens = self.tokenizer.tokenize_owned(query);
        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl: self.total_tokens as f32 / self.num_docs as f32,
        };

        // Use a flat score array indexed by doc_id for O(1) accumulation
        let mut scores = vec![0.0f32; self.num_docs as usize];
        let mut touched = Vec::new();

        let mut seen_terms: HashSet<u32> = HashSet::new();

        for token in &query_tokens {
            let term_id = match self.vocab.get(token.as_str()) {
                Some(&id) => id,
                None => continue,
            };
            if !seen_terms.insert(term_id) {
                continue;
            }

            let df = self.doc_freqs.get(term_id as usize).copied().unwrap_or(0);
            if df == 0 {
                continue;
            }

            let idf_val = scoring::idf(self.method, self.num_docs, df);

            self.for_each_posting(term_id, |doc_id, tf| {
                let dl = self.get_doc_length(doc_id);
                let s = scoring::score(self.method, tf, dl, &params, idf_val);
                let idx = doc_id as usize;
                if scores[idx] == 0.0 {
                    touched.push(doc_id);
                }
                scores[idx] += s;
            });
        }

        Self::topk_from_scores(&scores, &touched, k)
    }

    /// Batch search: run multiple queries in parallel using rayon.
    /// Returns one result list per query. Much faster than sequential search()
    /// because it amortizes score array allocation and enables CPU parallelism.
    pub fn search_batch(&self, queries: &[&str], k: usize) -> Vec<Vec<SearchResult>> {
        if self.num_docs == 0 {
            return queries.iter().map(|_| Vec::new()).collect();
        }

        let pool = capped_pool();
        pool.install(|| {
            queries
                .par_iter()
                .map(|query| self.search(query, k))
                .collect()
        })
    }

    /// Batch filtered search: run multiple queries with per-query subsets in parallel.
    pub fn search_filtered_batch(
        &self,
        queries: &[&str],
        k: usize,
        subsets: &[&[usize]],
    ) -> Vec<Vec<SearchResult>> {
        if self.num_docs == 0 {
            return queries.iter().map(|_| Vec::new()).collect();
        }

        let pool = capped_pool();
        pool.install(|| {
            queries
                .par_iter()
                .zip(subsets.par_iter())
                .map(|(query, subset)| self.search_filtered(query, k, subset))
                .collect()
        })
    }

    /// Batch GPU search: process multiple queries across multiple GPUs.
    /// If a MultiGpuSearchIndex is provided, queries are distributed across GPUs.
    /// Otherwise falls back to single-GPU sequential processing.
    #[cfg(feature = "cuda")]
    pub fn search_gpu_batch(
        &self,
        gpu_index: &mut crate::cuda::GpuSearchIndex,
        queries: &[&str],
        k: usize,
    ) -> Vec<Vec<SearchResult>> {
        let ctx = match crate::cuda::get_global_context() {
            Some(c) => c,
            None => return self.search_batch(queries, k),
        };

        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl: self.total_tokens as f32 / self.num_docs as f32,
        };

        let all_query_terms = self.tokenize_queries(queries);

        let mut results = Vec::with_capacity(queries.len());
        for query_terms in &all_query_terms {
            match gpu_index.search(&ctx, query_terms, params.k1, params.b, params.avgdl, k) {
                Ok(r) => results.push(
                    r.into_iter()
                        .map(|(doc_id, score)| SearchResult {
                            index: doc_id as usize,
                            score,
                        })
                        .collect(),
                ),
                Err(e) => {
                    eprintln!("[bm25x] GPU batch search failed: {}", e);
                    results.push(Vec::new());
                }
            }
        }
        results
    }

    /// Batch GPU search across multiple GPUs. Queries distributed evenly.
    #[cfg(feature = "cuda")]
    pub fn search_multi_gpu_batch(
        &self,
        multi_gpu: &mut crate::multi_gpu::MultiGpuSearchIndex,
        queries: &[&str],
        k: usize,
    ) -> Vec<Vec<SearchResult>> {
        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl: self.total_tokens as f32 / self.num_docs as f32,
        };

        let all_query_terms = self.tokenize_queries(queries);

        let raw_results =
            multi_gpu.search_batch(&all_query_terms, params.k1, params.b, params.avgdl, k);

        raw_results
            .into_iter()
            .map(|r| {
                r.into_iter()
                    .map(|(doc_id, score)| SearchResult {
                        index: doc_id as usize,
                        score,
                    })
                    .collect()
            })
            .collect()
    }

    /// Helper: tokenize + resolve term IDs for a batch of queries (parallel on CPU).
    #[cfg(feature = "cuda")]
    fn tokenize_queries(&self, queries: &[&str]) -> Vec<Vec<(u32, f32)>> {
        let pool = capped_pool();
        pool.install(|| {
            queries
                .par_iter()
                .map(|query| {
                    let query_tokens = self.tokenizer.tokenize_owned(query);
                    let mut seen = std::collections::HashSet::new();
                    let mut terms = Vec::new();
                    for token in &query_tokens {
                        if let Some(&term_id) = self.vocab.get(token.as_str()) {
                            if seen.insert(term_id) {
                                let df = self.doc_freqs.get(term_id as usize).copied().unwrap_or(0);
                                if df > 0 {
                                    terms.push((
                                        term_id,
                                        scoring::idf(self.method, self.num_docs, df),
                                    ));
                                }
                            }
                        }
                    }
                    terms
                })
                .collect()
        })
    }

    /// Search restricted to a subset of document IDs (pre-filtering).
    /// Only documents whose index is in `subset` will be scored.
    /// IDF is computed from global corpus stats so scores stay comparable.
    ///
    /// Uses a doc-centric approach: iterates subset IDs and looks up each doc's
    /// TF via binary search on posting lists. Cost is O(|subset| * |query_terms| * log(posting_len)).
    pub fn search_filtered(&self, query: &str, k: usize, subset: &[usize]) -> Vec<SearchResult> {
        if self.num_docs == 0 || subset.is_empty() {
            return Vec::new();
        }

        let query_tokens = self.tokenizer.tokenize_owned(query);
        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl: self.total_tokens as f32 / self.num_docs as f32,
        };

        let mut query_terms: Vec<(u32, f32)> = Vec::new();
        let mut seen_terms: HashSet<u32> = HashSet::new();
        for token in &query_tokens {
            let term_id = match self.vocab.get(token.as_str()) {
                Some(&id) => id,
                None => continue,
            };
            if !seen_terms.insert(term_id) {
                continue;
            }
            let df = self.doc_freqs.get(term_id as usize).copied().unwrap_or(0);
            if df == 0 {
                continue;
            }
            query_terms.push((term_id, scoring::idf(self.method, self.num_docs, df)));
        }
        if query_terms.is_empty() {
            return Vec::new();
        }

        let mut heap = BinaryHeap::with_capacity(k + 1);
        for &doc_idx in subset {
            let doc_id = doc_idx as u32;
            if doc_id >= self.num_docs {
                continue;
            }
            let dl = self.get_doc_length(doc_id);
            let mut total_score = 0.0f32;
            for &(term_id, idf_val) in &query_terms {
                if let Some(tf) = self.get_tf(term_id, doc_id) {
                    total_score += scoring::score(self.method, tf, dl, &params, idf_val);
                }
            }
            if total_score > 0.0 {
                heap.push(MinScored(total_score, doc_id));
                if heap.len() > k {
                    heap.pop();
                }
            }
        }

        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|MinScored(score, doc_id)| SearchResult {
                index: doc_id as usize,
                score,
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results
    }

    fn topk_from_scores(scores: &[f32], touched: &[u32], k: usize) -> Vec<SearchResult> {
        let mut heap = BinaryHeap::with_capacity(k + 1);
        for &doc_id in touched {
            let s = scores[doc_id as usize];
            if s > 0.0 {
                heap.push(MinScored(s, doc_id));
                if heap.len() > k {
                    heap.pop();
                }
            }
        }
        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|MinScored(score, doc_id)| SearchResult {
                index: doc_id as usize,
                score,
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results
    }

    /// Delete one or more documents by their indices.
    /// All documents after a deleted index shift down to fill the gap.
    /// For example: deleting doc 1 from [0,1,2] makes old doc 2 become new doc 1.
    pub fn delete(&mut self, doc_ids: &[usize]) -> io::Result<()> {
        if doc_ids.is_empty() {
            return Ok(());
        }
        if self.mmap_data.is_some() {
            self.materialize_mmap();
        }

        // Sort and deduplicate, filter out-of-range
        let mut to_delete: Vec<u32> = doc_ids
            .iter()
            .map(|&id| id as u32)
            .filter(|&id| id < self.num_docs)
            .collect();
        to_delete.sort_unstable();
        to_delete.dedup();

        if to_delete.is_empty() {
            return Ok(());
        }

        // Subtract deleted doc lengths from total_tokens
        for &id in &to_delete {
            self.total_tokens -= self.doc_lengths[id as usize] as u64;
        }

        // Build old_id -> new_id mapping.
        // Deleted docs map to u32::MAX (sentinel), others shift down.
        let old_count = self.num_docs as usize;
        let mut id_map: Vec<u32> = Vec::with_capacity(old_count);
        let mut del_idx = 0;
        let mut shift = 0u32;
        for old_id in 0..old_count as u32 {
            if del_idx < to_delete.len() && to_delete[del_idx] == old_id {
                id_map.push(u32::MAX); // sentinel: deleted
                shift += 1;
                del_idx += 1;
            } else {
                id_map.push(old_id - shift);
            }
        }

        let new_count = self.num_docs - to_delete.len() as u32;

        // Compact doc_lengths
        let mut new_doc_lengths = Vec::with_capacity(new_count as usize);
        for (old_id, &dl) in self.doc_lengths.iter().enumerate() {
            if id_map[old_id] != u32::MAX {
                new_doc_lengths.push(dl);
            }
        }
        self.doc_lengths = new_doc_lengths;

        // Remap posting lists: remove deleted entries, remap doc_ids
        for (term_id, plist) in self.postings.iter_mut().enumerate() {
            let old_len = plist.len();
            plist.retain(|&(did, _)| id_map[did as usize] != u32::MAX);
            let removed = old_len - plist.len();
            if removed > 0 {
                self.doc_freqs[term_id] -= removed as u32;
            }
            for entry in plist.iter_mut() {
                entry.0 = id_map[entry.0 as usize];
            }
        }

        self.num_docs = new_count;
        self.auto_save()
    }

    /// Update a document's text. The document keeps its index.
    pub fn update(&mut self, doc_id: usize, new_text: &str) -> io::Result<()> {
        if self.mmap_data.is_some() {
            self.materialize_mmap();
        }

        let id = doc_id as u32;
        assert!(
            id < self.num_docs,
            "doc_id {} out of range (num_docs={})",
            doc_id,
            self.num_docs
        );

        // Remove old postings for this doc
        let old_dl = self.doc_lengths[id as usize];
        self.total_tokens -= old_dl as u64;
        for (term_id, postings) in self.postings.iter_mut().enumerate() {
            let old_len = postings.len();
            postings.retain(|&(did, _)| did != id);
            if postings.len() < old_len {
                self.doc_freqs[term_id] -= 1;
            }
        }

        // Re-tokenize and re-index
        let tokens = self.tokenizer.tokenize_owned(new_text);
        let doc_len = tokens.len() as u32;

        let mut tf_map: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *tf_map.entry(token.clone()).or_insert(0) += 1;
        }

        for (token, tf) in tf_map {
            let term_id = self.get_or_create_term(&token);
            let plist = &mut self.postings[term_id as usize];
            let pos = plist.partition_point(|&(did, _)| did < id);
            plist.insert(pos, (id, tf));
            self.doc_freqs[term_id as usize] += 1;
        }

        self.doc_lengths[id as usize] = doc_len;
        self.total_tokens += doc_len as u64;
        self.auto_save()
    }

    /// Get the number of documents.
    pub fn len(&self) -> usize {
        self.num_docs as usize
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.num_docs == 0
    }

    /// Score a query against a list of documents.
    /// Returns one score per document using the same tokenizer and BM25 parameters.
    /// The documents are treated as the corpus for IDF computation.
    pub fn score(&self, query: &str, documents: &[&str]) -> Vec<f32> {
        let n = documents.len();
        if n == 0 {
            return Vec::new();
        }

        // Tokenize documents in parallel, compute TFs and doc lengths
        let pool = capped_pool();
        let tokenized: Vec<(u32, HashMap<String, u32>)> = pool.install(|| {
            documents
                .par_iter()
                .map(|doc| {
                    let tokens = self.tokenizer.tokenize_owned(doc);
                    let dl = tokens.len() as u32;
                    let mut tf_map: HashMap<String, u32> = HashMap::new();
                    for t in tokens {
                        *tf_map.entry(t).or_insert(0) += 1;
                    }
                    (dl, tf_map)
                })
                .collect()
        });

        let mut doc_tokens: Vec<HashMap<String, u32>> = Vec::with_capacity(n);
        let mut doc_lens: Vec<u32> = Vec::with_capacity(n);
        let mut total_tokens = 0u64;
        for (dl, tf_map) in tokenized {
            total_tokens += dl as u64;
            doc_lens.push(dl);
            doc_tokens.push(tf_map);
        }
        let avgdl = total_tokens as f32 / n as f32;

        // Compute DF per term (across the provided documents)
        let mut df_map: HashMap<&str, u32> = HashMap::new();
        for tf_map in &doc_tokens {
            for term in tf_map.keys() {
                *df_map.entry(term.as_str()).or_insert(0) += 1;
            }
        }

        // Tokenize query
        let query_tokens = self.tokenizer.tokenize_owned(query);
        let mut seen: HashSet<&str> = HashSet::new();
        let query_terms: Vec<&str> = query_tokens
            .iter()
            .filter(|t| seen.insert(t.as_str()))
            .map(|t| t.as_str())
            .collect();

        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl,
        };

        // Score each document
        let n_u32 = n as u32;
        let mut scores = Vec::with_capacity(n);
        for (i, tf_map) in doc_tokens.iter().enumerate() {
            let dl = doc_lens[i];
            let mut total = 0.0f32;
            for &qt in &query_terms {
                if let Some(&tf) = tf_map.get(qt) {
                    let df = *df_map.get(qt).unwrap_or(&0);
                    let idf_val = scoring::idf(self.method, n_u32, df);
                    total += scoring::score(self.method, tf, dl, &params, idf_val);
                }
            }
            scores.push(total);
        }
        scores
    }

    /// Score multiple queries against their respective document lists.
    /// `queries[i]` is scored against `documents[i]`.
    pub fn score_batch(&self, queries: &[&str], documents: &[&[&str]]) -> Vec<Vec<f32>> {
        let pool = capped_pool();
        pool.install(|| {
            queries
                .par_iter()
                .zip(documents.par_iter())
                .map(|(q, docs)| self.score(q, docs))
                .collect()
        })
    }

    // --- Internal helpers ---

    fn get_or_create_term(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.vocab.get(token) {
            id
        } else {
            let id = self.postings.len() as u32;
            self.vocab.insert(token.to_string(), id);
            self.postings.push(Vec::new());
            self.doc_freqs.push(0);
            id
        }
    }

    /// Get document length for a doc_id.
    fn get_doc_length(&self, doc_id: u32) -> u32 {
        if let Some(ref mmap) = self.mmap_data {
            mmap.get_doc_length(doc_id)
        } else {
            *self.doc_lengths.get(doc_id as usize).unwrap_or(&0)
        }
    }

    /// Iterate over postings for a term.
    fn for_each_posting<F: FnMut(u32, u32)>(&self, term_id: u32, mut f: F) {
        if let Some(ref mmap) = self.mmap_data {
            mmap.for_each_posting(term_id, &mut f);
        } else if let Some(postings) = self.postings.get(term_id as usize) {
            for &(doc_id, tf) in postings {
                f(doc_id, tf);
            }
        }
    }

    /// Look up term frequency for a specific (term_id, doc_id) via binary search.
    #[inline]
    fn get_tf(&self, term_id: u32, doc_id: u32) -> Option<u32> {
        if let Some(ref mmap) = self.mmap_data {
            mmap.get_tf(term_id, doc_id)
        } else if let Some(postings) = self.postings.get(term_id as usize) {
            postings
                .binary_search_by_key(&doc_id, |&(did, _)| did)
                .ok()
                .map(|idx| postings[idx].1)
        } else {
            None
        }
    }

    /// Convert mmap-backed data to in-memory vectors (needed before mutation).
    fn materialize_mmap(&mut self) {
        if let Some(mmap) = self.mmap_data.take() {
            let num_terms = self.vocab.len();
            self.postings = Vec::with_capacity(num_terms);
            for term_id in 0..num_terms as u32 {
                let mut entries = Vec::new();
                mmap.for_each_posting(term_id, &mut |doc_id, tf| {
                    entries.push((doc_id, tf));
                });
                self.postings.push(entries);
            }
            self.doc_lengths = mmap.all_doc_lengths();
        }
    }

    // --- Accessors for storage module ---

    pub(crate) fn get_postings(&self) -> &Vec<Vec<(u32, u32)>> {
        &self.postings
    }

    pub(crate) fn get_doc_lengths_slice(&self) -> &[u32] {
        &self.doc_lengths
    }

    pub(crate) fn get_mmap_data(&self) -> Option<&MmapData> {
        self.mmap_data.as_ref()
    }

    pub(crate) fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    pub(crate) fn get_total_tokens(&self) -> u64 {
        self.total_tokens
    }

    pub(crate) fn get_num_docs(&self) -> u32 {
        self.num_docs
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_internals(
        &mut self,
        vocab: HashMap<String, u32>,
        doc_lengths: Vec<u32>,
        postings: Vec<Vec<(u32, u32)>>,
        total_tokens: u64,
        num_docs: u32,
    ) {
        self.doc_freqs = postings.iter().map(|p| p.len() as u32).collect();
        self.vocab = vocab;
        self.doc_lengths = doc_lengths;
        self.postings = postings;
        self.total_tokens = total_tokens;
        self.num_docs = num_docs;
    }

    pub(crate) fn set_mmap_internals(
        &mut self,
        vocab: HashMap<String, u32>,
        total_tokens: u64,
        num_docs: u32,
        mmap_data: MmapData,
    ) {
        let num_terms = vocab.len() as u32;
        self.doc_freqs = (0..num_terms).map(|t| mmap_data.posting_count(t)).collect();
        self.vocab = vocab;
        self.total_tokens = total_tokens;
        self.num_docs = num_docs;
        self.mmap_data = Some(mmap_data);
    }

    /// Create a multi-GPU search index. Replicates the index across all available GPUs.
    #[cfg(feature = "cuda")]
    pub fn to_multi_gpu_search_index(
        &self,
    ) -> Result<crate::multi_gpu::MultiGpuSearchIndex, String> {
        let postings_ref: Vec<Vec<(u32, u32)>>;
        let postings = if let Some(ref mmap) = self.mmap_data {
            postings_ref = (0..self.vocab.len() as u32)
                .map(|t| {
                    let mut entries = Vec::new();
                    mmap.for_each_posting(t, &mut |doc_id, tf| entries.push((doc_id, tf)));
                    entries
                })
                .collect();
            &postings_ref
        } else {
            &self.postings
        };
        let doc_lengths = if let Some(ref mmap) = self.mmap_data {
            mmap.all_doc_lengths()
        } else {
            self.doc_lengths.clone()
        };
        crate::multi_gpu::MultiGpuSearchIndex::from_index(postings, &doc_lengths, self.num_docs)
    }

    /// Create a GPU search index for fast search. Call once after indexing.
    #[cfg(feature = "cuda")]
    pub fn to_gpu_search_index(&self) -> Result<crate::cuda::GpuSearchIndex, String> {
        let ctx =
            crate::cuda::get_global_context().ok_or_else(|| "CUDA not available".to_string())?;

        // If mmap-backed, need to materialize posting lists
        let postings_ref: Vec<Vec<(u32, u32)>>;
        let postings = if let Some(ref mmap) = self.mmap_data {
            postings_ref = (0..self.vocab.len() as u32)
                .map(|t| {
                    let mut entries = Vec::new();
                    mmap.for_each_posting(t, &mut |doc_id, tf| entries.push((doc_id, tf)));
                    entries
                })
                .collect();
            &postings_ref
        } else {
            &self.postings
        };

        let doc_lengths = if let Some(ref mmap) = self.mmap_data {
            mmap.all_doc_lengths()
        } else {
            self.doc_lengths.clone()
        };

        crate::cuda::GpuSearchIndex::from_index(&ctx, postings, &doc_lengths, self.num_docs)
    }

    /// Search using GPU-resident index. Much faster than CPU search for large indices.
    #[cfg(feature = "cuda")]
    pub fn search_gpu(
        &self,
        gpu_index: &mut crate::cuda::GpuSearchIndex,
        query: &str,
        k: usize,
    ) -> Vec<SearchResult> {
        if self.num_docs == 0 {
            return Vec::new();
        }

        let ctx = match crate::cuda::get_global_context() {
            Some(c) => c,
            None => return self.search(query, k),
        };

        let query_tokens = self.tokenizer.tokenize_owned(query);
        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl: self.total_tokens as f32 / self.num_docs as f32,
        };

        // Build (term_id, idf) pairs for query
        let mut seen = std::collections::HashSet::new();
        let mut query_terms: Vec<(u32, f32)> = Vec::new();
        for token in &query_tokens {
            if let Some(&term_id) = self.vocab.get(token.as_str()) {
                if seen.insert(term_id) {
                    let df = self.doc_freqs.get(term_id as usize).copied().unwrap_or(0);
                    if df > 0 {
                        query_terms.push((term_id, scoring::idf(self.method, self.num_docs, df)));
                    }
                }
            }
        }

        match gpu_index.search(&ctx, &query_terms, params.k1, params.b, params.avgdl, k) {
            Ok(results) => results
                .into_iter()
                .map(|(doc_id, score)| SearchResult {
                    index: doc_id as usize,
                    score,
                })
                .collect(),
            Err(e) => {
                eprintln!("[bm25x] GPU search failed: {}, falling back to CPU", e);
                self.search(query, k)
            }
        }
    }
}

impl Default for BM25 {
    fn default() -> Self {
        Self::new(Method::Lucene, 1.5, 0.75, 0.5, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_search() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let ids = index
            .add(&[
                "the quick brown fox jumps over the lazy dog",
                "a fast brown car drives over the bridge",
                "the fox is quick and clever",
            ])
            .unwrap();
        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(index.len(), 3);

        let results = index.search("quick fox", 10);
        assert!(!results.is_empty());
        assert!(results[0].index == 0 || results[0].index == 2);
    }

    #[test]
    fn test_delete_compacts() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello world", "foo bar", "hello foo"]).unwrap();
        assert_eq!(index.len(), 3);

        // Delete doc 0 ("hello world")
        index.delete(&[0]).unwrap();
        assert_eq!(index.len(), 2);

        // Old doc 1 ("foo bar") is now doc 0
        // Old doc 2 ("hello foo") is now doc 1
        let results = index.search("foo", 10);
        assert_eq!(results.len(), 2);
        let ids: Vec<usize> = results.iter().map(|r| r.index).collect();
        assert!(ids.contains(&0)); // was "foo bar"
        assert!(ids.contains(&1)); // was "hello foo"
    }

    #[test]
    fn test_delete_middle() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["alpha", "beta", "gamma", "delta"]).unwrap();

        // Delete doc 1 ("beta"): [alpha, gamma, delta]
        index.delete(&[1]).unwrap();
        assert_eq!(index.len(), 3);

        let results = index.search("gamma", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 1); // gamma shifted from 2 to 1

        let results = index.search("delta", 10);
        assert_eq!(results[0].index, 2); // delta shifted from 3 to 2
    }

    #[test]
    fn test_delete_multiple() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["a", "b", "c", "d", "e"]).unwrap();

        // Delete docs 1 and 3: [a, c, e]
        index.delete(&[1, 3]).unwrap();
        assert_eq!(index.len(), 3);

        let results = index.search("c", 10);
        assert_eq!(results[0].index, 1); // c shifted from 2 to 1

        let results = index.search("e", 10);
        assert_eq!(results[0].index, 2); // e shifted from 4 to 2
    }

    #[test]
    fn test_update() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello world", "foo bar"]).unwrap();

        index.update(0, "goodbye universe").unwrap();

        let results = index.search("hello", 10);
        assert!(results.is_empty() || results.iter().all(|r| r.index != 0));

        let results = index.search("goodbye", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_empty_search() {
        let index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let results = index.search("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_streaming_add() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let ids1 = index.add(&["first document"]).unwrap();
        let ids2 = index.add(&["second document"]).unwrap();
        assert_eq!(ids1, vec![0]);
        assert_eq!(ids2, vec![1]);
        assert_eq!(index.len(), 2);

        let results = index.search("first", 10);
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_all_methods() {
        for method in [
            Method::Lucene,
            Method::Robertson,
            Method::Atire,
            Method::BM25L,
            Method::BM25Plus,
        ] {
            let mut index = BM25::new(method, 1.5, 0.75, 0.5, false);
            index
                .add(&[
                    "the cat sat on the mat",
                    "the dog played in the park",
                    "birds fly over the river",
                    "fish swim in the ocean",
                ])
                .unwrap();
            let results = index.search("cat mat", 10);
            assert!(!results.is_empty(), "{:?} returned no results", method);
            assert_eq!(results[0].index, 0, "{:?} ranked wrong doc first", method);
        }
    }

    #[test]
    fn test_search_filtered_basic() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index
            .add(&[
                "the quick brown fox",  // 0
                "the lazy brown dog",   // 1
                "the quick red car",    // 2
                "the slow brown truck", // 3
            ])
            .unwrap();

        let results = index.search("brown", 10);
        assert_eq!(results.len(), 3);

        let results = index.search_filtered("brown", 10, &[1, 3]);
        assert_eq!(results.len(), 2);
        let ids: Vec<usize> = results.iter().map(|r| r.index).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&0));
    }

    #[test]
    fn test_search_filtered_respects_k() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index
            .add(&[
                "apple banana cherry",   // 0
                "apple date elderberry", // 1
                "apple fig grape",       // 2
                "apple hazelnut ice",    // 3
                "apple jackfruit kiwi",  // 4
            ])
            .unwrap();

        let results = index.search_filtered("apple", 2, &[0, 1, 2, 3]);
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(r.index <= 3);
        }
    }

    #[test]
    fn test_search_filtered_empty_filter() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello world", "foo bar"]).unwrap();

        let results = index.search_filtered("hello", 10, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_filtered_no_overlap() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index
            .add(&[
                "the quick brown fox", // 0
                "the lazy dog",        // 1
            ])
            .unwrap();

        let results = index.search_filtered("fox", 10, &[1]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_filtered_scores_match_unfiltered() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index
            .add(&[
                "rust is fast and safe",    // 0
                "python is slow but easy",  // 1
                "rust and python together", // 2
            ])
            .unwrap();

        let unfiltered = index.search("rust", 10);
        let filtered = index.search_filtered("rust", 10, &[0]);

        let score_unfiltered = unfiltered.iter().find(|r| r.index == 0).unwrap().score;
        let score_filtered = filtered.iter().find(|r| r.index == 0).unwrap().score;
        assert!(
            (score_unfiltered - score_filtered).abs() < 1e-6,
            "scores differ: {} vs {}",
            score_unfiltered,
            score_filtered
        );
    }

    #[test]
    fn test_delete_then_add() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["alpha", "beta", "gamma"]).unwrap();
        index.delete(&[1]).unwrap(); // remove "beta", now [alpha, gamma]
        assert_eq!(index.len(), 2);

        let ids = index.add(&["delta"]).unwrap();
        assert_eq!(ids, vec![2]); // appended at end
        assert_eq!(index.len(), 3);

        let results = index.search("delta", 10);
        assert_eq!(results[0].index, 2);
    }

    #[test]
    fn test_score_basic() {
        let index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let scores = index.score("fox", &["the quick brown fox", "lazy dog", "fox fox fox"]);
        assert_eq!(scores.len(), 3);
        assert!(scores[0] > 0.0); // "fox" appears
        assert_eq!(scores[1], 0.0); // no match
        assert!(scores[2] > scores[0]); // more "fox" occurrences
    }

    #[test]
    fn test_score_empty() {
        let index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        assert!(index.score("hello", &[]).is_empty());
    }

    #[test]
    fn test_score_batch() {
        let index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let docs1: &[&str] = &["the cat", "the dog"];
        let docs2: &[&str] = &["rust lang", "python lang", "go lang"];
        let results = index.score_batch(&["cat", "rust"], &[docs1, docs2]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 3);
        assert!(results[0][0] > 0.0); // "cat" matches "the cat"
        assert!(results[1][0] > 0.0); // "rust" matches "rust lang"
    }

    /// Helper: assert score() and index+search produce identical scores.
    fn assert_scores_match(
        method: Method,
        k1: f32,
        b: f32,
        delta: f32,
        use_stopwords: bool,
        query: &str,
        docs: &[&str],
    ) {
        let scorer = BM25::new(method, k1, b, delta, use_stopwords);
        let direct = scorer.score(query, docs);

        let mut index = BM25::new(method, k1, b, delta, use_stopwords);
        index.add(docs).unwrap();

        // Check every document — including zero-scoring ones
        for (i, &ds) in direct.iter().enumerate() {
            let indexed_score = index
                .search(query, docs.len())
                .iter()
                .find(|r| r.index == i)
                .map(|r| r.score)
                .unwrap_or(0.0);
            assert!(
                (ds - indexed_score).abs() < 1e-6,
                "{:?} doc {}: score()={} != search()={} (query={:?})",
                method,
                i,
                ds,
                indexed_score,
                query
            );
        }
    }

    #[test]
    fn test_score_matches_search_lucene() {
        assert_scores_match(
            Method::Lucene,
            1.5,
            0.75,
            0.5,
            false,
            "beta gamma",
            &[
                "alpha beta gamma",
                "beta gamma delta",
                "gamma delta epsilon",
            ],
        );
    }

    #[test]
    fn test_score_matches_search_all_methods() {
        let docs = &[
            "the quick brown fox jumps over the lazy dog",
            "a brown dog chased the fox",
            "quick red car on the highway",
            "lazy sleeping cat on the mat",
        ];
        for method in [
            Method::Lucene,
            Method::Robertson,
            Method::Atire,
            Method::BM25L,
            Method::BM25Plus,
        ] {
            assert_scores_match(method, 1.5, 0.75, 0.5, false, "brown fox", docs);
            assert_scores_match(method, 1.5, 0.75, 0.5, false, "lazy", docs);
            assert_scores_match(method, 1.5, 0.75, 0.5, false, "quick brown fox", docs);
            assert_scores_match(method, 1.5, 0.75, 0.5, false, "nonexistent", docs);
        }
    }

    #[test]
    fn test_score_matches_search_with_stopwords() {
        let docs = &[
            "the quick brown fox",
            "a lazy brown dog",
            "the fox is quick and clever",
        ];
        // With stopwords enabled, "the" and "is" are removed
        assert_scores_match(Method::Lucene, 1.5, 0.75, 0.5, true, "the quick fox", docs);
        assert_scores_match(Method::Lucene, 1.5, 0.75, 0.5, true, "brown", docs);
    }

    #[test]
    fn test_score_matches_search_custom_params() {
        let docs = &["rust is fast", "python is easy", "rust and python together"];
        // Different k1, b values
        assert_scores_match(Method::Lucene, 2.0, 0.5, 0.5, false, "rust", docs);
        assert_scores_match(Method::Atire, 1.2, 0.9, 0.5, false, "rust python", docs);
        assert_scores_match(Method::BM25Plus, 1.5, 0.75, 1.0, false, "rust python", docs);
    }

    #[test]
    fn test_score_matches_search_single_doc() {
        assert_scores_match(
            Method::Lucene,
            1.5,
            0.75,
            0.5,
            false,
            "hello",
            &["hello world"],
        );
    }

    #[test]
    fn test_score_matches_search_no_match() {
        let scorer = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let scores = scorer.score("xyz", &["alpha beta", "gamma delta"]);
        assert!(scores.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_score_matches_search_duplicate_query_terms() {
        // "fox fox" should produce the same score as "fox"
        let docs = &["the quick brown fox", "lazy dog"];
        let scorer = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let scores_single = scorer.score("fox", docs);
        let scores_dup = scorer.score("fox fox", docs);
        for i in 0..docs.len() {
            assert!(
                (scores_single[i] - scores_dup[i]).abs() < 1e-6,
                "duplicate query terms changed score for doc {}: {} vs {}",
                i,
                scores_single[i],
                scores_dup[i]
            );
        }
    }

    #[test]
    fn test_score_batch_matches_individual() {
        let scorer = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let docs1: &[&str] = &["the cat sat", "the dog ran"];
        let docs2: &[&str] = &["rust lang", "python lang", "go lang"];

        let batch = scorer.score_batch(&["cat", "rust"], &[docs1, docs2]);
        let individual1 = scorer.score("cat", docs1);
        let individual2 = scorer.score("rust", docs2);

        for i in 0..docs1.len() {
            assert!(
                (batch[0][i] - individual1[i]).abs() < 1e-6,
                "batch[0][{}]={} != individual={}",
                i,
                batch[0][i],
                individual1[i]
            );
        }
        for i in 0..docs2.len() {
            assert!(
                (batch[1][i] - individual2[i]).abs() < 1e-6,
                "batch[1][{}]={} != individual={}",
                i,
                batch[1][i],
                individual2[i]
            );
        }
    }

    #[test]
    fn test_default_uses_best_config() {
        // BM25::default() must use the best configuration we benchmarked:
        // Lucene, k1=1.5, b=0.75, delta=0.5, UnicodeStem, stopwords=true
        let index = BM25::default();
        assert_eq!(index.method, Method::Lucene);
        assert_eq!(index.k1, 1.5);
        assert_eq!(index.b, 0.75);
        assert_eq!(index.delta, 0.5);
    }

    #[test]
    fn test_default_applies_stemming() {
        // Default must stem: "running" → "run"
        let index = BM25::default();
        let scores = index.score("run", &["running quickly", "sleeping deeply"]);
        assert!(scores[0] > 0.0, "stemming should match 'run' to 'running'");
        assert_eq!(scores[1], 0.0);
    }

    #[test]
    fn test_default_applies_unicode_folding() {
        // Default must fold diacritics: "cafe" matches "café"
        let index = BM25::default();
        let scores = index.score("cafe", &["café latte", "green tea"]);
        assert!(
            scores[0] > 0.0,
            "unicode folding should match 'cafe' to 'café'"
        );
        assert_eq!(scores[1], 0.0);
    }

    #[test]
    fn test_default_applies_stopwords() {
        // Default must remove stopwords: "the" alone should not match
        let mut index = BM25::default();
        index.add(&["the quick fox", "lazy dog"]).unwrap();
        // "the" is a stopword, so querying "the" should return no results
        let results = index.search("the", 10);
        assert!(results.is_empty(), "stopwords should be filtered");
    }

    #[test]
    fn test_default_scores_match_explicit_best_config() {
        // BM25::default() must produce identical scores to the explicit best config
        let docs = &[
            "the café résumé running cancellation",
            "quick brown fox jumping",
            "lazy sleeping dog",
        ];
        let query = "running fox café";

        let default_index = BM25::default();
        let explicit_index = BM25::with_tokenizer(
            Method::Lucene,
            1.5,
            0.75,
            0.5,
            TokenizerMode::UnicodeStem,
            true,
        );

        let default_scores = default_index.score(query, docs);
        let explicit_scores = explicit_index.score(query, docs);

        for i in 0..docs.len() {
            assert!(
                (default_scores[i] - explicit_scores[i]).abs() < 1e-6,
                "doc {}: default={} != explicit={}",
                i,
                default_scores[i],
                explicit_scores[i]
            );
        }
    }
}
