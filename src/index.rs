use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::scoring::{self, Method, ScoringParams};
use crate::storage::MmapData;
use crate::tokenizer::Tokenizer;

/// A scored document result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The document index (0-based, as returned by `add`).
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
pub struct BM25Index {
    // Scoring parameters
    pub k1: f32,
    pub b: f32,
    pub delta: f32,
    pub method: Method,

    // Inverted index: term_id -> Vec<(doc_id, tf)>
    postings: Vec<Vec<(u32, u32)>>,

    // Cached document frequency per term (avoids counting on every query)
    doc_freqs: Vec<u32>,

    // Document metadata
    doc_lengths: Vec<u32>,
    total_tokens: u64,
    num_docs: u32,

    // Deleted documents (tombstones)
    deleted: HashSet<u32>,

    // Vocabulary: token string -> term_id
    vocab: HashMap<String, u32>,

    // Next document ID to assign
    next_doc_id: u32,

    // Tokenizer
    tokenizer: Tokenizer,

    // Mmap backing (if loaded from disk)
    mmap_data: Option<MmapData>,
}

impl BM25Index {
    /// Create a new empty index.
    pub fn new(method: Method, k1: f32, b: f32, delta: f32, use_stopwords: bool) -> Self {
        BM25Index {
            k1,
            b,
            delta,
            method,
            postings: Vec::new(),
            doc_freqs: Vec::new(),
            doc_lengths: Vec::new(),
            total_tokens: 0,
            num_docs: 0,
            deleted: HashSet::new(),
            vocab: HashMap::new(),
            next_doc_id: 0,
            tokenizer: Tokenizer::new(use_stopwords),
            mmap_data: None,
        }
    }

    /// Add documents to the index. Returns the assigned document indices.
    pub fn add(&mut self, documents: &[&str]) -> Vec<usize> {
        // Mmap index is read-only for postings; convert to writable if needed
        if self.mmap_data.is_some() {
            self.materialize_mmap();
        }

        let mut ids = Vec::with_capacity(documents.len());

        for doc in documents {
            let doc_id = self.next_doc_id;
            self.next_doc_id += 1;

            let tokens = self.tokenizer.tokenize_owned(doc);
            let doc_len = tokens.len() as u32;

            // Count term frequencies
            let mut tf_map: HashMap<String, u32> = HashMap::new();
            for token in &tokens {
                *tf_map.entry(token.clone()).or_insert(0) += 1;
            }

            // Update inverted index and doc_freqs
            for (token, tf) in tf_map {
                let term_id = self.get_or_create_term(&token);
                self.postings[term_id as usize].push((doc_id, tf));
                self.doc_freqs[term_id as usize] += 1;
            }

            // Ensure doc_lengths is large enough
            if doc_id as usize >= self.doc_lengths.len() {
                self.doc_lengths.resize(doc_id as usize + 1, 0);
            }
            self.doc_lengths[doc_id as usize] = doc_len;
            self.total_tokens += doc_len as u64;
            self.num_docs += 1;

            ids.push(doc_id as usize);
        }

        ids
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
        let has_deleted = !self.deleted.is_empty();

        // Use a flat score array indexed by doc_id for O(1) accumulation
        let mut scores = vec![0.0f32; self.next_doc_id as usize];
        let mut touched = Vec::new(); // track which doc_ids got scores

        // Deduplicate query tokens to avoid redundant work
        let mut seen_terms: HashSet<u32> = HashSet::new();

        for token in &query_tokens {
            let term_id = match self.vocab.get(token.as_str()) {
                Some(&id) => id,
                None => continue,
            };
            if !seen_terms.insert(term_id) {
                continue;
            }

            let df = if !has_deleted {
                self.doc_freqs.get(term_id as usize).copied().unwrap_or(0)
            } else {
                self.doc_freq_fast(term_id, true)
            };
            if df == 0 {
                continue;
            }

            let idf_val = scoring::idf(self.method, self.num_docs, df);

            self.for_each_posting(term_id, |doc_id, tf| {
                if has_deleted && self.deleted.contains(&doc_id) {
                    return;
                }
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

    /// Search restricted to a subset of document IDs (pre-filtering).
    /// Only documents whose index is in `allowed_ids` will be scored.
    /// IDF is computed from global corpus stats so scores stay comparable.
    ///
    /// Uses a doc-centric approach: iterates allowed IDs and looks up each doc's
    /// TF via binary search on posting lists. Cost is O(|allowed| * |query_terms| * log(posting_len))
    /// which is much faster than scanning full posting lists when |allowed| is small.
    pub fn search_filtered(
        &self,
        query: &str,
        k: usize,
        allowed_ids: &[usize],
    ) -> Vec<SearchResult> {
        if self.num_docs == 0 || allowed_ids.is_empty() {
            return Vec::new();
        }

        let query_tokens = self.tokenizer.tokenize_owned(query);
        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl: self.total_tokens as f32 / self.num_docs as f32,
        };
        let has_deleted = !self.deleted.is_empty();

        // Resolve query tokens to term_ids + IDF values
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
            let df = if !has_deleted {
                self.doc_freqs.get(term_id as usize).copied().unwrap_or(0)
            } else {
                self.doc_freq_fast(term_id, true)
            };
            if df == 0 {
                continue;
            }
            query_terms.push((term_id, scoring::idf(self.method, self.num_docs, df)));
        }
        if query_terms.is_empty() {
            return Vec::new();
        }

        // Doc-centric scoring: for each allowed doc, look up TF via binary search.
        // Cost: O(|allowed| * |query_terms| * log(posting_len))
        let mut heap = BinaryHeap::with_capacity(k + 1);
        for &doc_idx in allowed_ids {
            let doc_id = doc_idx as u32;
            if doc_id >= self.next_doc_id {
                continue;
            }
            if has_deleted && self.deleted.contains(&doc_id) {
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
    pub fn delete(&mut self, doc_ids: &[usize]) {
        for &id in doc_ids {
            let doc_id = id as u32;
            if doc_id < self.next_doc_id && !self.deleted.contains(&doc_id) {
                self.deleted.insert(doc_id);
                let dl = self.get_doc_length(doc_id);
                self.total_tokens = self.total_tokens.saturating_sub(dl as u64);
                self.num_docs = self.num_docs.saturating_sub(1);
                // Note: doc_freqs become stale when there are deletions.
                // We handle this in search by falling back to counting when deleted is non-empty.
            }
        }
    }

    /// Update a document's text. The document keeps its original index.
    pub fn update(&mut self, doc_id: usize, new_text: &str) {
        if self.mmap_data.is_some() {
            self.materialize_mmap();
        }

        let id = doc_id as u32;

        // Remove old postings for this doc
        if !self.deleted.contains(&id) {
            let old_dl = self.get_doc_length(id);
            self.total_tokens = self.total_tokens.saturating_sub(old_dl as u64);
            // Remove from all posting lists, updating doc_freqs
            for (term_id, postings) in self.postings.iter_mut().enumerate() {
                let old_len = postings.len();
                postings.retain(|&(did, _)| did != id);
                if postings.len() < old_len {
                    self.doc_freqs[term_id] -= 1;
                }
            }
        } else {
            // Was deleted, re-adding
            self.deleted.remove(&id);
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
            // Insert sorted to maintain binary-search invariant
            let pos = plist.partition_point(|&(did, _)| did < id);
            plist.insert(pos, (id, tf));
            self.doc_freqs[term_id as usize] += 1;
        }

        if id as usize >= self.doc_lengths.len() {
            self.doc_lengths.resize(id as usize + 1, 0);
        }
        self.doc_lengths[id as usize] = doc_len;
        self.total_tokens += doc_len as u64;

        // If it was previously deleted, increment num_docs
        // (already handled above by removing from deleted set)
        if !self.deleted.contains(&id) {
            // Only increment if we removed it from deleted above
            // We need to check differently: if it was in `deleted` before this call
            // Actually, we handle this: if deleted contained it, we removed it and
            // didn't decrement num_docs (it was already decremented in delete()).
            // So we need to re-increment.
        }
        // Simpler: just recount. For correctness, let's recalculate.
        // Actually the logic is: the doc existed before and was not deleted (we subtracted its length),
        // or it was deleted (we removed from deleted set). Either way, after update, it should be active.
        // num_docs should reflect active docs. Let's just recalculate from state.
        self.num_docs = self.next_doc_id - self.deleted.len() as u32;
    }

    /// Get the number of active (non-deleted) documents.
    pub fn len(&self) -> usize {
        self.num_docs as usize
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.num_docs == 0
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

    /// Get document frequency for a term, with pre-computed has_deleted flag.
    #[inline]
    fn doc_freq_fast(&self, term_id: u32, has_deleted: bool) -> u32 {
        if !has_deleted {
            // No deletions: df = posting list length
            if let Some(ref mmap) = self.mmap_data {
                return mmap.posting_count(term_id);
            }
            return self
                .postings
                .get(term_id as usize)
                .map_or(0, |p| p.len() as u32);
        }
        let mut count = 0u32;
        self.for_each_posting(term_id, |doc_id, _| {
            if !self.deleted.contains(&doc_id) {
                count += 1;
            }
        });
        count
    }

    /// Get document length for a doc_id.
    fn get_doc_length(&self, doc_id: u32) -> u32 {
        if let Some(ref mmap) = self.mmap_data {
            mmap.get_doc_length(doc_id)
        } else {
            *self.doc_lengths.get(doc_id as usize).unwrap_or(&0)
        }
    }

    /// Iterate over postings for a term, calling `f(doc_id, tf)` for each entry.
    fn for_each_posting<F: FnMut(u32, u32)>(&self, term_id: u32, mut f: F) {
        if let Some(ref mmap) = self.mmap_data {
            mmap.for_each_posting(term_id, &mut f);
        } else if let Some(postings) = self.postings.get(term_id as usize) {
            for &(doc_id, tf) in postings {
                f(doc_id, tf);
            }
        }
    }

    /// Look up term frequency for a specific (term_id, doc_id) pair.
    /// Uses binary search on posting lists (which are sorted by doc_id).
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
            // doc_freqs already populated from set_mmap_internals
        }
    }

    // --- Accessors for storage module ---

    pub(crate) fn get_postings(&self) -> &Vec<Vec<(u32, u32)>> {
        &self.postings
    }

    pub(crate) fn get_doc_lengths_slice(&self) -> &[u32] {
        &self.doc_lengths
    }

    pub(crate) fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    pub(crate) fn get_deleted(&self) -> &HashSet<u32> {
        &self.deleted
    }

    pub(crate) fn get_total_tokens(&self) -> u64 {
        self.total_tokens
    }

    pub(crate) fn get_next_doc_id(&self) -> u32 {
        self.next_doc_id
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_internals(
        &mut self,
        vocab: HashMap<String, u32>,
        deleted: HashSet<u32>,
        doc_lengths: Vec<u32>,
        postings: Vec<Vec<(u32, u32)>>,
        total_tokens: u64,
        num_docs: u32,
        next_doc_id: u32,
    ) {
        self.doc_freqs = postings.iter().map(|p| p.len() as u32).collect();
        self.vocab = vocab;
        self.deleted = deleted;
        self.doc_lengths = doc_lengths;
        self.postings = postings;
        self.total_tokens = total_tokens;
        self.num_docs = num_docs;
        self.next_doc_id = next_doc_id;
    }

    pub(crate) fn set_mmap_internals(
        &mut self,
        vocab: HashMap<String, u32>,
        deleted: HashSet<u32>,
        total_tokens: u64,
        num_docs: u32,
        next_doc_id: u32,
        mmap_data: MmapData,
    ) {
        let num_terms = vocab.len() as u32;
        self.doc_freqs = (0..num_terms).map(|t| mmap_data.posting_count(t)).collect();
        self.vocab = vocab;
        self.deleted = deleted;
        self.total_tokens = total_tokens;
        self.num_docs = num_docs;
        self.next_doc_id = next_doc_id;
        self.mmap_data = Some(mmap_data);
    }
}

impl Default for BM25Index {
    fn default() -> Self {
        Self::new(Method::Lucene, 1.5, 0.75, 0.5, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_search() {
        let mut index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let ids = index.add(&[
            "the quick brown fox jumps over the lazy dog",
            "a fast brown car drives over the bridge",
            "the fox is quick and clever",
        ]);
        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(index.len(), 3);

        let results = index.search("quick fox", 10);
        assert!(!results.is_empty());
        // Doc 0 and 2 mention both "quick" and "fox"
        assert!(results[0].index == 0 || results[0].index == 2);
    }

    #[test]
    fn test_delete() {
        let mut index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello world", "foo bar", "hello foo"]);
        assert_eq!(index.len(), 3);

        index.delete(&[0]);
        assert_eq!(index.len(), 2);

        let results = index.search("hello", 10);
        // Should only find doc 2, not doc 0
        assert!(results.iter().all(|r| r.index != 0));
    }

    #[test]
    fn test_update() {
        let mut index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello world", "foo bar"]);

        // Update doc 0 to no longer contain "hello"
        index.update(0, "goodbye universe");

        let results = index.search("hello", 10);
        assert!(results.is_empty() || results.iter().all(|r| r.index != 0));

        let results = index.search("goodbye", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_empty_search() {
        let index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let results = index.search("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_streaming_add() {
        let mut index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let ids1 = index.add(&["first document"]);
        let ids2 = index.add(&["second document"]);
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
            let mut index = BM25Index::new(method, 1.5, 0.75, 0.5, false);
            // Use enough docs so Robertson IDF doesn't clamp all terms to 0
            index.add(&[
                "the cat sat on the mat",
                "the dog played in the park",
                "birds fly over the river",
                "fish swim in the ocean",
            ]);
            let results = index.search("cat mat", 10);
            assert!(!results.is_empty(), "{:?} returned no results", method);
            assert_eq!(results[0].index, 0, "{:?} ranked wrong doc first", method);
        }
    }

    #[test]
    fn test_search_filtered_basic() {
        let mut index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&[
            "the quick brown fox",  // 0
            "the lazy brown dog",   // 1
            "the quick red car",    // 2
            "the slow brown truck", // 3
        ]);

        // Unfiltered: "brown" matches 0, 1, 3
        let results = index.search("brown", 10);
        assert_eq!(results.len(), 3);

        // Filtered to only {1, 3}: should only return those two
        let results = index.search_filtered("brown", 10, &[1, 3]);
        assert_eq!(results.len(), 2);
        let ids: Vec<usize> = results.iter().map(|r| r.index).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&0));
    }

    #[test]
    fn test_search_filtered_respects_k() {
        let mut index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&[
            "apple banana cherry",   // 0
            "apple date elderberry", // 1
            "apple fig grape",       // 2
            "apple hazelnut ice",    // 3
            "apple jackfruit kiwi",  // 4
        ]);

        // All 5 match "apple", filter to {0,1,2,3}, ask for k=2
        let results = index.search_filtered("apple", 2, &[0, 1, 2, 3]);
        assert_eq!(results.len(), 2);
        // All filtered docs should be in {0,1,2,3}
        for r in &results {
            assert!(r.index <= 3);
        }
    }

    #[test]
    fn test_search_filtered_empty_filter() {
        let mut index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello world", "foo bar"]);

        let results = index.search_filtered("hello", 10, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_filtered_no_overlap() {
        let mut index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&[
            "the quick brown fox", // 0
            "the lazy dog",        // 1
        ]);

        // "fox" only in doc 0, but filter only allows doc 1
        let results = index.search_filtered("fox", 10, &[1]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_filtered_with_deletions() {
        let mut index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&[
            "alpha beta gamma", // 0
            "alpha delta",      // 1
            "alpha epsilon",    // 2
        ]);
        index.delete(&[1]);

        // Filter includes deleted doc 1 — it should still be excluded
        let results = index.search_filtered("alpha", 10, &[0, 1, 2]);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.index != 1));
    }

    #[test]
    fn test_search_filtered_scores_match_unfiltered() {
        let mut index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&[
            "rust is fast and safe",    // 0
            "python is slow but easy",  // 1
            "rust and python together", // 2
        ]);

        // Score for doc 0 should be the same whether we filter or not
        // (IDF uses global stats in both cases)
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
}
