use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use bm25rs_core::{Method, TokenizerMode};

fn parse_method(method: &str) -> PyResult<Method> {
    match method.to_lowercase().as_str() {
        "lucene" => Ok(Method::Lucene),
        "robertson" => Ok(Method::Robertson),
        "atire" => Ok(Method::Atire),
        "bm25l" => Ok(Method::BM25L),
        "bm25+" | "bm25plus" => Ok(Method::BM25Plus),
        _ => Err(PyValueError::new_err(format!("Unknown method: {}", method))),
    }
}

fn parse_tokenizer(tokenizer: &str) -> PyResult<TokenizerMode> {
    match tokenizer.to_lowercase().as_str() {
        "plain" => Ok(TokenizerMode::Plain),
        "unicode" => Ok(TokenizerMode::Unicode),
        "stem" => Ok(TokenizerMode::Stem),
        "unicode_stem" | "unicodestem" => Ok(TokenizerMode::UnicodeStem),
        _ => Err(PyValueError::new_err(format!(
            "Unknown tokenizer: {}. Choose from: plain, unicode, stem, unicode_stem",
            tokenizer
        ))),
    }
}

fn io_err(e: std::io::Error) -> PyErr {
    PyValueError::new_err(format!("{}", e))
}

#[pyclass(name = "BM25")]
struct PyBM25 {
    inner: bm25rs_core::BM25,
}

#[pymethods]
impl PyBM25 {
    /// Create a new index.
    ///
    /// If `index` is provided, the index is persisted to that directory.
    /// `tokenizer` can be: "plain", "unicode", "stem", "unicode_stem" (default).
    #[new]
    #[pyo3(signature = (index=None, method="lucene", k1=1.5, b=0.75, delta=0.5, tokenizer="unicode_stem", use_stopwords=true))]
    fn new(
        index: Option<&str>,
        method: &str,
        k1: f32,
        b: f32,
        delta: f32,
        tokenizer: &str,
        use_stopwords: bool,
    ) -> PyResult<Self> {
        let m = parse_method(method)?;
        let tok = parse_tokenizer(tokenizer)?;
        let inner = match index {
            Some(path) => {
                bm25rs_core::BM25::open(path, m, k1, b, delta, tok, use_stopwords).map_err(io_err)?
            }
            None => bm25rs_core::BM25::with_tokenizer(m, k1, b, delta, tok, use_stopwords),
        };
        Ok(PyBM25 { inner })
    }

    /// Add documents to the index. Returns list of assigned indices.
    fn add(&mut self, documents: Vec<String>) -> PyResult<Vec<usize>> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        self.inner.add(&refs).map_err(io_err)
    }

    /// Search the index. Returns list of (index, score) tuples.
    /// If `subset` is provided, only those document IDs are scored (pre-filtering).
    #[pyo3(signature = (query, k, subset=None))]
    fn search(&self, query: &str, k: usize, subset: Option<Vec<usize>>) -> Vec<(usize, f32)> {
        let results = match subset {
            Some(ids) => self.inner.search_filtered(query, k, &ids),
            None => self.inner.search(query, k),
        };
        results.into_iter().map(|r| (r.index, r.score)).collect()
    }

    /// Delete documents by their indices.
    fn delete(&mut self, doc_ids: Vec<usize>) -> PyResult<()> {
        self.inner.delete(&doc_ids).map_err(io_err)
    }

    /// Update a document's text at the given index.
    fn update(&mut self, doc_id: usize, new_text: &str) -> PyResult<()> {
        self.inner.update(doc_id, new_text).map_err(io_err)
    }

    /// Save the index to a directory (explicit save, useful for in-memory indices).
    fn save(&self, index: &str) -> PyResult<()> {
        self.inner.save(index).map_err(io_err)
    }

    /// Load an index from a directory.
    #[staticmethod]
    #[pyo3(signature = (index, mmap=false))]
    fn load(index: &str, mmap: bool) -> PyResult<Self> {
        let inner = bm25rs_core::BM25::load(index, mmap).map_err(io_err)?;
        Ok(PyBM25 { inner })
    }

    /// Score a query against a list of documents.
    fn score(&self, query: &Bound<'_, PyAny>, documents: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let py = query.py();
        if let Ok(q) = query.extract::<String>() {
            let docs: Vec<String> = documents.extract()?;
            let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
            let scores = self.inner.score(&q, &refs);
            Ok(scores.into_pyobject(py)?.into_any().unbind())
        } else {
            let queries: Vec<String> = query.extract()?;
            let doc_lists: Vec<Vec<String>> = documents.extract()?;
            let q_refs: Vec<&str> = queries.iter().map(|s| s.as_str()).collect();
            let d_refs: Vec<Vec<&str>> = doc_lists
                .iter()
                .map(|dl| dl.iter().map(|s| s.as_str()).collect())
                .collect();
            let d_slices: Vec<&[&str]> = d_refs.iter().map(|v| v.as_slice()).collect();
            let results = self.inner.score_batch(&q_refs, &d_slices);
            Ok(results.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Number of active documents.
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pymodule]
fn bm25rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBM25>()?;
    Ok(())
}
