use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use bm25rs::{BM25Index, Method};

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

#[pyclass]
struct PyBM25Index {
    inner: BM25Index,
}

#[pymethods]
impl PyBM25Index {
    #[new]
    #[pyo3(signature = (method="lucene", k1=1.5, b=0.75, delta=0.5, use_stopwords=true))]
    fn new(method: &str, k1: f32, b: f32, delta: f32, use_stopwords: bool) -> PyResult<Self> {
        let m = parse_method(method)?;
        Ok(PyBM25Index {
            inner: BM25Index::new(m, k1, b, delta, use_stopwords),
        })
    }

    /// Add documents to the index. Returns list of assigned indices.
    fn add(&mut self, documents: Vec<String>) -> Vec<usize> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        self.inner.add(&refs)
    }

    /// Search the index. Returns list of (index, score) tuples.
    fn search(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        self.inner
            .search(query, k)
            .into_iter()
            .map(|r| (r.index, r.score))
            .collect()
    }

    /// Delete documents by their indices.
    fn delete(&mut self, doc_ids: Vec<usize>) {
        self.inner.delete(&doc_ids);
    }

    /// Update a document's text at the given index.
    fn update(&mut self, doc_id: usize, new_text: &str) {
        self.inner.update(doc_id, new_text);
    }

    /// Save the index to a directory.
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| PyValueError::new_err(format!("Save failed: {}", e)))
    }

    /// Load an index from a directory.
    #[staticmethod]
    #[pyo3(signature = (path, mmap=false))]
    fn load(path: &str, mmap: bool) -> PyResult<Self> {
        let inner = BM25Index::load(path, mmap)
            .map_err(|e| PyValueError::new_err(format!("Load failed: {}", e)))?;
        Ok(PyBM25Index { inner })
    }

    /// Number of active documents.
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pymodule]
fn bm25rs_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBM25Index>()?;
    Ok(())
}
