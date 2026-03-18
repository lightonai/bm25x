use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use bm25rs_core::Method;

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
    /// If `index` is provided, the index is persisted to that directory:
    /// - If the directory already contains a saved index, it is loaded automatically.
    /// - Every mutation (add, delete, update) auto-saves to disk.
    #[new]
    #[pyo3(signature = (index=None, method="lucene", k1=1.5, b=0.75, delta=0.5, use_stopwords=true))]
    fn new(
        index: Option<&str>,
        method: &str,
        k1: f32,
        b: f32,
        delta: f32,
        use_stopwords: bool,
    ) -> PyResult<Self> {
        let m = parse_method(method)?;
        let inner = match index {
            Some(path) => bm25rs_core::BM25::open(path, m, k1, b, delta, use_stopwords).map_err(io_err)?,
            None => bm25rs_core::BM25::new(m, k1, b, delta, use_stopwords),
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
