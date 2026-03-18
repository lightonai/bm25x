use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use bm25x_core::{Method, TokenizerMode};

/// Wrapper for a raw (*const u8, usize) string pointer pair.
/// These are extracted under the GIL, then used after GIL release.
/// Safety: the Python list must stay alive (keeping the strings alive)
/// for the entire duration these are used.
struct RawStr {
    ptr: *const u8,
    len: usize,
}

// Raw pointers are not Send by default, but this is safe because:
// 1. The Python strings are immutable and reference-counted
// 2. The list object keeps all strings alive
// 3. We only read from these pointers, never write
unsafe impl Send for RawStr {}
unsafe impl Sync for RawStr {}

/// Extract raw UTF-8 pointers from a Python list of strings using
/// direct C-API calls. ~60-200ns per item vs ~2500ns for safe PyO3.
///
/// Safety: all items must be `str`. The returned pointers are valid
/// as long as the Python list (and its string elements) are alive.
fn extract_raw_ptrs(list: &Bound<'_, PyList>) -> PyResult<Vec<RawStr>> {
    let len = list.len();
    let mut result = Vec::with_capacity(len);
    unsafe {
        let list_ptr = list.as_ptr();
        for i in 0..len as pyo3::ffi::Py_ssize_t {
            let item = pyo3::ffi::PyList_GET_ITEM(list_ptr, i);
            let mut size: pyo3::ffi::Py_ssize_t = 0;
            let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
            if data.is_null() {
                // Clear Python error state and return our own error
                pyo3::ffi::PyErr_Clear();
                return Err(PyValueError::new_err(format!(
                    "element {} is not a string",
                    i
                )));
            }
            result.push(RawStr {
                ptr: data as *const u8,
                len: size as usize,
            });
        }
    }
    Ok(result)
}

/// Reconstruct &str slices from raw pointers.
/// Safety: pointers must be valid (Python strings still alive).
#[inline]
fn raw_to_strs(ptrs: &[RawStr]) -> Vec<&str> {
    ptrs.iter()
        .map(|r| unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(r.ptr, r.len))
        })
        .collect()
}

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
    inner: bm25x_core::BM25,
    /// If true, CUDA is required — errors are raised instead of silent fallback.
    cuda_required: bool,
    #[cfg(feature = "cuda")]
    gpu_search_index: Option<bm25x_core::cuda::GpuSearchIndex>,
    #[cfg(feature = "cuda")]
    multi_gpu_index: Option<bm25x_core::multi_gpu::MultiGpuSearchIndex>,
}

#[pymethods]
impl PyBM25 {
    /// Create a new index.
    ///
    /// If `index` is provided, the index is persisted to that directory.
    /// `tokenizer` can be: "plain", "unicode", "stem", "unicode_stem" (default).
    /// `cuda`: if True, require CUDA — raises an error if GPU is unavailable.
    ///         if False (default), auto-detect GPU and fall back to CPU silently.
    #[new]
    #[pyo3(signature = (index=None, method="lucene", k1=1.5, b=0.75, delta=0.5, tokenizer="unicode_stem", use_stopwords=true, cuda=false))]
    fn new(
        index: Option<&str>,
        method: &str,
        k1: f32,
        b: f32,
        delta: f32,
        tokenizer: &str,
        use_stopwords: bool,
        cuda: bool,
    ) -> PyResult<Self> {
        // If cuda=True, verify GPU is available immediately
        if cuda && !bm25x_core::is_gpu_available() {
            return Err(PyValueError::new_err(
                "cuda=True but no CUDA GPU is available. \
                 Check that CUDA drivers are installed and a GPU is visible \
                 (CUDA_VISIBLE_DEVICES).",
            ));
        }

        let m = parse_method(method)?;
        let tok = parse_tokenizer(tokenizer)?;
        let inner = match index {
            Some(path) => {
                bm25x_core::BM25::open(path, m, k1, b, delta, tok, use_stopwords).map_err(io_err)?
            }
            None => bm25x_core::BM25::with_tokenizer(m, k1, b, delta, tok, use_stopwords),
        };
        Ok(PyBM25 {
            inner,
            cuda_required: cuda,
            #[cfg(feature = "cuda")]
            gpu_search_index: None,
            #[cfg(feature = "cuda")]
            multi_gpu_index: None,
        })
    }

    /// Upload the index to GPU for fast search. Call once after adding documents.
    /// Single queries use one GPU. Batch queries auto-dispatch across all GPUs.
    #[cfg(feature = "cuda")]
    fn upload_to_gpu(&mut self) -> PyResult<()> {
        self.gpu_search_index = Some(
            self.inner
                .to_gpu_search_index()
                .map_err(PyValueError::new_err)?,
        );
        // Also create multi-GPU index for batch queries
        match self.inner.to_multi_gpu_search_index() {
            Ok(mgpu) => {
                self.multi_gpu_index = Some(mgpu);
            }
            Err(e) => {
                eprintln!("[bm25x] Multi-GPU init failed (batch will use single GPU): {}", e);
            }
        }
        Ok(())
    }

    /// Add documents to the index. Returns list of assigned indices.
    ///
    /// Uses unsafe FFI for ~12x faster string extraction, then releases
    /// the GIL for the entire Rust processing phase.
    fn add(&mut self, py: Python<'_>, documents: &Bound<'_, PyList>) -> PyResult<Vec<usize>> {
        // Phase 1: Extract raw UTF-8 pointers via C-API (~2s for 8.8M docs)
        let ptrs = extract_raw_ptrs(documents)?;

        // Phase 2: Release GIL, reconstruct &str, run Rust indexing (~11s, GIL-free)
        let inner = &mut self.inner;
        py.allow_threads(|| {
            let refs = raw_to_strs(&ptrs);
            inner.add(&refs).map_err(io_err)
        })
    }

    /// Add documents from a newline-delimited bytes blob.
    ///
    /// This is the fastest path: zero-copy buffer extraction (~0.1s for 2.6GB),
    /// then GIL-free Rust processing. Use from Python:
    ///
    /// ```python
    /// index.add_bytes(b"\n".join(s.encode() for s in corpus))
    /// # or if corpus is already List[str]:
    /// index.add_bytes("\n".join(corpus).encode("utf-8"))
    /// ```
    fn add_bytes(&mut self, py: Python<'_>, data: &[u8]) -> PyResult<Vec<usize>> {
        let t0 = std::time::Instant::now();
        let inner = &mut self.inner;
        let result = py.allow_threads(|| {
            let t_split = std::time::Instant::now();
            let docs: Vec<&str> = data
                .split(|&b| b == b'\n')
                .map(|chunk| {
                    std::str::from_utf8(chunk).map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                    })
                })
                .collect::<Result<Vec<&str>, _>>()?;
            let split_time = t_split.elapsed();
            let t_add = std::time::Instant::now();
            let r = inner.add(&docs);
            let add_time = t_add.elapsed();
            if std::env::var("BM25X_PROFILE").is_ok() {
                eprintln!(
                    "[add_bytes] split={:.3}s add={:.3}s docs={}",
                    split_time.as_secs_f64(),
                    add_time.as_secs_f64(),
                    docs.len()
                );
            }
            r
        });
        if std::env::var("BM25X_PROFILE").is_ok() {
            eprintln!("[add_bytes] total={:.3}s", t0.elapsed().as_secs_f64());
        }
        result.map_err(io_err)
    }

    /// Search the index. Accepts a single query string or a list of queries.
    ///
    /// - Single query: `search("fox", k=10)` → `[(doc_id, score), ...]`
    /// - Batch queries: `search(["fox", "dog"], k=10)` → `[[(doc_id, score), ...], ...]`
    /// - With subset: `search("fox", k=10, subset=[0, 2])` for pre-filtered search
    /// - Batch with subsets: `search(["fox", "dog"], k=10, subset=[[0,2], [1,3]])`
    ///
    /// Batch mode is faster: CPU uses rayon parallelism, GPU amortizes kernel overhead.
    #[pyo3(signature = (query, k, subset=None))]
    fn search(
        &mut self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        // Auto-upload to GPU only when cuda=True
        #[cfg(feature = "cuda")]
        if self.cuda_required && self.gpu_search_index.is_none() && !self.inner.is_empty() {
            self.gpu_search_index = Some(
                self.inner
                    .to_gpu_search_index()
                    .map_err(|e| {
                        PyValueError::new_err(format!("cuda=True but GPU upload failed: {}", e))
                    })?,
            );
            self.multi_gpu_index = Some(
                self.inner
                    .to_multi_gpu_search_index()
                    .map_err(|e| {
                        PyValueError::new_err(format!(
                            "cuda=True but multi-GPU init failed: {}",
                            e
                        ))
                    })?,
            );
        }

        // Check if query is a list (batch mode) or a string (single mode)
        if let Ok(query_str) = query.extract::<&str>() {
            // Single query mode
            let results = match subset {
                Some(s) => {
                    let ids: Vec<usize> = s.extract()?;
                    self.inner.search_filtered(query_str, k, &ids)
                }
                None => {
                    #[cfg(feature = "cuda")]
                    {
                        if let Some(ref mut gpu_idx) = self.gpu_search_index {
                            return Ok(self
                                .inner
                                .search_gpu(gpu_idx, query_str, k)
                                .into_iter()
                                .map(|r| (r.index, r.score))
                                .collect::<Vec<(usize, f32)>>()
                                .into_pyobject(py)?
                                .into_any()
                                .unbind());
                        }
                    }
                    self.inner.search(query_str, k)
                }
            };
            Ok(results
                .into_iter()
                .map(|r| (r.index, r.score))
                .collect::<Vec<(usize, f32)>>()
                .into_pyobject(py)?
                .into_any()
                .unbind())
        } else {
            // Batch mode: list of queries
            let query_list: Vec<String> = query.extract()?;
            let query_refs: Vec<&str> = query_list.iter().map(|s| s.as_str()).collect();

            let batch_results = match subset {
                Some(s) => {
                    let subset_lists: Vec<Vec<usize>> = s.extract()?;
                    let subset_refs: Vec<&[usize]> =
                        subset_lists.iter().map(|v| v.as_slice()).collect();
                    self.inner
                        .search_filtered_batch(&query_refs, k, &subset_refs)
                }
                None => {
                    #[cfg(feature = "cuda")]
                    {
                        // Multi-GPU batch: distribute queries across all GPUs
                        if let Some(ref mut mgpu) = self.multi_gpu_index {
                            return Ok(self
                                .inner
                                .search_multi_gpu_batch(mgpu, &query_refs, k)
                                .into_iter()
                                .map(|results| {
                                    results
                                        .into_iter()
                                        .map(|r| (r.index, r.score))
                                        .collect::<Vec<(usize, f32)>>()
                                })
                                .collect::<Vec<Vec<(usize, f32)>>>()
                                .into_pyobject(py)?
                                .into_any()
                                .unbind());
                        }
                        // Fallback: single-GPU batch
                        if let Some(ref mut gpu_idx) = self.gpu_search_index {
                            return Ok(self
                                .inner
                                .search_gpu_batch(gpu_idx, &query_refs, k)
                                .into_iter()
                                .map(|results| {
                                    results
                                        .into_iter()
                                        .map(|r| (r.index, r.score))
                                        .collect::<Vec<(usize, f32)>>()
                                })
                                .collect::<Vec<Vec<(usize, f32)>>>()
                                .into_pyobject(py)?
                                .into_any()
                                .unbind());
                        }
                    }
                    self.inner.search_batch(&query_refs, k)
                }
            };

            Ok(batch_results
                .into_iter()
                .map(|results| {
                    results
                        .into_iter()
                        .map(|r| (r.index, r.score))
                        .collect::<Vec<(usize, f32)>>()
                })
                .collect::<Vec<Vec<(usize, f32)>>>()
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }
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
    #[pyo3(signature = (index, mmap=false, cuda=false))]
    fn load(index: &str, mmap: bool, cuda: bool) -> PyResult<Self> {
        if cuda && !bm25x_core::is_gpu_available() {
            return Err(PyValueError::new_err(
                "cuda=True but no CUDA GPU is available.",
            ));
        }
        let inner = bm25x_core::BM25::load(index, mmap).map_err(io_err)?;
        Ok(PyBM25 {
            inner,
            cuda_required: cuda,
            #[cfg(feature = "cuda")]
            gpu_search_index: None,
            #[cfg(feature = "cuda")]
            multi_gpu_index: None,
        })
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

/// Returns True if bm25x was compiled with CUDA support and a GPU is available.
#[pyfunction]
fn is_gpu_available() -> bool {
    bm25x_core::is_gpu_available()
}

#[pymodule]
fn bm25x(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBM25>()?;
    m.add_function(wrap_pyfunction!(is_gpu_available, m)?)?;
    Ok(())
}
