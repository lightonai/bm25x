pub mod index;
pub mod scoring;
pub mod storage;
pub mod tokenizer;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod multi_gpu;

pub use index::{SearchResult, BM25};
pub use scoring::{Method, ScoringParams};
pub use tokenizer::TokenizerMode;

/// Returns true if the library was compiled with CUDA support and a GPU is available.
pub fn is_gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda::is_cuda_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}
