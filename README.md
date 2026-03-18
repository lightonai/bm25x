# bm25rs

A fast, streaming-friendly BM25 search engine written in Rust with memory-mapped (mmap) index support.

Inspired by [bm25s](https://github.com/xhluca/bm25s), but designed from the ground up for incremental indexing — add, delete, and update documents without rebuilding the entire index.

## Features

- **All 5 BM25 variants**: Lucene (default), Robertson, ATIRE, BM25L, BM25+
- **Streaming index**: Add documents incrementally, delete by ID, update in-place
- **Memory-mapped storage**: Load large indices with minimal RAM via mmap
- **Simple API**: Pass in strings, get back `(doc_index, score)` pairs
- **Python bindings**: Via [maturin](https://github.com/PyO3/maturin) / PyO3

## Quickstart (Rust)

```rust
use bm25rs::{BM25Index, Method};

let mut index = BM25Index::new(Method::Lucene, 1.5, 0.75, 0.5, true);

// Add documents (returns assigned indices)
let ids = index.add(&[
    "the quick brown fox jumps over the lazy dog",
    "a fast brown car drives over the bridge",
    "the fox is quick and clever",
]);

// Search
let results = index.search("quick fox", 10);
for r in &results {
    println!("doc {}: {:.4}", r.index, r.score);
}

// Streaming operations
index.add(&["a brand new document"]);   // append
index.delete(&[1]);                      // remove doc 1
index.update(0, "updated text for doc"); // replace doc 0's content

// Persistence with mmap
index.save("./my_index").unwrap();
let index = BM25Index::load("./my_index", true).unwrap(); // mmap=true
```

## Quickstart (Python)

Build the Python bindings:

```bash
cd python
uv venv && source .venv/bin/activate
uv pip install maturin
maturin develop --release
```

```python
from bm25rs_python import PyBM25Index

index = PyBM25Index(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
index.add(["the quick brown fox", "lazy dog on a mat", "fox and hound"])

results = index.search("quick fox", k=10)
for doc_idx, score in results:
    print(f"doc {doc_idx}: {score:.4f}")

# Streaming
index.add(["new document"])
index.delete([1])
index.update(0, "replaced text")

# Save / load with mmap
index.save("/tmp/my_index")
index = PyBM25Index.load("/tmp/my_index", mmap=True)
```

## API

### `BM25Index::new(method, k1, b, delta, use_stopwords)`

Create an empty index.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | `Method` | `Lucene` | BM25 variant: `Lucene`, `Robertson`, `Atire`, `BM25L`, `BM25Plus` |
| `k1` | `f32` | `1.5` | Term frequency saturation parameter |
| `b` | `f32` | `0.75` | Document length normalization |
| `delta` | `f32` | `0.5` | Delta parameter (BM25L/BM25+ only) |
| `use_stopwords` | `bool` | `true` | Remove English stopwords |

### Methods

| Method | Description |
|---|---|
| `add(&[&str]) -> Vec<usize>` | Add documents, returns their indices |
| `search(&str, k) -> Vec<SearchResult>` | Top-k search, returns `(index, score)` pairs |
| `delete(&[usize])` | Delete documents by index |
| `update(usize, &str)` | Replace a document's text at given index |
| `save(path)` | Persist index to directory |
| `load(path, mmap)` | Load index (optionally memory-mapped) |
| `len()` | Number of active documents |

## Benchmarks

### BEIR SciFact — NDCG@10

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

### Synthetic corpus (100k docs, 1k queries)

| Metric | bm25s | bm25rs |
|---|---|---|
| Index time | 6.09s | **1.85s** (3.3x faster) |
| Index memory | 282 MB | **100 MB** (2.8x less) |
| Mmap load | 10ms | **<1ms** |
| Search time | **1.20s** | 1.81s |
| Add 1 doc | N/A | 12ms |
| Delete 3 docs | N/A | 0.006ms |
| Update 1 doc | N/A | 3.7ms |

Search on the synthetic corpus is slower due to the lazy scoring tradeoff (scores computed at query time to enable streaming). On real-world data (SciFact and MS MARCO), bm25rs is faster across the board — up to 4x faster search on MS MARCO.

## Design

bm25s pre-computes all BM25 scores at index time (eager scoring). This makes queries fast but rebuilding the index is required to add or remove documents.

bm25rs uses **lazy scoring** — it stores raw term frequencies in an inverted index and computes BM25 scores at query time. This makes the index naturally streaming-friendly: add, delete, and update are cheap operations that don't require a full rebuild.

On-disk, the index uses a flat binary format with memory-mapped postings and document lengths, keeping RAM usage minimal for large indices.

## License

MIT
