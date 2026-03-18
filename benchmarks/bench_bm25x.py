"""Benchmark bm25x (Rust, mmap mode)."""

import json
import os
import shutil
import time

import bm25x
import psutil


def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def main():
    with open("benchmarks/data/corpus.json") as f:
        corpus = json.load(f)
    with open("benchmarks/data/queries.json") as f:
        queries = json.load(f)

    print(f"Corpus size: {len(corpus)} documents")
    print(f"Queries: {len(queries)}")
    print()

    # --- Index ---
    mem_before = get_memory_mb()
    t0 = time.perf_counter()

    index = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
    index.add(corpus)

    t_index = time.perf_counter() - t0
    mem_after_index = get_memory_mb()
    print(f"[bm25x] Index time: {t_index:.3f}s")
    print(
        f"[bm25x] Memory after indexing: {mem_after_index:.1f} MB (delta: {mem_after_index - mem_before:.1f} MB)"
    )

    # Save and reload with mmap
    index_dir = "/tmp/bm25x_bench_index"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    index.save(index_dir)

    import gc

    del index
    gc.collect()

    mem_before_mmap = get_memory_mb()
    t0 = time.perf_counter()
    index = bm25x.BM25.load(index_dir, mmap=True)
    t_load = time.perf_counter() - t0
    mem_after_mmap = get_memory_mb()
    print(f"[bm25x] Mmap load time: {t_load:.3f}s")
    print(
        f"[bm25x] Memory after mmap load: {mem_after_mmap:.1f} MB (delta: {mem_after_mmap - mem_before_mmap:.1f} MB)"
    )
    print()

    # --- Search ---
    # Warmup
    for q in queries[:10]:
        _ = index.search(q, 10)

    t0 = time.perf_counter()
    all_results = []
    for q in queries:
        all_results.append(index.search(q, 10))
    t_search = time.perf_counter() - t0
    mem_after_search = get_memory_mb()

    print(f"[bm25x] Search time ({len(queries)} queries, k=10): {t_search:.3f}s")
    print(f"[bm25x] Avg query time: {t_search / len(queries) * 1000:.3f}ms")
    print(f"[bm25x] Memory after search: {mem_after_search:.1f} MB")
    print()

    # Print sample results for comparison
    print("Sample results (query 0):")
    for idx, score in all_results[0][:5]:
        print(f"  doc_id={idx}, score={score:.4f}")

    # --- Test streaming add ---
    print()
    print("--- Streaming test ---")
    t0 = time.perf_counter()
    new_ids = index.add(["this is a brand new document added incrementally"])
    t_add = time.perf_counter() - t0
    print(
        f"[bm25x] Add 1 doc to existing index: {t_add * 1000:.3f}ms (new id: {new_ids})"
    )
    print(f"[bm25x] Index size after add: {len(index)}")

    # --- Test delete ---
    t0 = time.perf_counter()
    index.delete([0, 1, 2])
    t_del = time.perf_counter() - t0
    print(f"[bm25x] Delete 3 docs: {t_del * 1000:.3f}ms")
    print(f"[bm25x] Index size after delete: {len(index)}")

    # --- Test update ---
    t0 = time.perf_counter()
    index.update(3, "updated text for document three")
    t_upd = time.perf_counter() - t0
    print(f"[bm25x] Update 1 doc: {t_upd * 1000:.3f}ms")


if __name__ == "__main__":
    main()
