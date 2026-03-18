"""Benchmark bm25s (mmap mode)."""

import json
import os
import shutil
import time

import bm25s
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

    # Tokenize
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    t_index = time.perf_counter() - t0
    mem_after_index = get_memory_mb()
    print(f"[bm25s] Index time: {t_index:.3f}s")
    print(
        f"[bm25s] Memory after indexing: {mem_after_index:.1f} MB (delta: {mem_after_index - mem_before:.1f} MB)"
    )

    # Save and reload with mmap
    index_dir = "/tmp/bm25s_bench_index"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    retriever.save(index_dir)

    import gc

    del retriever
    gc.collect()

    mem_before_mmap = get_memory_mb()
    t0 = time.perf_counter()
    retriever = bm25s.BM25.load(index_dir, mmap=True)
    t_load = time.perf_counter() - t0
    mem_after_mmap = get_memory_mb()
    print(f"[bm25s] Mmap load time: {t_load:.3f}s")
    print(
        f"[bm25s] Memory after mmap load: {mem_after_mmap:.1f} MB (delta: {mem_after_mmap - mem_before_mmap:.1f} MB)"
    )
    print()

    # --- Search ---
    query_tokens = bm25s.tokenize(queries, stopwords="en")

    # Warmup
    _ = retriever.retrieve(query_tokens[:10], k=10)

    t0 = time.perf_counter()
    results = retriever.retrieve(query_tokens, k=10)
    t_search = time.perf_counter() - t0
    mem_after_search = get_memory_mb()

    print(f"[bm25s] Search time ({len(queries)} queries, k=10): {t_search:.3f}s")
    print(f"[bm25s] Avg query time: {t_search / len(queries) * 1000:.3f}ms")
    print(f"[bm25s] Memory after search: {mem_after_search:.1f} MB")
    print()

    # Print sample results for comparison
    print("Sample results (query 0):")
    doc_ids = results.documents[0]
    scores = results.scores[0]
    for i in range(min(5, len(doc_ids))):
        print(f"  doc_id={doc_ids[i]}, score={scores[i]:.4f}")


if __name__ == "__main__":
    main()
