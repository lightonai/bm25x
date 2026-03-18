"""Run both bm25s and bm25x benchmarks side by side."""

import gc
import json
import os
import shutil
import time

import psutil


def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def bench_bm25s(corpus, queries):
    import bm25s

    results = {}

    mem_before = get_memory_mb()
    t0 = time.perf_counter()
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    results["index_time"] = time.perf_counter() - t0
    results["index_mem_delta"] = get_memory_mb() - mem_before

    # Save and reload with mmap
    index_dir = "/tmp/bm25s_bench_index"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    retriever.save(index_dir)
    del retriever
    gc.collect()

    mem_before = get_memory_mb()
    t0 = time.perf_counter()
    retriever = bm25s.BM25.load(index_dir, mmap=True)
    results["load_time"] = time.perf_counter() - t0
    results["load_mem_delta"] = get_memory_mb() - mem_before

    # Search
    query_tokens = bm25s.tokenize(queries, stopwords="en")
    _ = retriever.retrieve(query_tokens[:10], k=10)  # warmup
    t0 = time.perf_counter()
    res = retriever.retrieve(query_tokens, k=10)
    results["search_time"] = time.perf_counter() - t0
    results["search_mem"] = get_memory_mb()

    # Top result for query 0
    results["sample_ids"] = res.documents[0][:5].tolist()
    results["sample_scores"] = res.scores[0][:5].tolist()

    return results


def bench_bm25x(corpus, queries):
    import bm25x

    results = {}

    mem_before = get_memory_mb()
    t0 = time.perf_counter()
    index = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
    index.add(corpus)
    results["index_time"] = time.perf_counter() - t0
    results["index_mem_delta"] = get_memory_mb() - mem_before

    # Save and reload with mmap
    index_dir = "/tmp/bm25x_bench_index"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    index.save(index_dir)
    del index
    gc.collect()

    mem_before = get_memory_mb()
    t0 = time.perf_counter()
    index = bm25x.BM25.load(index_dir, mmap=True)
    results["load_time"] = time.perf_counter() - t0
    results["load_mem_delta"] = get_memory_mb() - mem_before

    # Search
    for q in queries[:10]:
        _ = index.search(q, 10)  # warmup
    t0 = time.perf_counter()
    all_results = [index.search(q, 10) for q in queries]
    results["search_time"] = time.perf_counter() - t0
    results["search_mem"] = get_memory_mb()

    results["sample_ids"] = [r[0] for r in all_results[0][:5]]
    results["sample_scores"] = [r[1] for r in all_results[0][:5]]

    # Streaming operations
    t0 = time.perf_counter()
    index.add(["brand new document for streaming test"])
    results["add_time_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    index.delete([0, 1, 2])
    results["delete_time_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    index.update(3, "updated document text")
    results["update_time_ms"] = (time.perf_counter() - t0) * 1000

    return results


def main():
    with open("benchmarks/data/corpus.json") as f:
        corpus = json.load(f)
    with open("benchmarks/data/queries.json") as f:
        queries = json.load(f)

    print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)}")
    print("=" * 70)

    print("\n--- bm25s (Python, mmap) ---")
    bm25s_res = bench_bm25s(corpus, queries)

    print("\n--- bm25x (Rust, mmap) ---")
    bm25x_res = bench_bm25x(corpus, queries)

    print("\n" + "=" * 70)
    print(f"{'Metric':<30} {'bm25s':>12} {'bm25x':>12} {'Speedup':>10}")
    print("-" * 70)
    print(
        f"{'Index time (s)':<30} {bm25s_res['index_time']:>12.3f} {bm25x_res['index_time']:>12.3f} {bm25s_res['index_time'] / bm25x_res['index_time']:>9.1f}x"
    )
    print(
        f"{'Index mem delta (MB)':<30} {bm25s_res['index_mem_delta']:>12.1f} {bm25x_res['index_mem_delta']:>12.1f} {bm25s_res['index_mem_delta'] / bm25x_res['index_mem_delta']:>9.1f}x"
    )
    print(
        f"{'Mmap load time (s)':<30} {bm25s_res['load_time']:>12.3f} {bm25x_res['load_time']:>12.3f} {'N/A':>10}"
    )
    print(
        f"{'Search time (s)':<30} {bm25s_res['search_time']:>12.3f} {bm25x_res['search_time']:>12.3f} {bm25s_res['search_time'] / bm25x_res['search_time']:>9.1f}x"
    )
    print(
        f"{'Avg query (ms)':<30} {bm25s_res['search_time'] / len(queries) * 1000:>12.3f} {bm25x_res['search_time'] / len(queries) * 1000:>12.3f} {bm25s_res['search_time'] / bm25x_res['search_time']:>9.1f}x"
    )

    print(f"\n{'Streaming ops (bm25x only)':<30}")
    print(f"{'  Add 1 doc (ms)':<30} {'N/A':>12} {bm25x_res['add_time_ms']:>12.3f}")
    print(
        f"{'  Delete 3 docs (ms)':<30} {'N/A':>12} {bm25x_res['delete_time_ms']:>12.3f}"
    )
    print(
        f"{'  Update 1 doc (ms)':<30} {'N/A':>12} {bm25x_res['update_time_ms']:>12.3f}"
    )

    print("\nSample results (query 0):")
    print(
        f"  bm25s:  {list(zip(bm25s_res['sample_ids'], [f'{s:.4f}' for s in bm25s_res['sample_scores']]))}"
    )
    print(
        f"  bm25x: {list(zip(bm25x_res['sample_ids'], [f'{s:.4f}' for s in bm25x_res['sample_scores']]))}"
    )


if __name__ == "__main__":
    main()
