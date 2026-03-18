"""Benchmark bm25x with rayon parallelism vs previous single-threaded numbers."""

import json
import random
import time

import bm25x

# Load benchmark datasets
with open("benchmarks/data/corpus.json") as f:
    corpus = json.load(f)
with open("benchmarks/data/queries.json") as f:
    queries = json.load(f)

print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)}")
print("=" * 65)

# --- Indexing benchmark ---
# Warmup
warmup_idx = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
warmup_idx.add(corpus[:100])
del warmup_idx

# Actual run (3 trials)
index_times = []
for trial in range(3):
    index = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
    t0 = time.perf_counter()
    index.add(corpus)
    t_index = time.perf_counter() - t0
    index_times.append(t_index)
    if trial < 2:
        del index

best_index = min(index_times)
dps = len(corpus) / best_index
print(f"\nIndexing ({len(corpus)} docs):")
print(f"  Times: {[f'{t:.3f}s' for t in index_times]}")
print(f"  Best:  {best_index:.3f}s  ({dps:,.0f} d/s)")

# --- Search benchmark ---
# Warmup
for q in queries[:10]:
    index.search(q, 10)

search_times = []
for trial in range(3):
    t0 = time.perf_counter()
    for q in queries:
        index.search(q, 10)
    t_search = time.perf_counter() - t0
    search_times.append(t_search)

best_search = min(search_times)
qps = len(queries) / best_search
print(f"\nSearch ({len(queries)} queries):")
print(f"  Times: {[f'{t:.3f}s' for t in search_times]}")
print(f"  Best:  {best_search:.3f}s  ({qps:,.0f} q/s)")

# --- Filtered search benchmark ---
random.seed(42)
subsets = {
    "1k docs": [random.sample(range(len(corpus)), 1000) for _ in range(len(queries))],
    "100 docs": [random.sample(range(len(corpus)), 100) for _ in range(len(queries))],
}

for label, subs in subsets.items():
    # Warmup
    for q, sub in zip(queries[:10], subs[:10]):
        index.search(q, 10, subset=sub)

    times = []
    for trial in range(3):
        t0 = time.perf_counter()
        for q, sub in zip(queries, subs):
            index.search(q, 10, subset=sub)
        times.append(time.perf_counter() - t0)

    best = min(times)
    qps = len(queries) / best
    print(f"\nFiltered search ({label}, {len(queries)} queries):")
    print(f"  Times: {[f'{t:.3f}s' for t in times]}")
    print(f"  Best:  {best:.3f}s  ({qps:,.0f} q/s)")

print("\n" + "=" * 65)
print("Done.")
