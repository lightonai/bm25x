"""
BEIR MS MARCO evaluation: bm25s vs bm25x
Metrics: NDCG@10, memory usage (RSS), index time, d/s, q/s
"""

import gc
import os
import shutil
import time

import ir_measures
import psutil
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from ir_measures import nDCG


def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def load_msmarco():
    """Download and load MS MARCO via BEIR."""
    print("Downloading MS MARCO dataset (this may take a while)...")
    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
    )
    data_path = util.download_and_unzip(url, "benchmarks/data")

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")

    corpus_ids = list(corpus.keys())
    corpus_texts = [
        (corpus[did].get("title") or "") + " " + (corpus[did].get("text") or "")
        for did in corpus_ids
    ]

    ir_qrels = []
    for qid, rels in qrels.items():
        for did, score in rels.items():
            ir_qrels.append(ir_measures.Qrel(str(qid), str(did), int(score)))

    test_qids = sorted(qrels.keys())
    test_queries = {qid: queries[qid] for qid in test_qids if qid in queries}

    print(f"  Corpus: {len(corpus):,} docs")
    print(f"  Queries (dev): {len(test_queries):,}")
    print(f"  Qrels: {len(ir_qrels):,}")
    return corpus_ids, corpus_texts, test_queries, ir_qrels


def evaluate(run, qrels):
    return ir_measures.calc_aggregate([nDCG @ 10], qrels, run)


# ───────────────────────────── bm25s ──────────────────────────────


def run_bm25s(corpus_ids, corpus_texts, test_queries, qrels):
    import bm25s

    print("\n=== bm25s ===")
    gc.collect()
    mem_before = get_memory_mb()

    t0 = time.perf_counter()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    t_index = time.perf_counter() - t0

    mem_after_index = get_memory_mb()
    mem_index_delta = mem_after_index - mem_before
    num_docs = len(corpus_texts)
    docs_per_sec = num_docs / t_index

    print(f"  Index: {t_index:.1f}s  ({docs_per_sec:,.0f} d/s)")
    print(f"  Memory: {mem_after_index:.0f} MB  (delta: {mem_index_delta:.0f} MB)")

    # Save + reload mmap to measure mmap memory
    index_dir = "/tmp/bm25s_msmarco_index"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    retriever.save(index_dir)
    del retriever
    del corpus_tokens
    gc.collect()

    mem_before_mmap = get_memory_mb()
    retriever = bm25s.BM25.load(index_dir, mmap=True)
    mem_after_mmap = get_memory_mb()
    mem_mmap_delta = mem_after_mmap - mem_before_mmap
    print(f"  Mmap memory: {mem_after_mmap:.0f} MB  (delta: {mem_mmap_delta:.0f} MB)")

    # Search
    query_list = list(test_queries.values())
    qid_list = list(test_queries.keys())
    query_tokens = bm25s.tokenize(query_list, stopwords="en")

    t0 = time.perf_counter()
    results = retriever.retrieve(query_tokens, k=10)
    t_search = time.perf_counter() - t0

    num_queries = len(query_list)
    queries_per_sec = num_queries / t_search
    print(f"  Search: {t_search:.1f}s  ({queries_per_sec:,.0f} q/s)")

    run = []
    for i, qid in enumerate(qid_list):
        doc_indices = results.documents[i]
        scores = results.scores[i]
        for doc_idx, score in zip(doc_indices, scores):
            did = corpus_ids[int(doc_idx)]
            run.append(ir_measures.ScoredDoc(str(qid), str(did), float(score)))

    metrics = evaluate(run, qrels)
    ndcg10 = metrics[nDCG @ 10]
    print(f"  NDCG@10: {ndcg10:.4f}")

    del retriever
    gc.collect()

    return {
        "ndcg10": ndcg10,
        "index_time": t_index,
        "search_time": t_search,
        "docs_per_sec": docs_per_sec,
        "queries_per_sec": queries_per_sec,
        "index_mem_mb": mem_index_delta,
        "mmap_mem_mb": mem_mmap_delta,
    }


# ───────────────────────────── bm25x ─────────────────────────────


def run_bm25x(corpus_ids, corpus_texts, test_queries, qrels):
    import bm25x

    print("\n=== bm25x ===")
    gc.collect()
    mem_before = get_memory_mb()

    t0 = time.perf_counter()
    index = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
    ids = index.add(corpus_texts)
    t_index = time.perf_counter() - t0

    mem_after_index = get_memory_mb()
    mem_index_delta = mem_after_index - mem_before
    num_docs = len(corpus_texts)
    docs_per_sec = num_docs / t_index

    print(f"  Index: {t_index:.1f}s  ({docs_per_sec:,.0f} d/s)")
    print(f"  Memory: {mem_after_index:.0f} MB  (delta: {mem_index_delta:.0f} MB)")

    # Save + reload mmap to measure mmap memory
    index_dir = "/tmp/bm25x_msmarco_index"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    index.save(index_dir)
    del index
    gc.collect()

    mem_before_mmap = get_memory_mb()
    index = bm25x.BM25.load(index_dir, mmap=True)
    mem_after_mmap = get_memory_mb()
    mem_mmap_delta = mem_after_mmap - mem_before_mmap
    print(f"  Mmap memory: {mem_after_mmap:.0f} MB  (delta: {mem_mmap_delta:.0f} MB)")

    idx_to_did = {idx: corpus_ids[i] for i, idx in enumerate(ids)}

    query_list = list(test_queries.values())
    qid_list = list(test_queries.keys())

    t0 = time.perf_counter()
    all_results = [index.search(q, 10) for q in query_list]
    t_search = time.perf_counter() - t0

    num_queries = len(query_list)
    queries_per_sec = num_queries / t_search
    print(f"  Search: {t_search:.1f}s  ({queries_per_sec:,.0f} q/s)")

    run = []
    for i, qid in enumerate(qid_list):
        for doc_idx, score in all_results[i]:
            did = idx_to_did.get(doc_idx, str(doc_idx))
            run.append(ir_measures.ScoredDoc(str(qid), str(did), float(score)))

    metrics = evaluate(run, qrels)
    ndcg10 = metrics[nDCG @ 10]
    print(f"  NDCG@10: {ndcg10:.4f}")

    del index
    gc.collect()

    return {
        "ndcg10": ndcg10,
        "index_time": t_index,
        "search_time": t_search,
        "docs_per_sec": docs_per_sec,
        "queries_per_sec": queries_per_sec,
        "index_mem_mb": mem_index_delta,
        "mmap_mem_mb": mem_mmap_delta,
    }


# ───────────────────────────── main ───────────────────────────────


def main():
    corpus_ids, corpus_texts, test_queries, qrels = load_msmarco()

    bm25s_res = run_bm25s(corpus_ids, corpus_texts, test_queries, qrels)
    bm25x_res = run_bm25x(corpus_ids, corpus_texts, test_queries, qrels)

    print("\n" + "=" * 65)
    print("BEIR MS MARCO Benchmark")
    print("=" * 65)
    header = f"{'Metric':<25} {'bm25s':>15} {'bm25x':>15}"
    print(header)
    print("-" * 65)
    print(f"{'NDCG@10':<25} {bm25s_res['ndcg10']:>15.4f} {bm25x_res['ndcg10']:>15.4f}")
    print(
        f"{'Index time (s)':<25} {bm25s_res['index_time']:>15.1f} {bm25x_res['index_time']:>15.1f}"
    )
    print(
        f"{'Index (d/s)':<25} {bm25s_res['docs_per_sec']:>15,.0f} {bm25x_res['docs_per_sec']:>15,.0f}"
    )
    print(
        f"{'Search (q/s)':<25} {bm25s_res['queries_per_sec']:>15,.0f} {bm25x_res['queries_per_sec']:>15,.0f}"
    )
    print(
        f"{'Index mem delta (MB)':<25} {bm25s_res['index_mem_mb']:>15.0f} {bm25x_res['index_mem_mb']:>15.0f}"
    )
    print(
        f"{'Mmap mem delta (MB)':<25} {bm25s_res['mmap_mem_mb']:>15.0f} {bm25x_res['mmap_mem_mb']:>15.0f}"
    )
    print("=" * 65)

    # Write GitHub Actions summary
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        md = [
            "## BEIR MS MARCO Benchmark",
            "",
            "| Metric | bm25s | bm25x |",
            "|--------|------:|-------:|",
            f"| NDCG@10 | {bm25s_res['ndcg10']:.4f} | {bm25x_res['ndcg10']:.4f} |",
            f"| Index time (s) | {bm25s_res['index_time']:.1f} | {bm25x_res['index_time']:.1f} |",
            f"| Index (d/s) | {bm25s_res['docs_per_sec']:,.0f} | {bm25x_res['docs_per_sec']:,.0f} |",
            f"| Search (q/s) | {bm25s_res['queries_per_sec']:,.0f} | {bm25x_res['queries_per_sec']:,.0f} |",
            f"| Index mem (MB) | {bm25s_res['index_mem_mb']:.0f} | {bm25x_res['index_mem_mb']:.0f} |",
            f"| Mmap mem (MB) | {bm25s_res['mmap_mem_mb']:.0f} | {bm25x_res['mmap_mem_mb']:.0f} |",
        ]
        with open(summary_path, "a") as f:
            f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
