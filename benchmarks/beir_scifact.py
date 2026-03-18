"""
BEIR SciFact evaluation: bm25s vs bm25x — NDCG@10, q/s, d/s

Exit code 1 if bm25x NDCG@10 drops below threshold.
Writes a GitHub Actions job summary when $GITHUB_STEP_SUMMARY is set.
"""

import os
import sys
import time

import ir_measures
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from ir_measures import nDCG

# Fail the CI if bm25x NDCG@10 falls below this value
MIN_NDCG10 = 0.64


# ───────────────────────────── helpers ─────────────────────────────


def load_scifact():
    """Download and load SciFact via BEIR."""
    print("Downloading SciFact dataset...")
    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    )
    data_path = util.download_and_unzip(url, "benchmarks/data")

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

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

    print(f"  Corpus: {len(corpus)} docs")
    print(f"  Queries (test): {len(test_queries)}")
    print(f"  Qrels: {len(ir_qrels)}")
    return corpus_ids, corpus_texts, test_queries, ir_qrels


def evaluate(run, qrels):
    """Compute NDCG@10."""
    return ir_measures.calc_aggregate([nDCG @ 10], qrels, run)


# ───────────────────────────── bm25s ──────────────────────────────


def run_bm25s(corpus_ids, corpus_texts, test_queries, qrels):
    import bm25s

    print("\n=== bm25s ===")

    t0 = time.perf_counter()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    t_index = time.perf_counter() - t0

    num_docs = len(corpus_texts)
    docs_per_sec = num_docs / t_index
    print(f"  Index: {t_index:.3f}s  ({docs_per_sec:.0f} d/s)")

    query_list = list(test_queries.values())
    qid_list = list(test_queries.keys())
    query_tokens = bm25s.tokenize(query_list, stopwords="en")

    t0 = time.perf_counter()
    results = retriever.retrieve(query_tokens, k=10)
    t_search = time.perf_counter() - t0

    num_queries = len(query_list)
    queries_per_sec = num_queries / t_search
    print(f"  Search: {t_search:.3f}s  ({queries_per_sec:.0f} q/s)")

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

    return {
        "ndcg10": ndcg10,
        "index_time": t_index,
        "search_time": t_search,
        "docs_per_sec": docs_per_sec,
        "queries_per_sec": queries_per_sec,
    }


# ───────────────────────────── bm25x ─────────────────────────────


def run_bm25x(corpus_ids, corpus_texts, test_queries, qrels):
    import bm25x

    print("\n=== bm25x ===")

    t0 = time.perf_counter()
    index = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
    ids = index.add(corpus_texts)
    t_index = time.perf_counter() - t0

    num_docs = len(corpus_texts)
    docs_per_sec = num_docs / t_index
    print(f"  Index: {t_index:.3f}s  ({docs_per_sec:.0f} d/s)")

    idx_to_did = {idx: corpus_ids[i] for i, idx in enumerate(ids)}

    query_list = list(test_queries.values())
    qid_list = list(test_queries.keys())

    t0 = time.perf_counter()
    all_results = [index.search(q, 10) for q in query_list]
    t_search = time.perf_counter() - t0

    num_queries = len(query_list)
    queries_per_sec = num_queries / t_search
    print(f"  Search: {t_search:.3f}s  ({queries_per_sec:.0f} q/s)")

    run = []
    for i, qid in enumerate(qid_list):
        for doc_idx, score in all_results[i]:
            did = idx_to_did.get(doc_idx, str(doc_idx))
            run.append(ir_measures.ScoredDoc(str(qid), str(did), float(score)))

    metrics = evaluate(run, qrels)
    ndcg10 = metrics[nDCG @ 10]
    print(f"  NDCG@10: {ndcg10:.4f}")

    return {
        "ndcg10": ndcg10,
        "index_time": t_index,
        "search_time": t_search,
        "docs_per_sec": docs_per_sec,
        "queries_per_sec": queries_per_sec,
    }


# ───────────────────────────── output ─────────────────────────────


def write_summary(bm25s_res, bm25x_res):
    """Print results table and optionally write GitHub Actions summary."""
    header = f"{'Metric':<25} {'bm25s':>12} {'bm25x':>12}"
    sep = "-" * 55

    lines = [
        "BEIR SciFact Benchmark",
        "=" * 55,
        header,
        sep,
        f"{'NDCG@10':<25} {bm25s_res['ndcg10']:>12.4f} {bm25x_res['ndcg10']:>12.4f}",
        f"{'Index (d/s)':<25} {bm25s_res['docs_per_sec']:>12,.0f} {bm25x_res['docs_per_sec']:>12,.0f}",
        f"{'Search (q/s)':<25} {bm25s_res['queries_per_sec']:>12,.0f} {bm25x_res['queries_per_sec']:>12,.0f}",
        f"{'Index time (s)':<25} {bm25s_res['index_time']:>12.3f} {bm25x_res['index_time']:>12.3f}",
        f"{'Search time (s)':<25} {bm25s_res['search_time']:>12.3f} {bm25x_res['search_time']:>12.3f}",
        "=" * 55,
    ]
    text = "\n".join(lines)
    print("\n" + text)

    # Write GitHub Actions job summary (markdown table)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        md = [
            "## BEIR SciFact Benchmark",
            "",
            "| Metric | bm25s | bm25x |",
            "|--------|------:|-------:|",
            f"| NDCG@10 | {bm25s_res['ndcg10']:.4f} | {bm25x_res['ndcg10']:.4f} |",
            f"| Index (d/s) | {bm25s_res['docs_per_sec']:,.0f} | {bm25x_res['docs_per_sec']:,.0f} |",
            f"| Search (q/s) | {bm25s_res['queries_per_sec']:,.0f} | {bm25x_res['queries_per_sec']:,.0f} |",
            f"| Index time (s) | {bm25s_res['index_time']:.3f} | {bm25x_res['index_time']:.3f} |",
            f"| Search time (s) | {bm25s_res['search_time']:.3f} | {bm25x_res['search_time']:.3f} |",
            "",
            f"**Threshold check:** bm25x NDCG@10 = {bm25x_res['ndcg10']:.4f} "
            f"(min: {MIN_NDCG10}) "
            f"{'✅ PASS' if bm25x_res['ndcg10'] >= MIN_NDCG10 else '❌ FAIL'}",
        ]
        with open(summary_path, "a") as f:
            f.write("\n".join(md) + "\n")


# ───────────────────────────── main ───────────────────────────────


def main():
    corpus_ids, corpus_texts, test_queries, qrels = load_scifact()

    bm25s_res = run_bm25s(corpus_ids, corpus_texts, test_queries, qrels)
    bm25x_res = run_bm25x(corpus_ids, corpus_texts, test_queries, qrels)

    write_summary(bm25s_res, bm25x_res)

    # Gate: fail CI if NDCG@10 is below threshold
    if bm25x_res["ndcg10"] < MIN_NDCG10:
        print(
            f"\nFAIL: bm25x NDCG@10 = {bm25x_res['ndcg10']:.4f} < {MIN_NDCG10}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nPASS: bm25x NDCG@10 = {bm25x_res['ndcg10']:.4f} >= {MIN_NDCG10}")


if __name__ == "__main__":
    main()
