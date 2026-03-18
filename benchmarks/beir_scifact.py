"""
BEIR SciFact evaluation: bm25s vs bm25rs — NDCG@10
"""

import time

import ir_measures
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from ir_measures import nDCG

# ───────────────────────────── helpers ─────────────────────────────


def load_scifact():
    """Download and load SciFact via BEIR."""
    print("Downloading SciFact dataset...")
    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    )
    data_path = util.download_and_unzip(url, "benchmarks/data")

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # corpus: dict[doc_id] -> {"title": ..., "text": ...}
    # queries: dict[query_id] -> str
    # qrels: dict[query_id] -> dict[doc_id] -> relevance

    corpus_ids = list(corpus.keys())
    corpus_texts = [
        (corpus[did].get("title") or "") + " " + (corpus[did].get("text") or "")
        for did in corpus_ids
    ]

    # Build ir_measures qrels
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
    print(f"  Index time: {t_index:.3f}s")

    query_list = list(test_queries.values())
    qid_list = list(test_queries.keys())

    query_tokens = bm25s.tokenize(query_list, stopwords="en")

    t0 = time.perf_counter()
    results = retriever.retrieve(query_tokens, k=10)
    t_search = time.perf_counter() - t0
    print(f"  Search time ({len(query_list)} queries): {t_search:.3f}s")

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
    return ndcg10, t_index, t_search


# ───────────────────────────── bm25rs ─────────────────────────────


def run_bm25rs(corpus_ids, corpus_texts, test_queries, qrels):
    import bm25rs_python as bm25rs

    print("\n=== bm25rs ===")

    t0 = time.perf_counter()
    index = bm25rs.PyBM25Index(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
    ids = index.add(corpus_texts)
    t_index = time.perf_counter() - t0
    print(f"  Index time: {t_index:.3f}s")

    idx_to_did = {idx: corpus_ids[i] for i, idx in enumerate(ids)}

    query_list = list(test_queries.values())
    qid_list = list(test_queries.keys())

    t0 = time.perf_counter()
    all_results = [index.search(q, 10) for q in query_list]
    t_search = time.perf_counter() - t0
    print(f"  Search time ({len(query_list)} queries): {t_search:.3f}s")

    run = []
    for i, qid in enumerate(qid_list):
        for doc_idx, score in all_results[i]:
            did = idx_to_did.get(doc_idx, str(doc_idx))
            run.append(ir_measures.ScoredDoc(str(qid), str(did), float(score)))

    metrics = evaluate(run, qrels)
    ndcg10 = metrics[nDCG @ 10]
    print(f"  NDCG@10: {ndcg10:.4f}")
    return ndcg10, t_index, t_search


# ───────────────────────────── main ───────────────────────────────


def main():
    corpus_ids, corpus_texts, test_queries, qrels = load_scifact()

    ndcg_bm25s, idx_s, search_s = run_bm25s(
        corpus_ids, corpus_texts, test_queries, qrels
    )
    ndcg_bm25rs, idx_rs, search_rs = run_bm25rs(
        corpus_ids, corpus_texts, test_queries, qrels
    )

    print("\n" + "=" * 60)
    print(f"{'BEIR SciFact Results'}")
    print("=" * 60)
    print(f"{'Metric':<25} {'bm25s':>12} {'bm25rs':>12}")
    print("-" * 60)
    print(f"{'NDCG@10':<25} {ndcg_bm25s:>12.4f} {ndcg_bm25rs:>12.4f}")
    print(f"{'Index time (s)':<25} {idx_s:>12.3f} {idx_rs:>12.3f}")
    print(f"{'Search time (s)':<25} {search_s:>12.3f} {search_rs:>12.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
