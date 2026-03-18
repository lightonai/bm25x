"""Benchmark bm25s across all BEIR datasets for comparison."""

import time

import bm25s
import ir_measures
from beir.datasets.data_loader import GenericDataLoader
from ir_measures import nDCG

DATASETS = {
    "nfcorpus": "benchmarks/data/nfcorpus",
    "scifact": "benchmarks/data/scifact",
    "scidocs": "benchmarks/data/scidocs",
    "fiqa": "benchmarks/data/fiqa",
}


def load_dataset(name, path):
    split = "dev" if name == "msmarco" else "test"
    corpus, queries, qrels = GenericDataLoader(data_folder=path).load(split=split)

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

    return corpus_ids, corpus_texts, test_queries, ir_qrels


def bench_bm25s_dataset(corpus_ids, corpus_texts, test_queries, ir_qrels):
    # Index
    t0 = time.perf_counter()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    t_index = time.perf_counter() - t0

    num_docs = len(corpus_texts)
    dps = num_docs / t_index

    # Search
    query_list = list(test_queries.values())
    qid_list = list(test_queries.keys())
    query_tokens = bm25s.tokenize(query_list, stopwords="en")

    t0 = time.perf_counter()
    results = retriever.retrieve(query_tokens, k=10)
    t_search = time.perf_counter() - t0

    qps = len(query_list) / t_search

    # NDCG@10
    run = []
    for i, qid in enumerate(qid_list):
        doc_indices = results.documents[i]
        scores = results.scores[i]
        for doc_idx, score in zip(doc_indices, scores):
            did = corpus_ids[int(doc_idx)]
            run.append(ir_measures.ScoredDoc(str(qid), str(did), float(score)))

    metrics = ir_measures.calc_aggregate([nDCG @ 10], ir_qrels, run)
    ndcg10 = metrics[nDCG @ 10]

    return {
        "index_time": t_index,
        "dps": dps,
        "search_time": t_search,
        "qps": qps,
        "ndcg10": ndcg10,
        "num_docs": num_docs,
        "num_queries": len(query_list),
    }


print("=" * 70)
print(
    f"{'Dataset':<12} {'Docs':>10} {'Queries':>8} {'Index d/s':>12} {'Search q/s':>12} {'NDCG@10':>10}"
)
print("-" * 70)

for name, path in DATASETS.items():
    corpus_ids, corpus_texts, test_queries, ir_qrels = load_dataset(name, path)
    res = bench_bm25s_dataset(corpus_ids, corpus_texts, test_queries, ir_qrels)
    print(
        f"{name:<12} {res['num_docs']:>10,} {res['num_queries']:>8,} "
        f"{res['dps']:>12,.0f} {res['qps']:>12,.0f} {res['ndcg10']:>10.4f}"
    )

print("=" * 70)
