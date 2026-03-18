"""
Benchmark all bm25x tokenizer modes on MTEB retrieval tasks.
Evaluates: plain, unicode, stem, unicode_stem
Datasets: SciFact, NFCorpus, FiQA (small/fast MTEB retrieval tasks)
Metric: NDCG@10
"""

import time
from collections import defaultdict

import ir_measures
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from bm25x import BM25
from ir_measures import nDCG

DATASETS = {
    "scifact": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "split": "test",
    },
    "nfcorpus": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        "split": "test",
    },
    "fiqa": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
        "split": "test",
    },
    "scidocs": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
        "split": "test",
    },
}

TOKENIZER_MODES = ["plain", "unicode", "stem", "unicode_stem"]


def load_dataset(name):
    info = DATASETS[name]
    data_path = util.download_and_unzip(info["url"], "benchmarks/data")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
        split=info["split"]
    )

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


def evaluate_tokenizer(tokenizer_mode, corpus_ids, corpus_texts, test_queries, qrels):
    index = BM25(
        method="lucene", k1=1.5, b=0.75, tokenizer=tokenizer_mode, use_stopwords=True
    )

    t0 = time.perf_counter()
    ids = index.add(corpus_texts)
    t_index = time.perf_counter() - t0

    idx_to_did = {idx: corpus_ids[i] for i, idx in enumerate(ids)}

    query_list = list(test_queries.values())
    qid_list = list(test_queries.keys())

    t0 = time.perf_counter()
    all_results = [index.search(q, 10) for q in query_list]
    t_search = time.perf_counter() - t0

    run = []
    for i, qid in enumerate(qid_list):
        for doc_idx, score in all_results[i]:
            did = idx_to_did.get(doc_idx, str(doc_idx))
            run.append(ir_measures.ScoredDoc(str(qid), str(did), float(score)))

    metrics = ir_measures.calc_aggregate([nDCG @ 10], qrels, run)
    ndcg10 = metrics[nDCG @ 10]

    return {
        "ndcg10": ndcg10,
        "index_time": t_index,
        "search_time": t_search,
        "num_queries": len(query_list),
    }


def main():
    all_results = defaultdict(dict)

    for dataset_name in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        corpus_ids, corpus_texts, test_queries, qrels = load_dataset(dataset_name)
        print(f"  Corpus: {len(corpus_texts):,} docs | Queries: {len(test_queries)}")

        for mode in TOKENIZER_MODES:
            res = evaluate_tokenizer(
                mode, corpus_ids, corpus_texts, test_queries, qrels
            )
            all_results[dataset_name][mode] = res
            qs = res["num_queries"] / res["search_time"]
            print(
                f"  {mode:<15} NDCG@10={res['ndcg10']:.4f}  "
                f"index={res['index_time']:.2f}s  search={qs:.0f} q/s"
            )

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY: NDCG@10 by tokenizer mode")
    print(f"{'=' * 80}")

    header = f"{'Dataset':<15}"
    for mode in TOKENIZER_MODES:
        header += f" {mode:>15}"
    print(header)
    print("-" * 80)

    avg_scores = {mode: 0.0 for mode in TOKENIZER_MODES}
    for dataset_name in DATASETS:
        row = f"{dataset_name:<15}"
        for mode in TOKENIZER_MODES:
            ndcg = all_results[dataset_name][mode]["ndcg10"]
            avg_scores[mode] += ndcg
            row += f" {ndcg:>15.4f}"
        print(row)

    n = len(DATASETS)
    row = f"{'AVERAGE':<15}"
    for mode in TOKENIZER_MODES:
        avg = avg_scores[mode] / n
        row += f" {avg:>15.4f}"
    print("-" * 80)
    print(row)

    best = max(avg_scores, key=avg_scores.get)
    print(f"\nBest tokenizer: {best} (avg NDCG@10 = {avg_scores[best] / n:.4f})")

    # Speed summary
    print(f"\n{'=' * 80}")
    print("SUMMARY: Search throughput (q/s) by tokenizer mode")
    print(f"{'=' * 80}")
    header = f"{'Dataset':<15}"
    for mode in TOKENIZER_MODES:
        header += f" {mode:>15}"
    print(header)
    print("-" * 80)
    for dataset_name in DATASETS:
        row = f"{dataset_name:<15}"
        for mode in TOKENIZER_MODES:
            res = all_results[dataset_name][mode]
            qs = res["num_queries"] / res["search_time"]
            row += f" {qs:>15,.0f}"
        print(row)


if __name__ == "__main__":
    main()
