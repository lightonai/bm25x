"""
Benchmark BM25 accuracy improvements on BEIR datasets.
Tests title boosting (via repetition) and BM25 variant selection.
All techniques are zero-cost at query time.
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


def load_dataset(name):
    info = DATASETS[name]
    data_path = util.download_and_unzip(info["url"], "benchmarks/data")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
        split=info["split"]
    )
    corpus_ids = list(corpus.keys())

    test_qids = sorted(qrels.keys())
    test_queries = {qid: queries[qid] for qid in test_qids if qid in queries}

    ir_qrels = []
    for qid, rels in qrels.items():
        for did, score in rels.items():
            ir_qrels.append(ir_measures.Qrel(str(qid), str(did), int(score)))

    return corpus, corpus_ids, test_queries, ir_qrels


def build_texts(corpus, corpus_ids, title_repeat=1):
    """Build document texts with optional title repetition."""
    texts = []
    for did in corpus_ids:
        title = corpus[did].get("title") or ""
        body = corpus[did].get("text") or ""
        if title and title_repeat > 1:
            boosted_title = " ".join([title] * title_repeat)
            texts.append(boosted_title + " " + body)
        else:
            texts.append(title + " " + body)
    return texts


def evaluate(index, corpus_ids, ids, test_queries, qrels):
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
    return metrics[nDCG @ 10], t_search, len(query_list)


def run_experiment(name, corpus, corpus_ids, test_queries, qrels, method, title_repeat):
    texts = build_texts(corpus, corpus_ids, title_repeat)
    index = BM25(
        method=method, k1=1.5, b=0.75, tokenizer="unicode_stem", use_stopwords=True
    )

    t0 = time.perf_counter()
    ids = index.add(texts)
    t_index = time.perf_counter() - t0

    ndcg, t_search, nq = evaluate(index, corpus_ids, ids, test_queries, qrels)
    qs = nq / t_search
    return {"ndcg10": ndcg, "index_time": t_index, "qs": qs}


# Experiments to run
EXPERIMENTS = [
    # (label, method, title_repeat)
    ("baseline", "lucene", 1),
    ("title_x2", "lucene", 2),
    ("title_x3", "lucene", 3),
    ("title_x5", "lucene", 5),
    ("atire", "atire", 1),
    ("atire_title_x3", "atire", 3),
    ("robertson", "robertson", 1),
    ("robertson_t_x3", "robertson", 3),
    ("bm25l", "bm25l", 1),
    ("bm25l_title_x3", "bm25l", 3),
    ("bm25plus", "bm25+", 1),
    ("bm25plus_title_x3", "bm25+", 3),
]


def main():
    all_results = defaultdict(dict)

    for dataset_name in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        corpus, corpus_ids, test_queries, qrels = load_dataset(dataset_name)
        print(f"  Corpus: {len(corpus_ids):,} docs | Queries: {len(test_queries)}")

        for label, method, title_repeat in EXPERIMENTS:
            res = run_experiment(
                label, corpus, corpus_ids, test_queries, qrels, method, title_repeat
            )
            all_results[dataset_name][label] = res
            print(
                f"  {label:<20} NDCG@10={res['ndcg10']:.4f}  "
                f"index={res['index_time']:.2f}s  {res['qs']:.0f} q/s"
            )

    # Summary
    labels = [label for label, _, _ in EXPERIMENTS]
    print(f"\n\n{'=' * 100}")
    print("SUMMARY: NDCG@10")
    print(f"{'=' * 100}")
    header = f"{'Experiment':<22}"
    for ds in DATASETS:
        header += f" {ds:>12}"
    header += f" {'AVERAGE':>12}"
    print(header)
    print("-" * 100)

    best_label = None
    best_avg = 0.0
    for label in labels:
        row = f"{label:<22}"
        total = 0.0
        for ds in DATASETS:
            ndcg = all_results[ds][label]["ndcg10"]
            total += ndcg
            row += f" {ndcg:>12.4f}"
        avg = total / len(DATASETS)
        row += f" {avg:>12.4f}"
        if avg > best_avg:
            best_avg = avg
            best_label = label
        print(row)

    print("-" * 100)
    baseline_avg = sum(all_results[ds]["baseline"]["ndcg10"] for ds in DATASETS) / len(
        DATASETS
    )
    print(
        f"\nBest: {best_label} (avg={best_avg:.4f}, "
        f"+{(best_avg - baseline_avg) / baseline_avg * 100:.1f}% vs baseline)"
    )


if __name__ == "__main__":
    main()
