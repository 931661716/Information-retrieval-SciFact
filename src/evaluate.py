"""Steps 4 and 6 - Run every retriever over the dev claims and compare the metrics.

Run:  python -m src.evaluate                       (TF-IDF and BM25)
      python -m src.evaluate --dense               (+ dense and hybrid)
      python -m src.evaluate --k1 0.9 --b 0.4      (BM25 with other parameters)
"""
import argparse
import json
import time
from pathlib import Path

from .data import load_corpus, load_queries_and_qrels
from .metrics import evaluate
from .sparse import BM25Retriever, InvertedIndex, TfIdfRetriever

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_retriever(retriever, queries, k=100):
    return {qid: [doc_id for doc_id, _ in retriever.search(q, k)] for qid, q in queries.items()}


def print_table(results):
    names = list(next(iter(results.values())))
    print(f"\n{'model':<14}" + "".join(f"{n:>11}" for n in names))
    for model, m in results.items():
        print(f"{model:<14}" + "".join(f"{m[n]:>11.4f}" for n in names))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["train", "dev"])
    p.add_argument("--k1", type=float, default=1.2)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--dense", action="store_true", help="also run dense + hybrid retrieval")
    args = p.parse_args()

    corpus = load_corpus()
    queries, qrels = load_queries_and_qrels(args.split)
    print(f"{len(corpus)} documents, {len(queries)} {args.split} queries")

    t = time.time()
    index = InvertedIndex(corpus)
    print(f"Built inverted index: {len(index.postings)} terms in {time.time() - t:.1f}s")

    retrievers = {
        "TF-IDF": TfIdfRetriever(index),
        "BM25": BM25Retriever(index, k1=args.k1, b=args.b),
    }
    if args.dense:
        from .dense import DenseRetriever, HybridRetriever
        retrievers["Dense"] = DenseRetriever(corpus)
        retrievers["Hybrid (RRF)"] = HybridRetriever([retrievers["BM25"], retrievers["Dense"]])

    results = {}
    for name, r in retrievers.items():
        t = time.time()
        run = run_retriever(r, queries)
        results[name] = evaluate(run, qrels)
        print(f"  {name}: {time.time() - t:.1f}s")

    print_table(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"metrics_{args.split}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
