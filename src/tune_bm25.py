"""Step 5 - Tune BM25's k1 and b on the TRAIN claims.

Never tune on the data you report results on: we pick parameters on train, then
check them once on dev (python -m src.evaluate --k1 ... --b ...).

Run:  python -m src.tune_bm25
"""
from .data import load_corpus, load_queries_and_qrels
from .evaluate import run_retriever
from .metrics import evaluate
from .sparse import BM25Retriever, InvertedIndex


def main():
    corpus = load_corpus()
    queries, qrels = load_queries_and_qrels("train")
    index = InvertedIndex(corpus)

    k1_values = [0.5, 0.9, 1.2, 1.5, 2.0]
    b_values = [0.25, 0.5, 0.75, 1.0]
    print(f"nDCG@10 on {len(queries)} train queries\n")
    print("k1 \\ b " + "".join(f"{b:>8}" for b in b_values))
    best = (0.0, None, None)
    for k1 in k1_values:
        row = []
        for b in b_values:
            score = evaluate(run_retriever(BM25Retriever(index, k1, b), queries), qrels)["nDCG@10"]
            row.append(score)
            best = max(best, (score, k1, b))
        print(f"{k1:<7}" + "".join(f"{s:>8.4f}" for s in row))
    print(f"\nBest: k1={best[1]}, b={best[2]} (nDCG@10={best[0]:.4f})")
    print(f"Now check it on dev:  python -m src.evaluate --k1 {best[1]} --b {best[2]}")


if __name__ == "__main__":
    main()
