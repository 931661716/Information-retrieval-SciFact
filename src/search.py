"""Step 7 - Interactive search: type a scientific claim, see the top abstracts.

Run:  python -m src.search            (BM25)
      python -m src.search --dense    (dense retrieval)
"""
import argparse

from .data import load_corpus
from .sparse import BM25Retriever, InvertedIndex


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dense", action="store_true")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    corpus = load_corpus()
    if args.dense:
        from .dense import DenseRetriever
        retriever = DenseRetriever(corpus)
    else:
        retriever = BM25Retriever(InvertedIndex(corpus))

    print("Type a claim (empty line to quit).")
    while True:
        query = input("\nclaim> ").strip()
        if not query:
            break
        for rank, (doc_id, score) in enumerate(retriever.search(query, args.k), start=1):
            doc = corpus[doc_id]
            print(f"{rank}. [{score:.3f}] ({doc_id}) {doc['title']}")
            print(f"   {doc['text'][:200]}...")


if __name__ == "__main__":
    main()
