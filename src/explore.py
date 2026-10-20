"""Step 1 - Explore the data before building anything.

Run:  python -m src.explore
"""
from collections import Counter

from .data import load_claims, load_corpus, load_queries_and_qrels
from .preprocess import tokenize


def main():
    corpus = load_corpus()
    lengths = [len(tokenize(d["title"] + " " + d["text"], stem=False, remove_stopwords=False))
               for d in corpus.values()]
    print("=== Corpus ===")
    print(f"documents:            {len(corpus)}")
    print(f"avg length (tokens):  {sum(lengths) / len(lengths):.1f}")
    print(f"min / max length:     {min(lengths)} / {max(lengths)}")

    first_id = next(iter(corpus))
    print(f"\nExample document {first_id}:")
    print("  title:", corpus[first_id]["title"])
    print("  text: ", corpus[first_id]["text"][:300], "...")

    print("\n=== Claims (queries) ===")
    for split in ["train", "dev", "test"]:
        claims = load_claims(split)
        labelled = sum(1 for c in claims if "evidence" in c)
        queries, qrels = load_queries_and_qrels(split)
        rel_counts = Counter(len(r) for r in qrels.values())
        print(f"{split:5s}: {len(claims):4d} claims, {labelled:4d} labelled, "
              f"{len(queries):4d} with relevant docs; relevant docs per query: {dict(sorted(rel_counts.items()))}")

    queries, qrels = load_queries_and_qrels("dev")
    qid = next(iter(queries))
    print(f"\nExample query {qid}: {queries[qid]}")
    for doc_id in qrels[qid]:
        print(f"  relevant doc {doc_id}: {corpus[doc_id]['title']}")


if __name__ == "__main__":
    main()
