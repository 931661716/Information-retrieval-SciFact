"""Loading the corpus, the queries (claims) and the relevance judgments (qrels)."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scifact"


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus():
    """Return {doc_id (str): {"title": str, "text": str}}.

    Each abstract is stored as a list of sentences; we join them into one string.
    """
    corpus = {}
    for d in _read_jsonl(DATA_DIR / "corpus.jsonl"):
        corpus[str(d["doc_id"])] = {"title": d["title"], "text": " ".join(d["abstract"])}
    return corpus


def load_claims(split):
    """Return the raw claim records for 'train', 'dev' or 'test'."""
    return _read_jsonl(DATA_DIR / f"claims_{split}.jsonl")


def load_queries_and_qrels(split):
    """Turn claims into an IR test collection.

    queries: {query_id: claim text}
    qrels:   {query_id: {doc_id: relevance}}  (binary relevance: 1)

    A document is relevant to a claim if it contains evidence for it (the keys of
    `evidence`). Claims with no evidence ("not enough info") have no relevant documents,
    so they cannot be scored and are skipped. That leaves 188 of the 300 dev claims, so
    our numbers are not directly comparable to the BEIR leaderboard, which uses all 300.
    """
    queries, qrels = {}, {}
    for c in load_claims(split):
        evidence = c.get("evidence") or {}
        if not evidence:
            continue
        qid = str(c["id"])
        queries[qid] = c["claim"]
        qrels[qid] = {str(doc_id): 1 for doc_id in evidence}
    return queries, qrels
