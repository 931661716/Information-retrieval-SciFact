"""Step 4 - Evaluation metrics, written from scratch.

For one query we have a ranked list of doc_ids and a set of relevant doc_ids.

  Precision@k  fraction of the top k that are relevant
  Recall@k     fraction of all relevant docs that appear in the top k
  MRR@k        1 / rank of the first relevant doc (0 if none in the top k)
  AP           average of Precision@i over every rank i holding a relevant doc
               (averaged over queries -> MAP)
  nDCG@k       rewards relevant docs near the top with a log discount, normalized
               by the best possible ranking. The main metric of the BEIR benchmark.
"""
import math


def precision_at_k(ranked, relevant, k):
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked, relevant, k):
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def mrr_at_k(ranked, relevant, k):
    for rank, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(ranked, relevant):
    hits, total = 0, 0.0
    for rank, d in enumerate(ranked, start=1):
        if d in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def ndcg_at_k(ranked, relevant, k):
    # Binary relevance: gain is 1 for a relevant doc, 0 otherwise.
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def evaluate(run, qrels):
    """run: {qid: [doc_id, ...] ranked}, qrels: {qid: {doc_id: rel}} -> mean metrics."""
    metrics = {
        "nDCG@10": lambda r, rel: ndcg_at_k(r, rel, 10),
        "MRR@10": lambda r, rel: mrr_at_k(r, rel, 10),
        "P@1": lambda r, rel: precision_at_k(r, rel, 1),
        "Recall@10": lambda r, rel: recall_at_k(r, rel, 10),
        "Recall@100": lambda r, rel: recall_at_k(r, rel, 100),
        "MAP": average_precision,
    }
    totals = {name: 0.0 for name in metrics}
    for qid, relevant in qrels.items():
        ranked = run.get(qid, [])
        for name, fn in metrics.items():
            totals[name] += fn(ranked, relevant)
    return {name: v / len(qrels) for name, v in totals.items()}
