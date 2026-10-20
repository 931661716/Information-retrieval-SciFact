"""Step 6 - Dense (neural) retrieval and hybrid fusion.

Instead of matching words, a neural encoder maps each text to a vector (an
"embedding") so that texts with similar meaning are close together. Retrieval is
then a nearest-neighbour search: score(q, d) = cos(embed(q), embed(d)).

This handles synonyms and paraphrases that BM25 misses ("heart attack" vs
"myocardial infarction"), but it can miss exact rare terms that BM25 nails.
A hybrid combines both.
"""
from pathlib import Path

import numpy as np

from .sparse import Retriever

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"

# A small (33M parameter) model that runs fine on a CPU.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained with this instruction prepended to *queries* (not documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever(Retriever):
    def __init__(self, corpus, model_name=DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        self.model = SentenceTransformer(model_name)
        self.doc_ids = list(corpus)
        cache = CACHE_DIR / (model_name.replace("/", "__") + ".npy")
        if cache.exists():
            self.doc_emb = np.load(cache)
        else:
            print(f"Encoding {len(corpus)} documents with {model_name} (first run only)...")
            texts = [corpus[d]["title"] + ". " + corpus[d]["text"] for d in self.doc_ids]
            self.doc_emb = self.model.encode(texts, batch_size=32, show_progress_bar=True,
                                             normalize_embeddings=True)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.doc_emb)

    def search(self, query, k=100):
        q = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        scores = self.doc_emb @ q  # vectors are unit length, so dot product = cosine
        top = np.argsort(-scores)[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in top]


class HybridRetriever(Retriever):
    """Reciprocal Rank Fusion (RRF): score(d) = sum over systems of 1 / (c + rank(d)).

    RRF uses only ranks, so we don't need to make BM25 scores (unbounded) and cosine
    similarities (between -1 and 1) comparable.
    """

    def __init__(self, retrievers, c=60, depth=100):
        self.retrievers, self.c, self.depth = retrievers, c, depth

    def search(self, query, k=100):
        fused = {}
        for r in self.retrievers:
            for rank, (doc_id, _) in enumerate(r.search(query, self.depth), start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.c + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
