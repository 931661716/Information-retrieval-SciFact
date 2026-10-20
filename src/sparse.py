"""Step 3 - Inverted index + two classic ranking functions, written from scratch.

An inverted index maps each term to the documents that contain it ("postings"):

    "microglia" -> {doc_17: 3, doc_902: 1, ...}     (doc -> term frequency)

To answer a query we only visit the postings of the query terms, instead of
scanning all 5,183 documents. This is how search engines stay fast.
"""
import math
from collections import Counter, defaultdict

from .preprocess import tokenize


class InvertedIndex:
    def __init__(self, corpus):
        self.doc_ids = list(corpus)                 # internal index -> doc_id
        self.postings = defaultdict(dict)           # term -> {doc index: tf}
        self.doc_len = []                           # number of terms per document
        for i, doc_id in enumerate(self.doc_ids):
            doc = corpus[doc_id]
            terms = tokenize(doc["title"] + " " + doc["text"])
            self.doc_len.append(len(terms))
            for term, tf in Counter(terms).items():
                self.postings[term][i] = tf
        self.N = len(self.doc_ids)
        self.avgdl = sum(self.doc_len) / self.N

    def df(self, term):
        """Document frequency: in how many documents does the term appear?"""
        return len(self.postings.get(term, {}))


class Retriever:
    """Common interface: search(query, k) -> [(doc_id, score), ...] best first."""

    def search(self, query, k=100):
        raise NotImplementedError


def _top_k(scores, doc_ids, k):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(doc_ids[i], s) for i, s in ranked]


class TfIdfRetriever(Retriever):
    """Vector space model: documents and the query are TF-IDF vectors, ranked by cosine.

        weight(t, d) = (1 + log tf(t, d)) * log(N / df(t))
        score(q, d)  = (q . d) / (|q| |d|)
    """

    def __init__(self, index):
        self.index = index
        self.idf = {t: math.log(index.N / len(p)) for t, p in index.postings.items()}
        # Precompute each document's vector length for cosine normalization.
        sq = [0.0] * index.N
        for t, plist in index.postings.items():
            for i, tf in plist.items():
                sq[i] += ((1 + math.log(tf)) * self.idf[t]) ** 2
        self.doc_norm = [math.sqrt(x) or 1.0 for x in sq]

    def search(self, query, k=100):
        q_tf = Counter(t for t in tokenize(query) if t in self.idf)
        q_w = {t: (1 + math.log(tf)) * self.idf[t] for t, tf in q_tf.items()}
        q_norm = math.sqrt(sum(w * w for w in q_w.values())) or 1.0
        scores = defaultdict(float)
        for t, wq in q_w.items():
            for i, tf in self.index.postings[t].items():
                scores[i] += wq * (1 + math.log(tf)) * self.idf[t]
        for i in scores:
            scores[i] /= self.doc_norm[i] * q_norm
        return _top_k(scores, self.index.doc_ids, k)


class BM25Retriever(Retriever):
    """Okapi BM25, the standard lexical baseline.

        score(q, d) = sum over t in q of
            idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

        idf(t) = log(1 + (N - df + 0.5) / (df + 0.5))

    k1 controls term-frequency saturation (the 10th occurrence of a word adds much
    less than the 1st). b controls document-length normalization (b=0: none, b=1: full).
    """

    def __init__(self, index, k1=1.2, b=0.75):
        self.index, self.k1, self.b = index, k1, b
        self.idf = {t: math.log(1 + (index.N - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in index.postings.items()}

    def search(self, query, k=100):
        idx, k1, b = self.index, self.k1, self.b
        scores = defaultdict(float)
        for t in tokenize(query):  # repeated query terms count again, as in standard BM25
            if t not in self.idf:
                continue
            for i, tf in idx.postings[t].items():
                norm = k1 * (1 - b + b * idx.doc_len[i] / idx.avgdl)
                scores[i] += self.idf[t] * tf * (k1 + 1) / (tf + norm)
        return _top_k(scores, idx.doc_ids, k)
