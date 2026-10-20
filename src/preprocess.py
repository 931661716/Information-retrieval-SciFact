"""Step 2 - Text preprocessing: turn raw text into index terms.

  "Microglia are activated in Alzheimer's disease"
    -> lowercase + tokenize  -> ["microglia", "are", "activated", "in", "alzheimer", "s", "disease"]
    -> remove stop words     -> ["microglia", "activated", "alzheimer", "s", "disease"]
    -> stem                  -> ["microglia", "activ", "alzheim", "s", "diseas"]

Stemming lets "activated" in a query match "activation" in a document.

Try it:  python -m src.preprocess "Microglia are activated in Alzheimer's disease"
"""
import re
import sys
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_stemmer = PorterStemmer()


@lru_cache(maxsize=None)
def _stem(word):
    # Caching matters: the same words appear thousands of times in the corpus.
    return _stemmer.stem(word)


def tokenize(text, stem=True, remove_stopwords=True):
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    if stem:
        tokens = [_stem(t) for t in tokens]
    return tokens


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "Microglia are activated in Alzheimer's disease"
    print("raw tokens:     ", tokenize(text, stem=False, remove_stopwords=False))
    print("no stop words:  ", tokenize(text, stem=False))
    print("stemmed:        ", tokenize(text))
