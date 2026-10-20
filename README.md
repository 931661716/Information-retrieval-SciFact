# Information Retrieval on SciFact

A learning project: given a scientific **claim**, find the research **abstracts** that
contain evidence for or against it. You build a search engine from scratch, measure
it, and compare it with a neural model.

Dataset: [allenai/scifact](https://huggingface.co/datasets/allenai/scifact) — 5,183
abstracts and 1,409 expert-written claims.

## Project layout

```
src/
  download.py    Step 0  get the data
  explore.py     Step 1  look at the data
  preprocess.py  Step 2  text -> index terms
  sparse.py      Step 3  inverted index, TF-IDF, BM25 (from scratch)
  metrics.py     Step 4  nDCG, MRR, Recall, MAP (from scratch)
  evaluate.py    Step 4  run retrievers and compare them
  tune_bm25.py   Step 5  tune BM25 parameters on train
  dense.py       Step 6  neural embeddings + hybrid fusion
  search.py      Step 7  interactive search
data/            downloaded data and cached embeddings (not committed)
results/         metrics saved as JSON
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

All commands below are run from the project root, as modules (`python -m src.xxx`).

---

## Step 0 — Get the data

```powershell
python -m src.download
```

The HuggingFace page is backed by a loading script (`scifact.py`) that downloads a
tarball from AllenAI. Newer versions of the `datasets` library no longer run such
scripts, so we download the same tarball directly into `data/scifact/`.

## Step 1 — Explore the data

```powershell
python -m src.explore
```

Every IR task needs three things — this is called a **test collection**:

| Piece | In SciFact | File |
|---|---|---|
| Documents | 5,183 abstracts (title + sentences) | `corpus.jsonl` |
| Queries | claims, e.g. *"1/2000 in UK have abnormal PrP positivity."* | `claims_*.jsonl` |
| Relevance judgments (**qrels**) | the doc ids under each claim's `evidence` | `claims_*.jsonl` |

Things you'll notice:
- **Test claims have no labels** (they're hidden for a leaderboard). So we *tune on
  train* and *report on dev*.
- About 1/3 of claims have empty `evidence` (not enough info). They have no relevant
  doc, so they can't be scored and are skipped → 505 train and 188 dev queries.
- Almost every query has exactly **one** relevant document. The job is to put that one
  document at the very top.

## Step 2 — Preprocessing

```powershell
python -m src.preprocess "Microglia are activated in Alzheimer's disease"
```

Lowercase → split into tokens → drop stop words ("are", "in") → stem with Porter
("activated" → "activ"). Stemming lets a query word match other forms of the same
word in documents. Exactly the same function must be applied to documents *and* queries.

## Step 3 — Inverted index, TF-IDF and BM25

Read [src/sparse.py](src/sparse.py). The **inverted index** maps
`term → {document: term frequency}`. A query only touches the postings of its own
terms, which is why search engines are fast.

Two ranking functions on top of the same index:

- **TF-IDF + cosine** (vector space model): a term counts more if it is frequent in
  the document (TF) and rare in the collection (IDF).
- **BM25**: like TF-IDF, but term frequency *saturates* (parameter `k1`) and long
  documents are penalized (parameter `b`). BM25 is still the standard baseline.

## Step 4 — Evaluate

```powershell
python -m src.evaluate
```

Read [src/metrics.py](src/metrics.py). Metrics on the 188 dev queries:

| model | nDCG@10 | MRR@10 | P@1 | Recall@10 | Recall@100 | MAP |
|---|---|---|---|---|---|---|
| TF-IDF | 0.769 | 0.732 | 0.633 | 0.909 | 0.978 | 0.724 |
| BM25 (k1=1.2, b=0.75) | **0.833** | 0.801 | 0.718 | 0.941 | 0.978 | 0.796 |

How to read it: **P@1 = 0.72** → for 72% of claims the first result is the right
abstract. **Recall@100 = 0.98** → the right abstract is almost always somewhere in the
top 100, so the problem is *ranking*, not *finding*. That's why re-ranking and neural
models (later steps) are so common in practice.

> These scores look higher than published BEIR numbers (~0.67 for BM25) because we
> evaluate only claims with evidence, and BEIR uses a different query set. Compare
> models *within* this project, not with the leaderboard.

## Step 5 — Tune BM25 (properly)

```powershell
python -m src.tune_bm25
python -m src.evaluate --k1 1.2 --b 0.25
```

Grid search over `k1` and `b` on **train**. Best on train: `k1=1.2, b=0.25`
(0.862). On dev it gives **0.824** — slightly *worse* than the defaults (0.833).

Lesson: differences of ~0.01 are within noise for a few hundred queries, and a
setting that wins on one split doesn't always transfer. This is exactly why you never
tune on the set you report on — otherwise you'd have fooled yourself.

## Step 6 — Dense retrieval and hybrid

```powershell
python -m src.evaluate --dense
```

Read [src/dense.py](src/dense.py). A neural encoder (`BAAI/bge-small-en-v1.5`, 33M
parameters) turns every text into a 384-dimensional vector; the score is cosine
similarity. The first run downloads the model and encodes the corpus (a few minutes
on a CPU), then caches embeddings in `data/embeddings/`.

**Hybrid** combines BM25 and dense with **Reciprocal Rank Fusion**:
`score(d) = Σ 1 / (60 + rank(d))`. It uses only ranks, so the very different score
scales don't matter.

| model | nDCG@10 | MRR@10 | P@1 | Recall@10 | Recall@100 | MAP |
|---|---|---|---|---|---|---|
| TF-IDF | 0.769 | 0.732 | 0.633 | 0.909 | 0.978 | 0.724 |
| BM25 (k1=1.2, b=0.25) | 0.824 | 0.796 | 0.713 | 0.923 | 0.978 | 0.792 |
| Dense (bge-small) | **0.860** | 0.832 | **0.766** | **0.957** | 0.992 | **0.829** |
| Hybrid (RRF) | 0.858 | **0.833** | 0.750 | 0.953 | **0.995** | 0.826 |

The dense model wins on ranking quality; hybrid gives the best recall but doesn't beat
dense at the top here.

## Step 7 — Try it yourself

```powershell
python -m src.search            # BM25
python -m src.search --dense    # neural
```

Type claims and compare the two. Try paraphrases with no word overlap (e.g.
"heart attack" vs "myocardial infarction") — that's where dense retrieval shines.

---

## Exercises to go further

1. **Ablation**: turn off stemming or stop-word removal in `tokenize` — how much
   does BM25 change?
2. **Error analysis**: print dev claims where BM25 ranks the relevant doc below 10.
   Why did it fail? Does the dense model fix them?
3. **Title boosting**: index the title twice. Does it help?
4. **Re-ranking**: take BM25's top 100 and re-score with a cross-encoder
   (`sentence_transformers.CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`).
5. **Statistical significance**: run a paired t-test on per-query nDCG@10 between
   BM25 and Dense. Is the difference real?
6. **Bigger models**: swap `DEFAULT_MODEL` in `dense.py` for `BAAI/bge-base-en-v1.5`.
