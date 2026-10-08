# Jevector search

The main example is now reciprocal profile matching. See [matching instructions](../matching/README.md) and the [project README](../README.md). The commands below run the separate support-search example.

Use explicit questions to create document vectors. Compare retrieval with BM25, BGE-small, RRF, and a cross-encoder.
Read the [results](RESULTS.md) before choosing a method. BGE-small leads the current public evaluation.

## Install

Run these commands from the repository root:

```sh
python3 -m venv .venv-search312
.venv-search312/bin/python -m pip install -r search_demo/requirements.txt
```

The local wrapper is `search_demo/python`. On this Mac, it supplies the Homebrew expat library path.
If local Python fails to load expat during environment creation, run:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib python3.12 -m venv .venv-search312
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib uv pip install \
  --python .venv-search312/bin/python -r search_demo/requirements.txt
```

## Build the small support example

```sh
export CLOUDFLARE_ACCOUNT_ID='YOUR_ACCOUNT_ID'
search_demo/python -m search_demo.fixtures
search_demo/python -m search_demo.run --provider clef --account "$CLOUDFLARE_ACCOUNT_ID"
search_demo/python -m search_demo.run --provider jev --prompt-key
search_demo/python -m search_demo.embedding_baseline --account "$CLOUDFLARE_ACCOUNT_ID"
search_demo/python -m search_demo.rerank_baseline
```

Use `--dimensions 64` or `--dimensions 256` to select one size. The default builds both.
Clef batches at most 64 questions. Jev accepts 256 questions per request.
The 256-dimension vectors do not improve top-1 on the saved support example.

The support data has 16 authored articles and 20 authored queries.
Its task catalog closely matches the examples. It is a development fixture, not an independent benchmark.

## Search

```sh
search_demo/python -m search_demo.search \
  --index search_demo/runs/clef-64/index \
  --account "$CLOUDFLARE_ACCOUNT_ID" \
  --query 'How do I recover access after losing my authentication device?' \
  --k 5 --save output/search.json
```

The command encodes the query and searches the saved index. A matching cache entry prevents a new model call.
A low-activation flag is a heuristic. It is not a validated rejection rule.

## Requirements and exclusions

Create `output/constraints.json` with explicit dimension IDs:

```json
{
  "required": {"account.0.goal": 0.7},
  "excluded": {"data.1.goal": 0.5}
}
```

This requires account-recovery coverage of at least 0.7 and excludes account-erasure coverage of at least 0.5.
The command does not infer these constraints from the query text.

```sh
search_demo/python -m search_demo.search \
  --index search_demo/runs/clef-64/index \
  --account "$CLOUDFLARE_ACCOUNT_ID" \
  --query 'Recover access to my account.' \
  --constraints output/constraints.json --scoring coverage --k 5
```

Constraints filter documents before ranking. Unknown dimension IDs fail validation.
The thresholds act on model outputs, so a constraint does not guarantee the underlying fact.
Use authoritative metadata for access permissions and other exact restrictions.

## Scores

| Score | Calculation | Search |
|---|---|---|
| Cosine | Compare vector directions | HNSW |
| Coverage | `sum(q[i] * d[i]) / sum(q[i])` | Exact scan |

Coverage does not reduce the score when a document contains additional unrequested content.
It does not automatically penalize forbidden content. Use explicit exclusions for that requirement.
Coverage is an alternative objective, not a demonstrated accuracy improvement. Its measured controls appear in the full report.

Use `--min-score` only with a threshold calibrated for the domain and score type.
The public evaluation calibrates on separate queries and reports false acceptances. Rejection remains imperfect.

## Index settings

| Setting | Default |
|---|---|
| Distance | Cosine |
| M | 16 |
| Construction effort | 200 |
| Search effort | 64 |
| Random seed | 42 |
| Threads | 1 |

`SearchIndex.build(..., ef_search=64)` sets search effort independently of corpus size.
Search raises it to at least the requested result count.
The index validates compatible models, schemas, roles, and saved-file checksums.
Exact coverage search does not use cosine HNSW candidates because they can omit the best coverage result.

## Evaluation and viewer

Follow the [evaluation protocol](../evaluation/README.md) to run the public and profile tests.
Then generate the viewer:

```sh
search_demo/python -m search_demo.render
python3 -m http.server 8876 --bind 127.0.0.1 --directory search_demo
```

Open `http://127.0.0.1:8876/`.
The page shows aggregate measurements. It contains no query inspector or vector-value controls.
The benchmark generator also updates the README tables from the saved results.

## Files

| File | Contents |
|---|---|
| `schema.py` | Support-task questions |
| `engine.py` | Vector conversion, persisted index, scoring, and constraints |
| `search.py` | Query command |
| `data.json` | Synthetic support inputs and labels |
| `runs/` | Saved support vectors and results |
| `../evaluation/` | Public evaluation, baseline audit, and reference methods |
| `../runs/evaluation/` | Evaluation artifacts |

Change the schema version when changing dimensions. Rebuild affected indexes.
Do not mix vectors from different models or schemas.
