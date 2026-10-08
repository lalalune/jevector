# Jevector

Jevector uses Clef-flash or Jev to encode profile attributes and requested partner attributes as decision vectors.
HNSW retrieves candidate profiles. Reciprocal checks reject candidates that fail mandatory conditions. The remaining candidates are ranked by preference coverage.

**Decision-model encoding → HNSW retrieval → reciprocal checks → ranked results.**

[Viewer](search_demo/index.html) · [Usage and rules](matching/README.md) · [Fixed findings](matching/WEAKNESSES.md)

<!-- benchmarks:start -->

## Retrieval comparison

| Method | Top-1 | Top-5 | Top-10 | MRR@10 | nDCG@10 | Tie-adjusted top-1 |
|---|---|---|---|---|---|---|
| Clef 64 | 50.0% | 100.0% | 100.0% | 0.750 | 0.815 | 50.0% |
| Clef 256 | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 | 100.0% |
| Jev 64 | 87.5% | 100.0% | 100.0% | 0.938 | 0.954 | 44.8% |
| Jev 256 | 87.5% | 100.0% | 100.0% | 0.938 | 0.954 | 81.2% |
| BM25 | 12.5% | 12.5% | 12.5% | 0.125 | 0.125 | 12.5% |
| BGE-small 384 | 0.0% | 12.5% | 12.5% | 0.042 | 0.062 | 0.0% |
| BM25 + BGE-small | 0.0% | 0.0% | 12.5% | 0.018 | 0.042 | 0.0% |

Same 52-profile gallery and eight validation queries from four held-out pair families. Each query has one intended partner. Self-matches are excluded. Four no-match queries are excluded from these ranking metrics. Development and full-set metrics and ranked IDs are saved in runs/matching-v2/comparison.json.
Top-k is the fraction of queries with the intended partner in the first k results. With one relevant partner, this is also Recall@k. MRR@10 measures reciprocal partner rank; nDCG@10 discounts partner rank. Missing partners score zero. Higher is better for every column.
Tie-adjusted top-1 is expected credit under random ordering of equal scores (absolute tolerance 0.0000001). Other columns use descending score, then stable profile ID. Small numeric differences can affect rank.
No mandatory-condition filtering or preference-coverage reranking is used in this table. Clef and Jev use the existing decision-vector projection and HNSW with 32 candidates. BGE-small uses 384-dimensional cosine scores over the full gallery. BM25 uses lexical scores over the full gallery. Scores account for both profile-to-request directions.
BM25 + BGE-small selects ten BM25 candidates, then sorts them by BGE-small cosine score. It is not a cross-encoder. BGE uses unprefixed requests, token-bounded chunks, normalized mean pooling across chunk embeddings, and cached Cloudflare outputs from qualification.
Clef and Jev receive a task-specific attribute catalog. BGE and BM25 receive the profile and request text, including declared facts and rules. This compares configured retrieval systems, not equal training data or equal task supervision. Clef and Jev are trained models too; the distinction is decision vectors versus general-purpose embeddings.
On this synthetic validation set, decision vectors retrieve the intended partners more reliably than these BGE and BM25 configurations. This does not establish an advantage on general search or real profiles. The data was built around the catalog, includes near-duplicate alternatives, and has only four independent validation families. The 256-value advantage tests one added interest: cycling.

## Complete pipeline comparison

| Method | Top-1 | Top-5 | Top-10 | MRR@10 | nDCG@10 | Tie-adjusted top-1 |
|---|---|---|---|---|---|---|
| Clef 64 | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 | 50.0% |
| Clef 256 | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 | 100.0% |
| Jev 64 | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 | 50.0% |
| Jev 256 | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 | 100.0% |
| BM25 | 12.5% | 12.5% | 12.5% | 0.125 | 0.125 | 12.5% |
| BGE-small 384 | 0.0% | 12.5% | 12.5% | 0.042 | 0.062 | 0.0% |
| BM25 + BGE-small | 0.0% | 0.0% | 12.5% | 0.018 | 0.042 | 0.0% |

Same 52-profile gallery and eight validation queries from four held-out pair families. Each query has one intended partner. Self-matches are excluded. Four no-match queries are excluded from these ranking metrics. Development and full-set metrics and ranked IDs are saved in runs/matching-v2/comparison.json.
Top-k is the fraction of queries with the intended partner in the first k results. With one relevant partner, this is also Recall@k. MRR@10 measures reciprocal partner rank; nDCG@10 discounts partner rank. Missing partners score zero. Higher is better for every column.
Tie-adjusted top-1 is expected credit under random ordering of equal scores (absolute tolerance 0.0000001). Other columns use descending score, then stable profile ID. Small numeric differences can affect rank.
Clef and Jev now apply reciprocal mandatory checks and preference-coverage ranking after HNSW. The three reference methods keep their retrieval rankings. This table measures the complete configured systems; gains here cannot be attributed to vector encoding alone.
The detailed condition and no-match checks remain below. Use the retrieval comparison above to assess decision vectors without these rule advantages.

<!-- benchmarks:end -->

## Reproduce the comparison

```sh
search_demo/python -m matching.comparison
search_demo/python -m search_demo.render
```

The comparison reads saved Clef, Jev, and BGE outputs. It makes no API requests. Missing BGE cache entries stop the run. Use the qualification command below to obtain new model outputs.

The saved comparison includes development, validation, and full-set metrics and ranked IDs. Do not select settings from validation results and then describe the same results as an independent test.

## Install

```sh
python3 -m venv .venv-search312
.venv-search312/bin/python -m pip install -r search_demo/requirements.txt
```

`search_demo/python` runs this environment. On macOS, it supplies the Homebrew expat library path when available.
For Cloudflare, set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`, or use an existing Wrangler login.
For Jev, use `JEV_API_KEY`, `TYPESAFE_API_KEY`, or the hidden `--prompt-key` prompt.

## Extract a profile

Each input needs a stable `profile_id`, `about_me` text, and `looking_for` lists.
Use catalog IDs or exact catalog descriptions for conditions. Arbitrary mandatory text is rejected instead of silently discarded.
Models encode all four channels. The original explicit condition lists also control mandatory checks and preference accounting.

```sh
search_demo/python -m matching.cli extract --provider clef --dimensions 256 \
  --account "$CLOUDFLARE_ACCOUNT_ID" \
  --input matching/example-v2.json --output output/profile.json

search_demo/python -m matching.cli extract --provider jev --dimensions 256 --prompt-key \
  --input matching/example-v2.json --output output/profile-jev.json
```

| Values | Attributes | Channels |
|---|---:|---|
| 64 | 16 | Self, prefer, require, exclude |
| 256 | 64 | Self, prefer, require, exclude |

Each profile uses four channel requests with either provider. Its stored record also contains evidence masks, identity, rules, and provenance.
The 64 or 256 value count is not the total record size.

## Retrieve with HNSW

After the qualification run below, this example uses saved live vectors without another API request:

```sh
search_demo/python -m matching.cli rank \
  --query runs/matching-v2/jev-256/family0-persona.json \
  --gallery runs/matching-v2/jev-256/*.json \
  --method hnsw --candidates 32 --ef 64 --k 5
```

Use `--method exact` to check every candidate.
Use `--exact-fallback` to run an exact scan when HNSW returns fewer than `--k` eligible results.
That fallback does not guarantee exact results when the approximate path already returns `--k` results.

The output reports candidate count, fallback use, and whether the search was exhaustive.
An empty approximate result does not establish that no valid match exists.
Larger candidate pools can improve recall at additional cost.

The library provides `matching.ann.ProfileIndex` for repeated queries against one in-memory index.
The CLI rebuilds its index on each invocation. It does not yet persist or incrementally update an index.

## Explicit rules

The rule engine supports `all`, `any`, `if`/`then`, numeric bounds, and geographic distance.
Missing facts reject a mandatory rule. Numeric facts and locations are supplied explicitly; models do not infer them.
See [rule examples](matching/README.md).

## Run qualification

```sh
search_demo/python -m matching.qualification_data
search_demo/python -m matching.qualification --account "$CLOUDFLARE_ACCOUNT_ID" --prompt-key
search_demo/python -m matching.index_scale
search_demo/python -m search_demo.render
```

The qualification uses 52 independently worded synthetic profiles and disjoint development/validation families.
The index test uses 1,024 structured synthetic profiles. It measures retrieval mechanics, not model accuracy.
BM25, BGE-small, cosine reranking, RRF, and a declared-attribute control are reported separately.
The declared-attribute control measures matching without model extraction.

These experiments do not validate real-world compatibility. They do not establish that HNSW always preserves exact reciprocal rankings.

## Tests

```sh
search_demo/python -m unittest -v matching.test_production matching.test_matching \
  test_jevector search_demo.test_search evaluation.test_evaluation
```

## Earlier experiments

The earlier 164-profile benchmark and its audits remain available as historical results.
Its vectors use a different decoder and lack the current provenance fields. The current CLI rejects them.
The historical implementation is isolated in `matching/legacy_engine.py`; it is not the active CLI path.

- [Earlier Clef diagnosis](matching/TROUBLESHOOTING.md)
- [Earlier baseline audit](matching/BASELINE_AUDIT.md)
- [Earlier retrieval experiments](evaluation/README.md)

The old `jevector.py` extraction CLI and `match.py` also use a separate personality schema.
Use `matching.cli` for the current paired-profile/HNSW workflow.
