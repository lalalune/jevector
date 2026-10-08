# JEVECTOR / PROFILE MATCHING

Explicit profile values, reciprocal requirements, and measured matching results. Results use saved API responses.

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

## Decision vectors with HNSW

| Method | Validation top-1 | Tie-adjusted top-1 | Candidate recall vs exact | All returned violations | No-match rejected |
|---|---|---|---|---|---|
| Clef 64 | 8/8 | 4.00/8 | 100.0% | 0/16 | 4/4 |
| Clef 256 | 8/8 | 8.00/8 | 100.0% | 0/16 | 4/4 |
| Jev 64 | 8/8 | 4.00/8 | 100.0% | 0/16 | 4/4 |
| Jev 256 | 8/8 | 8.00/8 | 100.0% | 0/16 | 4/4 |

Decision models encode the self, prefer, require, and exclude channels. HNSW compares cross-role projections of these values. Reciprocal rules then filter and rerank candidates.
52 newly worded synthetic profiles in eight pair families. Four families are development data; four are validation data. The validation queries include eight paired searches and four different no-match cases.
The displayed HNSW run retrieves up to 32 of 51 other profiles, with ef=64. Candidate recall is measured across all 20 queries against exact reciprocal matching.
The 64-value schema cannot represent cycling. The unsupported optional preference remains in the score denominator and is reported. The 256-value schema can distinguish these alternatives.
Tie-adjusted credit divides one correct result by the number of equal best scores. This prevents stable IDs from creating apparent ranking accuracy.
Mandatory conditions must use supported catalog IDs or explicit rules. Numeric facts and location facts are declared input; models do not infer them.
These are synthetic qualification results, not evidence of real relationship outcomes. An independent reference evaluator supplies expected eligibility and ranking.

## Qualification reference methods

| Method | Validation partner top-1 | No-match rejected |
|---|---|---|
| BM25 | 1/8 | 0/4 |
| BGE-small 384 | 0/8 | 0/4 |
| BM25 + BGE-small cosine rerank | 0/8 | 0/4 |
| BM25 + BGE RRF | 1/8 | 0/4 |
| BGE-small 384 with prefix | 0/8 | 0/4 |
| Declared attributes + rules (no model) | 8/8 | 4/4 |

BM25 and BGE receive the descriptions, declared facts, requested conditions, and mandatory rules. They rank by reciprocal similarity without enforcing conditions or abstaining.
The cosine rerank method sorts ten BM25 candidates using BGE-small. It is not a BGE cross-encoder.
The declared-attribute control receives source attribute values instead of text. It isolates rule matching from model extraction; it is not a blind retrieval baseline.

## HNSW candidate checks

| Method | Candidates | Eligible candidate recall | Exact top-5 list agreement |
|---|---|---|---|
| Clef 64 | 8 | 100.0% | 100.0% |
| Clef 64 | 16 | 100.0% | 100.0% |
| Clef 64 | 32 | 100.0% | 100.0% |
| Clef 64 | 51 | 100.0% | 100.0% |
| Clef 256 | 8 | 100.0% | 100.0% |
| Clef 256 | 16 | 100.0% | 100.0% |
| Clef 256 | 32 | 100.0% | 100.0% |
| Clef 256 | 51 | 100.0% | 100.0% |
| Jev 64 | 8 | 100.0% | 100.0% |
| Jev 64 | 16 | 100.0% | 100.0% |
| Jev 64 | 32 | 100.0% | 100.0% |
| Jev 64 | 51 | 100.0% | 100.0% |
| Jev 256 | 8 | 100.0% | 100.0% |
| Jev 256 | 16 | 100.0% | 100.0% |
| Jev 256 | 32 | 100.0% | 100.0% |
| Jev 256 | 51 | 100.0% | 100.0% |

No exact fallback was used for these measurements. Returning no HNSW result does not prove there is no eligible profile outside the candidate pool.

## HNSW on 1,024 structured profiles

| Values | Candidates | Exact top-5 ID recall | Top-5 score agreement | HNSW + checks | Exact checks |
|---|---|---|---|---|---|
| 64 | 16 | 40.6% | 0.0% | 1.48 ms | 48.18 ms |
| 64 | 64 | 78.3% | 50.0% | 2.77 ms | 48.18 ms |
| 64 | 128 | 88.4% | 75.0% | 5.56 ms | 48.18 ms |
| 256 | 16 | 37.0% | 0.0% | 2.75 ms | 107.78 ms |
| 256 | 64 | 69.9% | 37.5% | 4.78 ms | 107.78 ms |
| 256 | 128 | 80.8% | 43.8% | 9.74 ms | 107.78 ms |

Sixteen queries per size. This is a structured synthetic index test, not a live model benchmark. Both paths use the actual reciprocal matcher.
Shared validation and decoding are excluded from query timings. HNSW index construction is excluded and recorded in the JSON artifact. These single-run local timings are not a service latency guarantee.
Larger candidate pools recover more exact results. Returned candidates pass the same checks, but approximate retrieval can omit better candidates. Equal-scoring candidate IDs may differ.
