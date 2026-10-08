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

## Historical reciprocal profile matching

| Method | Partner top-1 | Partner recall@5 | Both ways | Hard violations ↓ | Paired abstentions ↓ | No-match rejected ↑ |
|---|---|---|---|---|---|---|
| Clef 64 | 22/32 | 68.8% | 11/16 | 0/25 | 7/32 | 4/4 |
| Clef 256 | 16/32 | 50.0% | 8/16 | 0/18 | 14/32 | 4/4 |
| Jev 64 | 32/32 | 100.0% | 16/16 | 0/32 | 0/32 | 4/4 |
| Jev 256 | 32/32 | 100.0% | 16/16 | 0/32 | 0/32 | 4/4 |
| BM25 reciprocal | 4/32 | 100.0% | 1/16 | 29/36 | 0/32 | 0/4 |
| BGE reciprocal, no prefix | 1/32 | 3.1% | 0/16 | 34/36 | 0/32 | 0/4 |
| BGE reciprocal, query prefix | 1/32 | 3.1% | 0/16 | 35/36 | 0/32 | 0/4 |
| BM25 + BGE RRF | 2/32 | 40.6% | 0/16 | 31/36 | 0/32 | 0/4 |
| BM25 top-10 + BGE rerank | 3/32 | 53.1% | 0/16 | 30/36 | 0/32 | 0/4 |
| Template parser control | 32/32 | 100.0% | 16/16 | 0/32 | 0/32 | 4/4 |

164 synthetic profiles: 16 intended pairs of different people, 96 hard distractors, 32 valid lower-ranked alternatives, and four no-match queries.
Each of the 32 paired people searches the other 163 profiles. Each has exactly two source-valid candidates: the partner and one lower-ranked alternative. The matcher checks both people’s mandatory conditions.
This is a development fixture. Prompts and the input contract were corrected after pilot runs. It is not a held-out evaluation.
Pair assignments and source attributes are evaluator-only. Models receive the person’s own description and stated partner preferences.
Hard violations count incompatible first results among returned first results. Paired abstentions count queries where a valid partner exists but no result is returned.
The template-parser control recognizes the fixture wording without a model. This measures fixture simplicity; it is not a general free-text parser.
The profiles are generated from explicit matching specifications. This tests rule-based compatibility, not real-world relationship outcomes.
BM25 and BGE average two directional scores. These unfiltered reference methods do not enforce mandatory conditions or reject no-match queries.

## Extracted constraints with BGE ranking

| Method | Partner top-1 | Partner recall@5 | Hard violations ↓ |
|---|---|---|---|
| Clef 64 constraints + BGE | 12/32 | 68.8% | 0/25 |
| Clef 256 constraints + BGE | 10/32 | 50.0% | 0/18 |
| Jev 64 constraints + BGE | 17/32 | 100.0% | 0/32 |
| Jev 256 constraints + BGE | 17/32 | 100.0% | 0/32 |

The same model-extracted reciprocal conditions filter candidates. BGE then ranks eligible profiles.
This separates the contribution of explicit constraints from the choice of ranking score. No ground-truth attributes are used for filtering.

## Reciprocal extraction checks

| Model | Channel | Yes precision | Yes recall | Known accuracy | Unknown accuracy |
|---|---|---|---|---|---|
| Clef 64 | self | 99.7% | 99.5% | 99.2% | 93.8% |
| Clef 64 | prefer | 100.0% | 100.0% | 99.6% | — |
| Clef 64 | require | 100.0% | 100.0% | 99.8% | — |
| Clef 64 | exclude | 100.0% | 99.2% | 99.6% | — |
| Clef 256 | self | 98.0% | 99.5% | 99.4% | 97.5% |
| Clef 256 | prefer | 100.0% | 100.0% | 99.7% | — |
| Clef 256 | require | 100.0% | 99.8% | 99.6% | — |
| Clef 256 | exclude | 100.0% | 99.4% | 99.9% | — |
| Jev 64 | self | 100.0% | 100.0% | 100.0% | 100.0% |
| Jev 64 | prefer | 100.0% | 100.0% | 100.0% | — |
| Jev 64 | require | 100.0% | 100.0% | 100.0% | — |
| Jev 64 | exclude | 100.0% | 100.0% | 100.0% | — |
| Jev 256 | self | 99.3% | 100.0% | 100.0% | 99.5% |
| Jev 256 | prefer | 100.0% | 100.0% | 99.9% | — |
| Jev 256 | require | 100.0% | 100.0% | 100.0% | — |
| Jev 256 | exclude | 100.0% | 100.0% | 100.0% | — |

Self attributes, optional wishes, requirements, and exclusions are scored separately against the source specification.
Positive precision and recall expose missed or invented conditions. Accuracy alone can hide errors when most conditions are absent.
The 64-value schema uses 16 attributes across four channels. The 256-value schema uses 64 attributes across those channels.
A single uncertain mandatory condition can reject a valid pair. High average attribute accuracy does not guarantee high match recall.
All mandatory fixture conditions use the core attributes. Extra attributes affect optional ranking. This schema differs from the older profile-retrieval schema.

## Reciprocal extraction cost

| Model | Input tokens | 164-profile input cost | Profile p50 |
|---|---|---|---|
| Clef 64 | 1,737,080 | $0.1563 | 2.018 s |
| Clef 256 | 6,596,072 | $0.5936 | 3.858 s |
| Jev 64 | 1,660,318 | $0.0697 | 1.183 s |
| Jev 256 | 5,890,206 | $0.2474 | 1.181 s |

Estimates use API input-token counts and published input rates. They exclude failed requests, retries, storage, and hosting.
Profiles are extracted once and reused for both sides of every match. Matching existing vectors requires no model call.
Profiles run concurrently. Each profile uses four sequential channel requests at either size; its duration is their sum.

## Public document retrieval

| Method | Top-1 | 95% interval | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| BM25 | 34/45 | 61%–86% | 90.9% | 0.822 | 0.829 |
| BGE-small 384 | 38/45 | 71%–92% | 94.4% | 0.877 | 0.879 |
| BM25 + BGE RRF | 36/45 | 66%–89% | 97.8% | 0.857 | 0.877 |
| RRF + cross-encoder | 35/45 | 64%–87% | 98.0% | 0.854 | 0.874 |
| Clef 64 | 16/45 | 23%–50% | 78.6% | 0.503 | 0.570 |
| Clef binary tags | 4/45 | 4%–21% | 43.9% | 0.200 | 0.254 |
| Jev 64 | 17/45 | 25%–52% | 67.0% | 0.476 | 0.511 |
| Jev binary tags | 9/45 | 11%–34% | 43.3% | 0.287 | 0.319 |

SciFact subset: 500 documents and 80 queries. The test split has 45 queries with labeled answers and 15 without labeled answers.
The remaining 20 queries calibrate rejection thresholds. Retrieval scores above use the 45 test queries with labeled answers.
The 64-question science rubric was fixed before loading this evaluation data. Both providers use the same rubric.
BGE-small leads this test. The rubric uses broad categories and may omit paper-specific details.
This is a sampled subset with random distractors, not the official full SciFact benchmark. The public labels were not written for this project.

## Answer rejection

| Method | Answered accepted ↑ | Unanswerable accepted ↓ | Balanced accuracy ↑ |
|---|---|---|---|
| BM25 | 62.2% | 13.3% | 74.4% |
| BGE-small 384 | 55.6% | 13.3% | 71.1% |
| BM25 + BGE RRF | 80.0% | 53.3% | 63.3% |
| RRF + cross-encoder | 66.7% | 6.7% | 80.0% |
| Clef 64 | 55.6% | 33.3% | 61.1% |
| Clef binary tags | 0.0% | 0.0% | 50.0% |
| Jev 64 | 68.9% | 66.7% | 51.1% |
| Jev binary tags | 71.1% | 53.3% | 58.9% |

Each method uses one score threshold selected on the calibration split only. The test has 45 answerable and 15 unanswerable queries.
Unanswerable means all labeled relevant documents were removed. Incomplete relevance labels can omit other valid answers.
Rejection remains unreliable for the question vectors. A high similarity score is not a correctness probability.

## Synthetic support example

| Method | Dimensions | Top-1 | Recall@3 |
|---|---|---|---|
| Clef | 64 | 18/20 | 95% |
| Clef | 256 | 18/20 | 95% |
| Jev | 64 | 19/20 | 100% |
| Jev | 256 | 19/20 | 100% |
| BM25 | — | 11/20 | 75% |
| BGE-small | 384 | 17/20 | 100% |
| BM25 → BGE cosine | 384 | 16/20 | 95% |
| Clef one-question classifier | 1 label | 18/20 | 90% |

16 authored documents and 20 authored queries. The examples closely match the task catalog. These are development results.
The one-question classifier selects a task label and retrieves documents with that label. It matches Clef vector top-1 accuracy here.
The 256-dimension vectors do not improve top-1 over 64 dimensions in this example. Use 64 as the starting point.
BM25 → BGE cosine selects 10 lexical candidates. It is not a cross-encoder. The public evaluation above includes RRF and a cross-encoder.

## Support scoring controls

| Method | Top-1 | Recall@10 |
|---|---|---|
| clef-64 binary tags | 17/20 | 95.0% |
| clef-64 coverage | 18/20 | 100.0% |
| clef-256 binary tags | 17/20 | 100.0% |
| clef-256 coverage | 18/20 | 100.0% |
| jev-64 binary tags | 15/20 | 90.0% |
| jev-64 coverage | 15/20 | 100.0% |
| jev-256 binary tags | 15/20 | 90.0% |
| jev-256 coverage | 15/20 | 100.0% |

These controls use the same saved extraction outputs as the support example. They do not require additional model calls.
Coverage ranks the weighted fraction of requested dimensions supplied by a document. Extra document content does not lower this score.
Coverage does not improve Clef top-1 and reduces Jev top-1 here. It is an optional scoring rule, not a superior default.

## Synthetic profile retrieval

| Method | Dimensions | Original top-1 | Additional top-1 |
|---|---|---|---|
| Clef | 64 | 32/32 | 16/16 |
| Clef | 256 | 32/32 | 16/16 |
| Jev | 64 | 32/32 | 16/16 |
| Jev | 256 | 32/32 | 16/16 |
| BM25, plain text | — | 14/32 | 16/16 |
| BGE, plain text, no prefix | 384 | 27/32 | 11/16 |
| BGE, plain text, query prefix | 384 | 23/32 | 13/16 |
| BM25 + BGE RRF | 384 | 27/32 | 13/16 |

Retrieve a second description of the same synthetic person. This does not measure compatibility between different people or personality accuracy.
The original set has 32 people and informed matcher development. The additional set has 16 people and shares the preference catalog.
Both BGE prefix variants are shown. No best-per-set BGE score is selected. Profile fields are preserved in the plain-text conversion.
All 96 legacy BGE inputs fit within 512 tokens. Formatting and prefix choices affected the scores; truncation did not explain the legacy failures.

## Profile baseline sensitivity

| Variant | Original top-1 | Additional top-1 |
|---|---|---|
| BM25 json | 14/32 | 16/16 |
| BGE json, no prefix, chunked | 20/32 | 9/16 |
| BGE json, query prefix, chunked | 19/32 | 3/16 |
| BM25 plain | 14/32 | 16/16 |
| BGE plain, no prefix, chunked | 27/32 | 11/16 |
| BM25 + BGE RRF | 27/32 | 13/16 |
| BGE plain, query prefix, chunked | 23/32 | 13/16 |

The chunking code uses the pinned BGE tokenizer. It limits each input to 512 tokens, including the prefix and special tokens.
Long texts use 460-token chunks. Normalized chunk vectors are averaged and normalized again. The profile fixtures require no chunks beyond one.

## Profile extraction checks

| Model | Dimensions | Preference accuracy | Controls |
|---|---|---|---|
| Clef | 64 | 94.10% | 17/17 |
| Clef | 256 | 97.07% | 20/20 |
| Jev | 64 | 94.99% | 16/17 |
| Jev | 256 | 98.15% | 19/20 |

These scores use the additional synthetic set. BM25 and BGE are retrieval methods and were not used to extract preferences.
Jev fails the contradictory dating-intent control. Expected: mixed. Received: no. Confirmed user preferences must not be replaced by model inference.

## Model cost and request latency

| Method | Query p50 | Query p95 | 500-doc input cost | 1,000-query input cost | Measurement |
|---|---|---|---|---|---|
| BGE-small 384 | 0.304 s | 0.817 s | $0.0034 | $0.0006 | 10 sequential queries; local token estimate |
| Clef 64 | 0.632 s | 0.844 s | $0.3095 | $0.5908 | 80 queries; 6 workers; API token usage |
| Jev 64 | 0.221 s | 0.329 s | $0.1290 | $0.2444 | 80 queries; 6 workers; API token usage |

Costs are estimates from input tokens and published rates, not invoices. Failed requests, retries, storage, and index hosting are excluded.
Recorded request durations include network time and any client retries. Concurrency differs as shown. These are not controlled service speed comparisons.
Indexing and query encoding are separate costs. Reducing vector dimensions does not establish lower total search cost.
Rates per million input tokens: BGE-small $0.0202; Clef-flash $0.09; Jev $0.042. Sources are linked in the evaluation instructions.

## Cross-encoder conditions

RRF selects 20 candidates. BGE-reranker-base scores each query-document pair. Median recorded rerank request: 0.705 seconds.
453 candidate inputs were shortened to fit a conservative token budget. The budget includes the query.
This truncation can affect results. Embedding vectors retain all text through chunk pooling.

## Index mechanics: 20,000 random vectors

| Dimensions | Search effort | Top-10 agreement | HNSW p50 ms | HNSW p95 ms | Exact p50 ms |
|---|---|---|---|---|---|
| 64 | 16 | 46.9% | 0.030 | 0.039 | 1.793 |
| 64 | 64 | 80.2% | 0.090 | 0.110 | 1.793 |
| 64 | 128 | 93.2% | 0.168 | 0.197 | 1.793 |
| 256 | 16 | 23.8% | 0.129 | 0.186 | 2.386 |
| 256 | 64 | 52.6% | 0.373 | 0.453 | 2.386 |
| 256 | 128 | 69.8% | 0.691 | 0.766 | 2.386 |
| 384 | 16 | 22.5% | 0.250 | 0.487 | 2.651 |
| 384 | 64 | 48.4% | 0.725 | 0.919 | 2.651 |
| 384 | 128 | 65.9% | 1.393 | 2.810 | 2.651 |

100 seeded random queries, one HNSW thread, M=16, and construction effort 200. This measures index mechanics, not semantic relevance.
Search effort is configurable and no longer grows automatically with corpus size. Higher effort trades speed for exact-search agreement.
The full artifact also includes 500 and 5,000 vectors. Local timings include result ordering and have no production service guarantee.
