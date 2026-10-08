# Evaluation

The evaluation reports failures and reference methods beside question-vector results.
It does not claim that a question rubric replaces general embeddings.

## Public protocol

Source: [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact), distributed under CC-BY-SA-4.0.
The download URL follows the [BEIR example](https://github.com/beir-cellar/beir).

1. Define the 64-question science rubric in `science_schema.py`.
2. Save the rubric hash and protocol before reading the downloaded labels.
3. Sample 80 labeled queries with seed 20261006.
4. Retain all labeled relevant documents for 60 queries.
5. Remove all labeled relevant documents for the other 20 queries.
6. Add seeded random distractors to make 500 documents.
7. Use 15 answered and 5 unanswerable queries for threshold calibration.
8. Evaluate retrieval on the other 45 answered queries.
9. Evaluate rejection on those 45 queries and the other 15 unanswerable queries.

Queries that share relevant documents are replaced through a deterministic disjoint selection rule when needed.
The protocol, schema hash, corpus archive hash, and selected IDs are saved locally.
The question rubric is unchanged after evaluation. No test labels are passed to provider models.
Public source material may be present in model training data. That exposure is unknown.

This subset is easier than the full corpus because it uses random distractors.
It is not an official SciFact result. Missing relevance labels can affect rejection measurements.
The relevance labels come from the public dataset; they are not independent validation of the profile rubric.

## Methods

| Method | Settings |
|---|---|
| BM25 | Lowercase word tokens; k1=1.2; b=0.75 |
| BGE-small | CLS pooling; query prefix for scientific claims; 384 dimensions |
| RRF | Full BM25 and BGE rankings; constant 60 |
| Cross-encoder | BGE-reranker-base on 20 RRF candidates |
| Clef and Jev | Same fixed 64-question rubric; exact cosine comparison |
| Binary tags | Threshold the same extracted values at 0.5; rank by query-tag coverage |

The public vector comparison uses exact search to isolate representation quality from index error.
The binary-tag control still needs the same extraction calls. It does not demonstrate lower inference cost.
The support classifier uses one categorical question per text. It has a separate measured result.

## Token handling

BGE inputs use the pinned BGE-small tokenizer. No input is silently shortened by this evaluator.
Long text is split into chunks of at most 460 tokens before the optional prefix is added.
The code checks every request against the 512-token limit.
Normalize each chunk vector, average the vectors, then normalize the result.

The cross-encoder uses its own pinned tokenizer.
Candidate text is explicitly shortened when needed to fit the query-document pair within 512 tokens.
The result artifact counts shortened candidate inputs. Reranker truncation remains a limitation.

The profile audit compares JSON and plain text, each with and without the query prefix.
It preserves field names and explicit rejections. It excludes person IDs and expected labels.
All 96 legacy profile inputs fit the model limit. Their low scores were not caused by truncation.
All variants are reported. No per-set winner is selected as the baseline.

## Costs and timings

Successful Clef and Jev responses supply input-token counts.
BGE token counts come from the pinned tokenizer and include chunk prefixes and special tokens.
Cost estimates exclude failures, retries, hosting, and storage. They are not invoices.

Published rates checked for this evaluation:

- [BGE-small](https://developers.cloudflare.com/workers-ai/models/bge-small-en-v1.5/): $0.0202 per million input tokens.
- [Clef-flash](https://developers.cloudflare.com/workers-ai/models/clef-flash/): $0.09 per million input tokens.
- [Jev](https://docs.typesafe.ai/models): $0.042 per million input tokens.
- [BGE-reranker-base](https://developers.cloudflare.com/workers-ai/models/bge-reranker-base/): $0.00311 per million input tokens.

The report separates indexing input cost from estimated input cost per 1,000 queries.
BGE single-query latency uses ten sequential requests. Clef and Jev use six workers over 80 queries.
Reranker requests use four workers. Concurrency differs, so these timings are not controlled provider speed comparisons.
Cached records retain their original request durations. Cached execution time is not live latency.

## Index mechanics

Run `evaluation.index_benchmark` to test 500, 5,000, and 20,000 random vectors.
It tests 64, 256, and 384 dimensions at search effort 16, 64, and 128.
The index uses one thread, M=16, and construction effort 200.
Compare each top-ten set with exact search over the same matrix.
These random-vector results measure index mechanics only. They do not measure relevance or production capacity.

## Reproduce

Run from the repository root after authentication:

```sh
search_demo/python -m evaluation.profile_audit --account "$CLOUDFLARE_ACCOUNT_ID"
search_demo/python -m evaluation.public_data
search_demo/python -m evaluation.public_run --account "$CLOUDFLARE_ACCOUNT_ID" \
  --providers clef jev --prompt-key
search_demo/python -m evaluation.support_controls --account "$CLOUDFLARE_ACCOUNT_ID"
search_demo/python -m evaluation.query_latency --account "$CLOUDFLARE_ACCOUNT_ID"
search_demo/python -m evaluation.index_benchmark
search_demo/python -m search_demo.render
```

Use the same protocol and schema to reproduce a result. Use a new version for any changed schema.
The model aliases can change their underlying implementation. Cached responses preserve the tested outputs.

## Remaining limits

The scientific rubric has broad categories and can lose names, numbers, and specific mechanisms.
The synthetic profile generator shares vocabulary with the profile schema. Perfect retrieval is development evidence only.
The current data does not show a retrieval benefit from 256 dimensions.
A score threshold does not reliably reject all unsupported queries.
Explicit dimension constraints act on model predictions, not verified facts.
No evaluation establishes relationship compatibility, personality accuracy, or general superiority over embeddings.
