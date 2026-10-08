# BM25 and BGE baseline audit

The saved benchmark rankings and metrics reproduce exactly. The main benchmark remains unchanged.

## Implementation checks

- BM25 agrees with a separate scalar formula. Maximum absolute score difference: 2.49e-14.
- BM25 uses k1=1.2, b=0.75, positive log-IDF, lowercase alphanumeric tokens, and query term frequency.
- Reciprocal BM25 normalizes each directional score by its query maximum, then averages both directions.
- Excluding the requester from that normalization maximum did not change top-1 accuracy.
- BGE uses bge-small-en-v1.5, 384 values, CLS pooling, and unit-normalized vectors.
- Direct two-way cosine calculations agree with the matrix implementation.
- Both prefixed and unprefixed query rankings reproduce. Passages have no query prefix.
- All 164 self descriptions and preference texts fit within 512 tokens. No chunk aggregation was needed.
- Self profiles are excluded. Candidate IDs retain the input order.
- Three live semantic controls retrieved their intended documents. Reversing an API batch preserved document identities.
- RRF and BM25 top-ten reranking reproduce exactly.
- The rerank baseline uses BGE-small cosine scores. It does not use the separate BGE cross-encoder reranker.

## Input-format tests

These are post-hoc diagnostic tests on the same synthetic profiles. No source labels or pair IDs enter the text transformations.

| Method | Partner top-1 | Partner recall@5 |
|---|---:|---:|
| BM25 original | 4/32 | 100.0% |
| BM25 exclude self from normalization | 4/32 | 100.0% |
| BGE original no prefix | 1/32 | 3.1% |
| BGE original query prefix | 1/32 | 3.1% |
| BM25 explicit query | 4/32 | 53.1% |
| BGE explicit query, no prefix | 0/32 | 3.1% |
| BGE explicit query, query prefix | 1/32 | 6.2% |
| BM25 explicit query and rewritten self | 1/32 | 96.9% |
| BGE explicit query and rewritten self, no prefix | 0/32 | 3.1% |
| BGE explicit query and rewritten self, query prefix | 0/32 | 3.1% |

The explicit-query variant writes required and excluded conditions as sentences.
The rewritten-self variant replaces fixture sentence wrappers while retaining positive and negative statements.
These changes do not improve BGE on this fixture.

## Interpretation

BM25 puts every intended partner in the first five results. Its low top-1 score is a ranking failure among near-duplicates.
The fixture mentions the same core attributes in positive and negative statements. BM25 does not enforce those signs.
Excluded traits also occur as words in the flattened request. Word overlap does not make them hard exclusions.
BGE similarity is not a mandatory-condition check. Averaging both directions does not create one.
The corpus tests exact logical distinctions between related profiles. It is not a representative general retrieval benchmark.
Jev/Clef methods also include explicit constraint logic. Unfiltered BM25 and BGE do not have that logic.
The existing constraint-plus-BGE rows isolate ranking after the same model-extracted filters.
BM25 has ties: random tie order gives 3/32 expected correct first results, versus 4/32 under the saved stable order.

## Reproduce

```sh
search_demo/python -m matching.baseline_audit --account "$CLOUDFLARE_ACCOUNT_ID" --format-tests
```

Raw measurements: `runs/matching/baseline-audit.json`.

Sources: [Cloudflare BGE configuration](https://developers.cloudflare.com/workers-ai/models/bge-small-en-v1.5/) and [BAAI model usage](https://huggingface.co/BAAI/bge-small-en-v1.5).
