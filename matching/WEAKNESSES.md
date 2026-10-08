# Matching approach audit

Status: historical findings reproduced before the matching-2 update. The active CLI now uses the fixes below. The original benchmark is preserved.

## Resolution

- Unsupported mandatory conditions: rejected by `engine.contract` before API calls. Optional gaps remain visible and penalized.
- Uncertainty and empty scores: fixed denominator, separate evidence coverage, no empty-profile perfect matches.
- Filename identity: replaced with required stable IDs and revision conflict checks.
- Concurrent writes: unique temporary paths and process-local request locks.
- Provenance: required model/provider fields, content hashes, and policy fingerprints.
- Shared oracle: replaced by `reference.py` in the new qualification test.
- Additional dimensions: new alternatives require a non-core attribute to break ties.
- Sample diversity: new wording, disjoint pair families, and four no-match causes. Human validation remains outside this synthetic experiment.
- Top-1-only safety: new reports count all returned violations and missed eligible candidates.
- Rules: alternatives, conjunctions, conditionals, numeric bounds, and distance are implemented.
- Retrieval: the active CLI uses HNSW, followed by reciprocal checks. Exact comparison and larger index measurements expose approximate recall loss.

Regression checks: `matching/test_production.py`. Current usage: [matching README](README.md).

The sections below retain the original findings and original line references.

## Confirmed defects

### P1: Unrepresented mandatory conditions disappear

`matching/engine.py:10` asks only questions in the selected catalog. No check requires each input condition to map to a supported question.
The 64-value schema has 16 attributes. Cycling is absent from that catalog.
A live Clef extraction with `require: ["enjoys cycling"]` accepted a candidate whose text explicitly says they do not enjoy cycling.
The candidate received score 1.0, without an unsupported-condition warning.
This is a representation failure. Even perfect answers to the available questions cannot enforce an absent attribute.
The 256-value schema has the same problem for conditions outside its 64 attributes.

Required correction: account for every mandatory input condition. Reject or defer unresolved conditions. Do not silently turn them into no requirement.
Use structured rules for unsupported conditions rather than assuming a larger vector will contain them.

### P1: Less evidence can produce a better score

`matching/engine.py:43` drops optional conditions decoded as unknown. This removes them from the denominator.
In the reproduction, a candidate satisfies one of two optional preferences. The reciprocal score is 0.75.
Changing the unmet preference to unknown raises that score to 1.0.
Two profiles with unknown self attributes and no conditions also receive an eligible result with score 1.0.
That is full coverage of an empty preference set, not evidence of compatibility. The output does not distinguish those cases.

Required correction: preserve specified preference counts and report evidence coverage separately. Define an insufficient-evidence result and test ranking monotonicity.

### P2: File names are used as person identities

`matching/cli.py:20` overwrites a gallery entry when its filename stem equals the query filename stem.
A query at `query/person.json` and a different candidate at `gallery/person.json` return zero candidates without an error.
A renamed copy of the query is accepted as another person. `rank_pool` excludes only the matching dictionary key.

Required correction: store a stable profile ID. Validate ID uniqueness and reject conflicting revisions. Use paths only to locate files.

### P2: Concurrent cache writes can fail

`jevector.py:27` uses the same temporary path for every writer targeting a cache entry.
Two calls with the same request hash can write and rename that path concurrently.
A controlled interleaving produced one successful write and one `FileNotFoundError`.
The provider client has no per-request lock. Equal condition lists can produce equal request hashes during concurrent extraction.

Required correction: use unique temporary files for atomic replacement. Deduplicate or lock concurrent requests for the same cache key.

### P2: Compatibility checks omit the matching policy

`matching/schema.py:46` fingerprints the questions. It does not fingerprint decoding thresholds, evidence rules, or the ranking policy.
`matching/engine.py:55` compares provider and model values but accepts vectors when both omit those fields.
The audit accepted two vectors with no provider or model provenance.
Vectors decoded by different policies can have the same schema hash and appear compatible.

Required correction: require provenance fields and version the decoder and matcher. Keep raw answers separate from derived representations.

## Evaluation limitations

### The oracle shares implementation with the matcher

`matching/fixtures.py:94` uses `reciprocal` to construct the expected ranking.
`matching/run.py:17` uses the same function to decide whether returned results violate ground-truth conditions.
A logic error in this shared function can affect both the result and its evaluation.
Individual unit tests cover some cases but do not make this evaluator independent.

Required correction: implement a small independent rule evaluator and hand-label boundary cases. Add symmetry, uncertainty, and monotonicity properties.

### The larger schema has no necessary role in this fixture

A source-attribute calculation restricted to the 16 core attributes identifies every intended partner as the unique best candidate: 32/32.
Every mandatory condition is a core condition. The extra attributes are unnecessary for the intended ranking.
The fixture therefore tests additional extraction load, but does not demonstrate a retrieval benefit from 256 values.

Required correction: include valid alternatives that differ only on extra attributes. Compare attribute subsets on a fixed independent test set.

### The apparent sample size overstates diversity

The fixture has 16 planted pairs. Its 128 distractors and lower-ranked alternatives are modifications of those paired profiles.
Each paired query has exactly two source-valid candidates. The four no-match profiles all require two intentions that the generator prevents candidates from combining.
The 32 directional queries share 16 pairs; they are not 32 independent compatibility examples.
A template parser already solves the fixture. Prompt and decoder changes have been examined on these same examples.

Required correction: separate pair families across development and test sets. Use independent wording, several no-match causes, and multiple plausible candidates.

### Top-1 safety does not cover returned candidates

The existing table counts violations only in the first result. Earlier diagnosis found one incompatible lower-ranked candidate accepted by Clef.
A zero top-1 violation count is not a zero filter-error count.

Required correction: report false accepts and false rejects across candidate comparisons, and violations in the returned top-k.

## Representation and retrieval limits

The condition representation supports conjunctions of positive and negative attributes.
It cannot directly express alternatives, numeric ranges, distances, or conditional requirements.
For example, `hiking OR cooking` cannot be represented by requiring both separate attributes.
Custom compound attributes could encode selected cases, but consume dimensions and require explicit schema changes.

The matching path thresholds numeric values into yes, no, and unknown, then performs an exact reciprocal scan.
Its current evidence supports an attribute-extraction and rule-matching system. It does not establish an advantage from continuous vectors or HNSW retrieval.
Random-vector HNSW tests do not validate candidate recall under these reciprocal rules.
A structured form plus a rule engine is an essential comparison when users can supply the attributes directly.

## Reproduce

```sh
search_demo/python -m matching.weakness_audit
```

Saved evidence: `runs/matching/weakness-audit.json`.
The command runs current regressions. The saved historical reproduction used live Clef responses.
The cache-race probe forces a valid concurrent interleaving; it does not estimate failure frequency.
These diagnostics replay the historical decoder. Current regression checks are in `matching/test_production.py`.
