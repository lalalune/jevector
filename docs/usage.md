# Decision-vector profile retrieval

The active pipeline has four steps:

1. Validate condition coverage and the stable profile ID.
2. Ask Clef or Jev to encode the self and three partner-condition channels.
3. Use HNSW inner-product search to retrieve candidate IDs.
4. Check mandatory conditions in both directions and rank eligible candidates by preference coverage.

`ProfileIndex` projects self and requested-partner values into complementary query and document positions.
The projection includes optional, required, and excluded model values. It is a retrieval score, not the final compatibility score.
Exact rule checks use explicit requested conditions and decoded candidate evidence.

## Input

```json
{
  "profile_id": "person-001",
  "about_me": "I enjoy hiking and cooking. I want platonic friendship.",
  "looking_for": {
    "prefer": ["activity.hiking"],
    "require": ["intent.friendship"],
    "exclude": []
  },
  "facts": {"budget": 40, "location": [37.77, -122.42]},
  "rules": [
    {"any": [
      {"attribute": "activity.hiking", "value": true},
      {"attribute": "activity.cooking", "value": true}
    ]},
    {"field": "budget", "min": 10, "max": 80},
    {"distance_km": 25}
  ]
}
```

Conditions accept catalog IDs or exact descriptions from `matching/schema.py`.
An unsupported mandatory condition is an error. It is never treated as an absent condition.
Unsupported optional preferences are listed in `unresolved_preferences`. They receive no coverage credit and retain their denominator weight.
The 64-value schema contains 16 attributes; the 256-value schema contains 64.

Facts are explicit finite numbers. `location` uses latitude and longitude in degrees.
Distance is great-circle distance in kilometres.
Rules are mandatory and apply to the candidate. Both people can have rules.
`if` and `then` refer to candidate attributes/facts. An unknown condition rejects the rule.
Use `all` for conjunction and `any` for alternatives. Unknown leaves do not become false facts.

## Scores and evidence

Self answers retain unknown probability. A selected yes or no needs at least 0.7 raw probability to become known.
No probability is increased by removing unknown mass.
HNSW uses the model-produced values; strict checks use the explicit condition lists and known candidate attributes.

Optional coverage counts satisfied preferences divided by all specified preferences, including unresolved ones.
A side with no optional preferences contributes no coverage score. It does not contribute a perfect score.
The reciprocal score averages the sides that specify preferences. Hard-rule-only matches have score zero.
Profiles need self evidence and at least one matching condition across the pair.
Evidence coverage is returned separately from the score.
Scores are coverage fractions, not probabilities of relationship success.

## Identity and compatibility

File names do not identify people. `profile_id` does.
Repeated identical records with the same ID are deduplicated. Conflicting records for an ID are rejected.
Provider, model, schema, decoding policy, and matching policy must agree across the gallery.
Records include a content hash. Changes require regeneration rather than silent reuse.
These hashes detect inconsistency; they are not signatures proving that input facts are true.

## Search

```sh
scripts/python -m matching.cli rank \
  --query runs/matching-v2/clef-256/family0-persona.json \
  --gallery runs/matching-v2/clef-256/*.json \
  --method hnsw --candidates 32 --ef 64 --k 5
```

Use `--method exact` for exhaustive matching.
HNSW results are approximate unless every other profile was retrieved or an exact fallback ran.
`--exact-fallback` runs an exact scan only when fewer than `--k` eligible results were found.
All retrieved candidates receive the same reciprocal checks. This protects eligibility, not recall.
An extraction error can still make those checks wrong; the benchmark reports source-label violations.

## Validation

`qualification_data.py` uses a separate reference evaluator in `reference.py`.
The reference does not import the matcher or production rule evaluator.
Four pair families are development cases; four are validation cases. Thresholds are not selected using validation results.
Alternatives differ on a non-core attribute. The no-match cases cover unavailable attributes, missing facts, numeric bounds, and distance.
The test remains synthetic and small. Family separation is not independent human validation.

`qualification.py` records exact results, HNSW candidate recall, top-five list agreement, ties, all-result false accepts, and eligible-candidate recall.
`index_scale.py` measures the real profile matcher on 1,024 structured synthetic profiles.
No candidate-level truth is used to filter model outputs.
The declared-attribute control intentionally supplies source attributes and is labeled separately.

## Run

```sh
scripts/python -m matching.qualification_data
scripts/python -m matching.qualification --account "$CLOUDFLARE_ACCOUNT_ID" --prompt-key
scripts/python -m matching.index_scale
scripts/python -m matching.render
```

Use `--providers clef` or `--providers jev` to select a provider.
Use `--workers 2` to reduce concurrent requests. Use `--baselines-only` to rerun only the reference comparisons.
Artifacts are under `runs/matching-v2/`.

Run the tests with `scripts/python -m unittest discover -s tests`.
