# Profile test results: version 2.1

Historical same-person retrieval test. This does not test matching different people. See the [current paired benchmark](README.md).

Schema: jevector-2.1. Matcher: coverage-v1.

## Original set

| Configuration | Previous matches | New matcher with old answers | New live matches |
|---|---:|---:|---:|
| clef-64 | 29/32 | 32/32 | 32/32 |
| clef-256 | 31/32 | 32/32 | 32/32 |
| jev-64 | 27/32 | 31/32 | 32/32 |
| jev-256 | 28/32 | 32/32 | 32/32 |

## Additional set

The additional set has 16 synthetic people. Alternate descriptions omit three interest labels.

| Configuration | Old matcher | New matcher | Preference accuracy | Correct controls |
|---|---:|---:|---:|---:|
| clef-64 | 13/16 | 16/16 | 94.10% | 17/17 |
| clef-256 | 10/16 | 16/16 | 97.07% | 20/20 |
| jev-64 | 16/16 | 16/16 | 94.99% | 16/17 |
| jev-256 | 16/16 | 16/16 | 98.15% | 19/20 |

## Preference extraction

| Configuration | Previous accuracy | Current accuracy |
|---|---:|---:|
| clef-64 | 98.72% | 97.19% |
| clef-256 | 99.28% | 98.70% |
| jev-64 | 99.49% | 97.96% |
| jev-256 | 99.87% | 99.61% |

## Calculation

The matcher compares shared dimensions. It gives equal weight to each group with evidence on either side.
Each group receives the additional cost `0.25 × (1 − shared_evidence / union_evidence)`.
The result is the square root of the mean group cost.
The matcher requires at least eight shared dimensions.
Missing evidence is not a negative preference.

## Remaining errors

Jev returns no for the contradictory dating-intent control. The expected answer is mixed.
Preference extraction accuracy is lower on some tests.
The original set informed the change. It is development evidence.
The additional set is synthetic. It is not independent human validation.

## Verification

The tests checked 432 complete profile outputs.
The tests checked missing evidence, equal scores, group weights, response validation, and cache reuse.
The data does not establish production accuracy.

## Commands

```sh
python3 -m unittest -v test_jevector
python3 run_improvements.py --provider both --account "$CLOUDFLARE_ACCOUNT_ID" --prompt-key
python3 improvement_report.py
```

## Files

Original-set outputs: runs/v21/.
Additional-set outputs: runs/fresh-v21/.
Comparison data: runs/improvements.json.
