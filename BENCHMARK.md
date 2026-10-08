# Profile test results: version 2.0

Historical same-person retrieval test. This does not test matching different people. See the [current paired benchmark](README.md).

Test set: 32 synthetic people with two descriptions each. Each configuration also has four controls.

## Results

| Configuration | Correct first matches | Preference accuracy | Median request time | Estimated cost |
|---|---:|---:|---:|---:|
| clef-64 | 29/32 | 98.72% | 0.928 s | $0.0685 |
| clef-256 | 31/32 | 99.28% | 3.668 s | $0.2655 |
| jev-64 | 27/32 | 99.49% | 0.362 s | $0.0275 |
| jev-256 | 28/32 | 99.87% | 0.516 s | $0.1004 |

## Definitions

Top-1 is the proportion of queries with the correct profile first.
Preference accuracy compares model answers with the stated preference labels.
Request time includes network time. Detailed Clef profiles require four sequential requests.
Cost uses published input-token rates. It is not a billing statement.

## Controls

- clef-64: 8/9 labeled control answers are correct.
- clef-64, negation, relationship.not_dating: expected yes; received no.
- clef-256: 11/12 labeled control answers are correct.
- clef-256, negation, relationship.not_dating: expected yes; received no.
- jev-64: 9/9 labeled control answers are correct.
- jev-256: 12/12 labeled control answers are correct.

## Repeat tests

| Configuration | Test | Shared-value RMS difference | Mask changes | Choice changes |
|---|---|---:|---:|---:|
| clef-64 | repeat | 0.0000 | 0 | 0 |
| clef-64 | question_order | 0.0981 | 4 | 9 |
| clef-64 | option_order | 0.0000 | 0 | 0 |
| clef-256 | repeat | 0.0000 | 0 | 0 |
| clef-256 | question_order | 0.0885 | 9 | 25 |
| clef-256 | option_order | 0.0000 | 0 | 0 |
| jev-64 | repeat | 0.0137 | 0 | 0 |
| jev-64 | question_order | 0.0106 | 0 | 0 |
| jev-64 | option_order | 0.0237 | 0 | 2 |
| jev-256 | repeat | 0.0112 | 0 | 4 |
| jev-256 | question_order | 0.0126 | 0 | 6 |
| jev-256 | option_order | 0.0267 | 2 | 10 |

## Limits

The test data is synthetic. The descriptions are related text variants.
The results do not establish personality accuracy or relationship compatibility.
Question order and request grouping can change the output.
The old matcher can favor profiles with little shared evidence.
Current results are in IMPROVEMENTS.md.

## Data

Saved results: runs/<configuration>/metrics.json.
Saved questions: data/archive/schema-64-v2.0.json and data/archive/schema-256-v2.0.json.
Input-token rates: Clef-flash $0.09 per million; Jev $0.042 per million.
