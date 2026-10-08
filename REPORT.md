# Initial personality test

Historical same-person retrieval test. This does not test matching different people. See the [current paired benchmark](README.md).

Model: Cloudflare Clef-flash. Input: 12 synthetic people with two descriptions each. Vector size: 64.

## Results

Correct first matches: 12/12.
Top-three accuracy: 100%.
TF-IDF first-match accuracy: 0%.

| Query | First match | Correct rank | Correct distance | Closest other distance |
|---|---|---:|---:|---:|
| p01 | p01 | 1 | 0.1024 | 0.1399 |
| p02 | p02 | 1 | 0.1068 | 0.2061 |
| p03 | p03 | 1 | 0.1142 | 0.2114 |
| p04 | p04 | 1 | 0.0864 | 0.2098 |
| p05 | p05 | 1 | 0.0870 | 0.1824 |
| p06 | p06 | 1 | 0.0976 | 0.1489 |
| p07 | p07 | 1 | 0.1010 | 0.1799 |
| p08 | p08 | 1 | 0.0977 | 0.1321 |
| p09 | p09 | 1 | 0.0591 | 0.2870 |
| p10 | p10 | 1 | 0.0911 | 0.1360 |
| p11 | p11 | 1 | 0.0881 | 0.2197 |
| p12 | p12 | 1 | 0.0663 | 0.1798 |

## Method

Each dimension uses an expected score from 0 to 4. The program divides the score by four.
The test uses equal-weight root-mean-square distance.
The midpoint combines mixed evidence and missing evidence.
Later schemas separate unknown from mixed evidence.

## Limits

The descriptions are synthetic text variants. The test measures retrieval consistency.
It does not establish personality accuracy or interpersonal compatibility.
Raw responses and vectors are in results/.
