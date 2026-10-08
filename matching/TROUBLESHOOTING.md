# Clef decoder diagnosis

These are post-hoc tests on the existing synthetic development fixture. The original benchmark is unchanged.

| Model | Decoder | Partner top-1 | False accepts / 5,804 incompatible comparisons | False rejects / 64 eligible comparisons |
|---|---|---:|---:|---:|
| Clef 64 | original | 22/32 | 1 | 18 |
| Clef 64 | threshold_0.60 | 32/32 | 1 | 2 |
| Clef 64 | threshold_0.55 | 32/32 | 1 | 2 |
| Clef 64 | condition_choice | 30/32 | 1 | 5 |
| Clef 64 | selected_choice | 32/32 | 1 | 1 |
| Clef 256 | original | 16/32 | 1 | 33 |
| Clef 256 | threshold_0.60 | 30/32 | 1 | 7 |
| Clef 256 | threshold_0.55 | 32/32 | 1 | 4 |
| Clef 256 | condition_choice | 30/32 | 1 | 5 |
| Clef 256 | selected_choice | 32/32 | 1 | 2 |
| Jev 64 | original | 32/32 | 0 | 0 |
| Jev 64 | threshold_0.60 | 32/32 | 0 | 0 |
| Jev 64 | threshold_0.55 | 32/32 | 0 | 0 |
| Jev 64 | condition_choice | 32/32 | 0 | 0 |
| Jev 64 | selected_choice | 32/32 | 0 | 0 |
| Jev 256 | original | 32/32 | 0 | 0 |
| Jev 256 | threshold_0.60 | 32/32 | 0 | 0 |
| Jev 256 | threshold_0.55 | 32/32 | 0 | 0 |
| Jev 256 | condition_choice | 32/32 | 0 | 0 |
| Jev 256 | selected_choice | 32/32 | 0 | 0 |

## Cause

The original decoder requires a numeric value of at least 0.7 for yes or at most 0.3 for no.
Other values become unknown. Any unknown required or excluded condition rejects the pair.
This also rejects pairs when a low-confidence no concerns an unstated condition.
The larger schema has more opportunities for this rejection.

## Remaining extraction error

Clef accepts one incompatible lower-ranked candidate under both original and selected-answer decoding.
That profile omits its alcohol-free preference. Clef infers yes instead of unknown.
The intended partner remains first, so top-1 violation counts do not expose this error.
The all-candidate table includes it. Counts refer to directed query/candidate comparisons.

## Interpretation

`condition_choice` uses selected answers for partner conditions and retains the original self-evidence checks.
`selected_choice` uses selected answers for all channels. Explicit mixed and unknown answers remain unknown.
The threshold variants retain the original evidence mask and change only the yes/no cutoff.
No API requests, prompt changes, or ground-truth filtering are used. Labels score the results only.

Equal cutoffs do not establish equal error rates across providers. Model probabilities have not been calibrated here.
Clef still has more selected-answer extraction errors. Improved match scores do not remove those errors.
Do not deploy a lower threshold based on these pairs. Validate it on independently generated profiles and missing-evidence cases first.

Run `search_demo/python -m matching.troubleshoot` to reproduce these tables.
Detailed answers and channel error counts are in `runs/matching/troubleshooting.json`.
