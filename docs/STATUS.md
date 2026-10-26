# Current status — Pimientero

One continuation completed; no additional run or submission tag. Candidate: **continued**, `models/continued20k`.
Selection was fixed on calibration before either final holdout was evaluated. All metrics below use the fresh 1,024.

| Profile | Overall EM | Overall F1 | Answerable EM | Answerable F1 | Impossible EM | 75/25 EM | 75/25 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 52.25 | 53.50 | 22.35 | 24.87 | 81.91 | 37.24 | 39.13 |
| recalibrated | 37.70 | 41.94 | 47.65 | 56.17 | 27.82 | 42.69 | 49.08 |
| continued | 53.61 | 56.18 | 33.92 | 39.08 | 73.15 | 43.73 | 47.60 |

Public accuracy: 4/5. Fifty local CPU requests: 14.08s; maximum 0.632s.
Training: original 26m22s plus continuation 42.98 minutes. Original weights are preserved.
See CONTINUATION_EXPERIMENT.md, logs/continued20k/comparison.json and COMPLIANCE.md.

Pending: instructor token entered locally, current-model HTTPS and another-network test, backup tunnel,
official dry run and active URL, authorized separate weight distribution, pitch/Q&A rehearsal and final review.
The API test process was stopped after verification. Start it with the README commands and the candidate model path.
No further training is authorized automatically. Submission is not declared complete.
