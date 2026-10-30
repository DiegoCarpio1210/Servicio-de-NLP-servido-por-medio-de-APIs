# Completed quality40k comparison — 2026-10-30

**Recommendation: select quality40k with its locked length05 policy, subject to
user approval. Production still uses continued20k checkpoint-1600.** No further
training or public exposure was performed during this recovery/evaluation.

## Training and artifact integrity

40,000 examples, including 13,406 impossible (33.515%); 512 selection examples;
one complete epoch; 2,531 optimizer steps; 2,590.554 seconds (43m11s), CUDA Quadro
P1000, peak reserved 2,798 MiB. Minimum selection loss 1.441129 at step 2000
selected the exported checkpoint. Selected epoch 0.7902 differs from the completed
run's 1.0 epoch. This selection criterion is not final-set accuracy.

Exported weights SHA-256:
`d42cc12e1ba17bc04e1ff652bd7e3bc7292d6d9ce39032501820c470b844d60f`.
Model, tokenizer and config match checkpoint-2000. Retained checkpoints 2000,
2400 and 2531 have hashed optimizer/scheduler/RNG files; resuming was not executed.
Stable weights and calibration hashes remain unchanged. See `artifact_audit.json`.

## Methodology

The protocol was committed in `9a546ec`; model/policy/threshold decisions were
locked in `37fc668` before final predictions. Both weight sets were evaluated on
the same 1,024 final examples (512 answerable, 512 impossible). Thresholds use the
original 512 calibration examples; policies use a separate 1,024 development set.
No final/public/ISS example selected settings. No retuning follows this evaluation.

All official-validation contexts had already been consulted. These final examples
come from prospectively reserved official-TRAIN contexts excluded from retained
model training, development, calibration and selection. Training references have
fewer answer variants. An abandoned full-data run has unknown consumed IDs and no
retained descendant in this comparison: absolute historical non-exposure cannot
be certified. Selection/calibration IDs are disjoint but share 132 contexts.
These are local experimental results, not a new official or instructor test.

## Same-example results (percent)

| Profile | EM | F1 | Answerable EM | Answerable F1 | Impossible EM | 75/25 EM |
|---|---:|---:|---:|---:|---:|---:|
| continued20k stable | 57.71 | 60.81 | 36.33 | 42.52 | 79.10 | 47.02 |
| continued20k short15 | 57.81 | 60.83 | 36.33 | 42.36 | 79.30 | 47.07 |
| quality40k + length05 | 61.82 | 65.53 | 48.05 | 55.47 | 75.59 | 54.93 |

Candidate gains: EM +4.10 points; F1 +4.71; answerable EM +11.72; answerable F1
+12.94; impossible EM -3.52; weighted EM +7.91 (+16.82% relative to stable).
The paired context-cluster bootstrap (3,000 resamples, seed 42) gives weighted-EM
gain 95% CI [+5.48, +10.43] points. This supports an improvement on this sample,
not a guaranteed improvement on the instructor's different texts. The inference-
only gain is +0.049 points, CI [0, 0.160], not convincing evidence for replacement.

The measured gain belongs to the full trained-and-calibrated candidate; it does
not isolate the causal effect of training from its selected postprocessor.
Targets EM>65, F1>70 and answerable EM>50 remain unmet. Impossible EM stays above
70 on this sample. Historical 53.61/56.18 scores use different examples and must
not be subtracted from these candidate scores.

Abstention falls from 62.70% to 52.83%. Non-exact predictions fall from 42.29% to
38.18%. Wrong non-null answers increase from 19.14% to 23.14% of all examples;
among answered questions their fraction falls from 51.31% to 49.07%. “Wrong” here
means non-exact after normalization, including partially correct spans. Reduced
abstention increases coverage and the absolute count of unsupported/wrong spans.

## Descriptive question types

| Question prefix | Count | Stable EM | Candidate EM |
|---|---:|---:|---:|
| other | 205 | 53.66 | 62.93 |
| when | 51 | 60.78 | 66.67 |
| what | 496 | 60.08 | 61.69 |
| how many | 42 | 54.76 | 52.38 |
| why | 13 | 69.23 | 76.92 |
| who | 105 | 61.90 | 64.76 |
| which | 32 | 43.75 | 62.50 |
| how | 28 | 57.14 | 64.29 |
| where | 41 | 53.66 | 56.10 |
| how much | 11 | 27.27 | 27.27 |

Prefixes are a descriptive heuristic, not a semantic classification. Small groups
are uncertain; they did not influence selection. F1 and error/abstention rates
per group are in `final_details.json`.

## Real local CPU HTTP comparison

| Profile | 50 calls (s) | Mean (s) | Maximum (s) | Sampled RAM (MiB) | Mean CPU (% all logical cores) |
|---|---:|---:|---:|---:|---:|
| continued20k stable | 22.93 | 0.443 | 1.186 | 826.9 | 31.91 |
| continued20k short15 | 27.72 | 0.535 | 1.494 | 825.8 | 32.76 |
| quality40k + length05 | 28.54 | 0.551 | 1.419 | 826.0 | 32.45 |

Same adapter, one server at a time, four Torch threads, real port-8000 HTTP and
fresh in-memory test tokens. Fifty requests repeat the five public examples ten
times, not fifty independent hidden questions. All profiles pass <5s/request and
<150s total. No timeout or inference exception occurred. RAM is sampled process-
tree RSS, not a continuous system-wide peak. CPU is process-tree CPU time divided
by elapsed time and logical-core count. Single-run timings are descriptive, not
a statistically established speed difference. Each server was stopped afterward.

All profiles passed the 18 real HTTP checks: exact authentication/rejection,
invalid JSON/schema/length errors, Unicode offsets and a 5,000-character context.
The shared academic API contract is unchanged. No new HTTPS or mobile test is
claimed for quality40k. Previous Cloudflare/mobile/ngrok evidence belongs to stable.

## Public conformance (no tuning)

| Item | quality40k answer | Start | Correct |
|---|---|---:|---|
| sample-01 | 16 | 383 | Yes |
| sample-02 | 3.5 days | 437 | Yes |
| sample-03 | null | null | Yes |
| sample-04 | 10,000 | 263 | No |
| sample-05 | 50,257 | 3157 | Yes |

Stable and inference-only also score 4/5. Sample-04 remains a semantic attribution
failure: the numeric span describes GPT-3 rather than the asked-about BERT model.
This mandatory public-sample goal remains FAILED; no special exception was added.

## Evidence and operational status

Evidence resides in `logs/quality20261029/`: artifact_audit, partition_audit,
stable_selection, trained_selection, final_lock, final_comparison, final_details,
and cpu_*/result plus real_http_contract JSON files. Full local prediction files
are gitignored. Training metadata/history/loss curve are in `logs/quality40k/`.

60 automated tests passed on recovery. Production GUI design, default model and
dashboard are unchanged. See `CANDIDATE_REVIEW_ES.md` for prepared configuration;
approval is still required before integration. The official dry run, active
instructor-token service, public-sample failure, weight distribution approval,
pitch/Q&A preparation and final submission tag remain open. REPORT.md is retained;
no final PDF or submission tag is generated.

The stable launcher was started again without opening a browser; four real UI
gateway cases passed (Paris, null, Unicode, long-context offset 3600), as did
oversized-input, host and cross-site rejection. The Windows stop BAT closed its
services; ports 8000/8765 were unavailable afterward. `ui_recovery.json` and
`final_audit.json` record this check. Prior browser visual checks are reused;
no new visual-session verification is claimed. REPORT.md has the exact ten
template headings and 756 prose words excluding tables. Original PDF/YAML/sample/
template hashes remain unchanged.
