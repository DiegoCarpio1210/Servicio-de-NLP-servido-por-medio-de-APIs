# DeBERTa LoRA answerability and calibration analysis

This is an experimental calibration study, not a production activation or an independent final test. No training occurred. Main, original DeBERTa, quality40k, historical predictions, all LoRA checkpoints, API and dashboard are preserved.

## Completed pilot verified

The saved pilot completed 200/200 optimizer updates: 461.20 seconds optimizer-loop time, 523.11 seconds training-phase time. Checkpoints 0000, 0025, 0028, 0050, 0075, 0100, 0125, 0150, 0175 and 0200 exist. Recovery from update 28 restored optimizer, scheduler, cursor and RNG. The predeclared development weighted-EM selector retained step 0. See [pilot report](DEBERTA_LORA_PILOT200.md) for measured memory, losses and provenance.

## Data roles and limitations

Training used 1,024 examples. Previously observed development contains 128 examples (64 answerable/64 impossible); reserved calibration contains a different 128 (64/64). All 1,280 selected normalized contexts/questions and IDs are distinct. The training implementation consumes train only; checkpoint scoring consumes development only, confirmed by prediction IDs. Calibration was not used for the pilot or checkpoint selection. In this new study it is used only for threshold selection.

Answerability is confounded with source: AdversarialQA supplies all positive examples and SQuAD2 all negatives. SQuAD2 was used by the vendor; AdversarialQA shares Wikipedia/SQuAD contexts. Historical benchmark IDs/context exclusions are retained, but universal vendor exposure or near-duplicate independence cannot be certified. The old 500/1,024 benchmarks and public samples were not used for this tuning or rerun. No credible independent final set is available locally; none is invented here.

## Actual error breakdown at the frozen threshold

| Step | Answerable null | Exact non-null | Partial non-null | Zero-overlap non-null | All inexact non-null | Correct impossible null | Exact candidate rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24/64 | 14/64 | 15/64 | 11/64 | 26/64 | 64/64 | 7 |
| 100 | 22/64 | 14/64 | 17/64 | 11/64 | 28/64 | 64/64 | 7 |
| 200 | 22/64 | 14/64 | 17/64 | 11/64 | 28/64 | 64/64 | 7 |

The null/exact/partial/zero-overlap answerable columns are mutually exclusive; the inexact column sums partial and zero-overlap. Partial means positive token F1 below 1. The threshold rejects useful spans, but extraction quality is also limiting:

| Step | Candidate answerable EM before abstention | Candidate answerable F1 before abstention |
|---:|---:|---:|
| 0 | 32.81% | 52.13% |
| 100 | 32.81% | 52.13% |
| 200 | 32.81% | 52.13% |

Step 200 changed 1 candidate text/start pairs and 2 null decisions versus step 0. Mean margin shifts: answerable -1.2792, impossible -1.9787. These measured shifts must not be mistaken for improved span accuracy.

All 64 answerable candidate text/start spans were identical at steps 0 and 200; the one changed candidate belongs to an impossible question. Every stored candidate span and returned offset was checked against the supplied original development context. Current GPU inference reproduced three saved examples per state (one previously abstained answerable, one answered example and one impossible) within 0.0002 margin tolerance. All 50 saved adapter/QA-head tensors at steps 100 and 200 were checked against active loaded tensors. The loaded QA head differs from the original at both trained checkpoints; step 0 has the original head and zero LoRA B matrices. There is no silent fallback to the base model.

## Margin convention and threshold selection

The project computes margin = minimum window CLS(start+end) score minus the best valid span score. `qa.spans.decide` returns null when `start is None OR margin > threshold`; equality answers. Increasing the threshold reduces abstention. The API score is sigmoid(abs(margin-threshold)), a distance signal rather than a calibrated correctness probability.

Before reserved inference, commit `a9ed983` declared: maximize calibration 75/25 weighted EM subject to empirical unanswerable EM >=95%; ties maximize weighted F1, then unanswerable EM, then choose the lower threshold. Comparisons round metrics to 10 decimals to avoid numerical tie noise. The existing exact weighted_threshold_sweep tests minimum margin minus one and each distinct margin. Models, span policy and candidate lists stay fixed; thresholds are stored only under the ignored experimental directory.

Selected original threshold: **14.584873199463**. Selected step-200 threshold: **12.982471942902**. Production remains **2.777294039726**. With 64 negatives, the 95% floor permits at most three errors; it is an empirical constraint, not a population guarantee.

## Calibration results — 128 examples, used for threshold fitting

| System | Overall EM | Overall F1 | Answerable EM | Answerable F1 | Impossible EM | Weighted EM | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| original · original threshold | 64.06 | 68.03 | 29.69 | 37.62 | 98.44 | 46.88 | 52.83 |
| original · calibrated | 65.62 | 69.59 | 34.38 | 42.31 | 96.88 | 50.00 | 55.95 |
| step200 · original threshold | 64.06 | 68.03 | 29.69 | 37.62 | 98.44 | 46.88 | 52.83 |
| step200 · calibrated | 64.84 | 68.81 | 34.38 | 42.31 | 95.31 | 49.61 | 55.56 |

original calibration-set effect: EM +1.56 points; F1 +1.56 points. These are fitted-set changes.

step200 calibration-set effect: EM +0.78 points; F1 +0.78 points. These are fitted-set changes.

## Previously observed development diagnostics — 128 examples

These examples were previously inspected and used for checkpoint selection. They did not select the new thresholds and are not a fresh independent final test.

| System | Overall EM | Overall F1 | Answerable EM | Answerable F1 | Impossible EM | Weighted EM | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| original_policy | 60.94 | 67.64 | 21.88 | 35.29 | 100.00 | 41.41 | 51.46 |
| original_calibrated | 60.16 | 69.01 | 25.00 | 42.71 | 95.31 | 42.58 | 55.86 |
| step200_original_policy | 60.94 | 68.88 | 21.88 | 37.76 | 100.00 | 41.41 | 53.32 |
| step200_calibrated | 60.94 | 69.79 | 26.56 | 44.27 | 95.31 | 43.75 | 57.03 |

Original calibration effect on the same observed development examples: EM -0.78 points; F1 +1.37 points.

Step200 calibration effect on the same observed development examples: EM +0.00 points; F1 +0.91 points.

## Pareto tradeoff on calibration

| Model | Threshold | Answerable EM | Impossible EM | Meets 95% floor |
|---|---:|---:|---:|---|
| original | -0.942612 | 29.69 | 100.00 | yes |
| original | 10.951324 | 32.81 | 98.44 | yes |
| original | 14.584873 | 34.38 | 96.88 | yes |
| step200 | -2.207719 | 29.69 | 100.00 | yes |
| step200 | 9.008813 | 32.81 | 98.44 | yes |
| step200 | 12.982472 | 34.38 | 95.31 | yes |

Only nondominated answerable-EM / impossible-EM operating points are shown; tied pairs retain the declared F1/conservatism preference. Points below the floor are not selected. Full threshold sweeps and raw predictions remain local.

## Margin distributions (min / Q1 / median / Q3 / max)

| Cohort / model / class | Margin quantiles |
|---|---|
| development step 0 / answerable | -24.831 / -9.798 / -0.738 / 10.455 / 30.608 |
| development step 0 / unanswerable | 12.987 / 22.254 / 25.645 / 27.358 / 32.077 |
| development step 100 / answerable | -24.395 / -10.288 / -1.796 / 9.106 / 28.972 |
| development step 100 / unanswerable | 11.968 / 21.020 / 24.263 / 25.771 / 30.440 |
| development step 200 / answerable | -24.282 / -10.577 / -2.230 / 8.487 / 28.378 |
| development step 200 / unanswerable | 11.571 / 20.585 / 23.784 / 25.228 / 29.857 |
| calibration original / answerable | -23.832 / -15.332 / -7.698 / 15.124 / 29.875 |
| calibration original / unanswerable | 1.542 / 21.760 / 25.451 / 27.345 / 30.568 |
| calibration step200 / answerable | -23.488 / -15.666 / -8.371 / 13.455 / 27.591 |
| calibration step200 / unanswerable | 0.220 / 19.732 / 23.419 / 25.199 / 28.373 |

## Uncertainty and direct paired comparisons

Stratified paired bootstrap: 3,000 resamples, seed 20261031, retaining 64 examples per class and sharing sampled indices across systems. Intervals hold fitted thresholds fixed. Calibration intervals therefore omit threshold-selection uncertainty; development intervals also cannot erase earlier model selection or source confounding. These are descriptive, not independent generalization guarantees.

| Cohort | Calibrated LoRA minus calibrated original EM 95% CI | F1 95% CI | Weighted EM 95% CI |
|---|---|---|
| calibration | [-2.34, +0.00] | [-2.34, +0.00] | [-1.17, +0.00] |
| observed_development | [-1.56, +3.12] | [-1.56, +3.13] | [-0.78, +4.30] |

On observed development, calibrated LoRA minus calibrated original: EM +0.78 points; F1 +0.78 points; weighted EM +1.17 points.

## Conclusion and next step

Best experimental candidate by the declared calibration objective is the original model at threshold 14.58487319946289: calibration weighted EM 50.00%, versus 49.61% for calibrated LoRA. This preference uses calibration only, not the development comparison. It remains a provisional experimental policy, not a deployment recommendation.
The frozen threshold over-abstains on this hard mixed-source cohort, but no threshold can fix an incorrect candidate span. Use the measured tables, not falling training loss, to judge the adapter. The pilot did not improve candidate-span EM/F1 on development and did not win its preregistered checkpoint criterion. There is no evidence here warranting production LoRA activation or another immediate training run.

Keep the stable model and threshold. Treat the fitted original and LoRA policies as experimental candidates only. Obtain a new source-balanced, independently annotated QA set with both answerable and impossible questions within each source, audit context/question overlap, lock one candidate policy before opening it, and evaluate once. Include SQuAD positives in a future mixed-source training design rather than confounding source with answerability. Another LoRA run should wait for a defensible data design and predeclared external evaluation.

The complete regression suite, including deterministic selection, strict threshold equality, empirical null-floor constraints, error partitions and paired-bootstrap checks, passed 72 tests (logs/deberta_lora_calibration_tests.txt). Frozen calibration inference used a recorded peak of 790.01 MiB allocated / 800 MiB reserved on the P1000. No gradient updates were performed.

Full results and all fixed-threshold confidence intervals: logs/deberta_lora_calibration_summary.json. Raw predictions, threshold sweeps, loaded-tensor verification and experimental thresholds are local under logs/deberta_lora/calibration_analysis/. No historical evaluation was overwritten, no training was restarted, and no adapter or experimental threshold was activated.
