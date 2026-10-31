# Final 500-question evaluation

DeBERTa is NOT recommended for activation under the requested all-targets rule. None of the three models fully satisfies all four requested requirements.

During the benchmark quality40k remained active; no model settings were changed or training performed. Subsequently the user explicitly authorized DeBERTa weighted_EM as the stable baseline despite the unmet targets. See STABLE_DEBERTA.md.

## Dataset and overlap audit

Exactly **500 English SQuAD 2.0 validation examples: 250 answerable and 250 unanswerable**. Seed `50020261030`; fingerprint `429b743d37b5caf6`. Selection was frozen in commit `bb9cacd`, before predictions. Ground truth is the original answer text and answer_start annotations, never generated labels.

The audit scanned 208 available JSON records and found 4415 previously referenced validation IDs. The eligible pool contained 7452 examples with unused IDs and normalized questions. The selected sample has **zero prior ID overlap, zero normalized-question overlap, but 500/500 previously used contexts**. Each selected example has a different context. No unused validation context remained. Context novelty was preferred before seeded sampling; none was available.

This is not a completely unseen or official instructor test. Vendor model selection and previous local work used SQuAD validation. The old abandoned full-data run has unknown consumed IDs. JSON records cannot establish universal historical non-exposure. Using the official training holdout instead would contaminate comparison with the SQuAD-pretrained models.

## Frozen configurations

The prior benchmark primary profiles are retained: weighted_EM for both Deepset models and active for quality40k. The alternative null95 calibration operating points were not substituted after seeing this sample. No threshold is fitted to these 500 examples. All models use 384-token windows, stride 128 and two CPU threads, in fresh sequential processes.

| Model | Threshold | Span policy |
|---|---:|---|
| deberta-v3-base-squad2 | 2.777294039726 | {"max_length": 30, "length_penalty": 0.0} |
| roberta-base-squad2 | 0.041785240173 | {"max_length": 30, "length_penalty": 0.0} |
| quality40k | -0.404942514417 | {"name": "length05", "max_length": 30, "length_penalty": 0.5} |

Weights and source-policy SHA-256 values are in `configs/final500_manifest.json`; local raw records are under `logs/final500/`. All three use precisely the same example IDs. Production settings and dashboard were preserved.

## Metric definitions and comparison

EM uses official SQuAD normalization (case, punctuation, articles, whitespace). F1 measures normalized token overlap, taking the maximum over original references. Null references are scored against an empty answer. Group scores use their own 250-example denominators; overall metrics weight both groups equally. Weighted EM is 0.75 × answerable EM + 0.25 × unanswerable EM. Abstention rate is the fraction of null predictions, not null-class accuracy. Scores are threshold-distance signals, not calibrated correctness probabilities.

| Model | Overall EM | Overall F1 | Answerable EM | Answerable F1 | Unanswerable EM | Weighted EM 75/25 | Abstention |
|---|---:|---:|---:|---:|---:|---:|---:|
| deberta-v3-base-squad2 | 83.00% | 86.69% | 77.60% | 84.97% | 88.40% | 80.30% | 47.20% |
| roberta-base-squad2 | 78.60% | 81.68% | 74.80% | 80.96% | 82.40% | 76.70% | 46.40% |
| quality40k | 58.40% | 61.06% | 46.40% | 51.73% | 70.40% | 52.40% | 53.40% |

## Independent target checks

Every required target must pass; a high average does not compensate for a failed class target. These are the thresholds requested by the user for this decision.

| Model | EM ≥85 | F1 ≥85 | Answerable EM ≥80 | Unanswerable EM ≥95 | All four |
|---|---|---|---|---|---|
| deberta-v3-base-squad2 | FAIL | PASS | FAIL | FAIL | FAIL |
| roberta-base-squad2 | FAIL | FAIL | FAIL | FAIL | FAIL |
| quality40k | FAIL | FAIL | FAIL | FAIL | FAIL |

## Confidence intervals

95% percentile bootstrap intervals use 3,000 resamples, seed 50020261030, stratified to retain 250 examples per class. Each question has a distinct context. The same resampled indices are used for paired model comparisons. These describe sampling uncertainty conditional on this historically exposed validation population, not protection against domain shift or prior selection bias.

| Model | EM 95% CI | F1 95% CI | Answerable EM 95% CI | Unanswerable EM 95% CI | Weighted EM 95% CI |
|---|---|---|---|---|---|
| deberta-v3-base-squad2 | [79.80, 86.00] | [83.86, 89.31] | [72.40, 82.80] | [84.40, 92.00] | [76.20, 84.10] |
| roberta-base-squad2 | [75.00, 82.20] | [78.32, 84.76] | [69.60, 80.00] | [77.60, 86.80] | [72.50, 80.90] |
| quality40k | [54.20, 62.61] | [56.94, 65.17] | [40.40, 52.40] | [64.80, 76.00] | [47.50, 57.30] |

Paired weighted-EM gain versus quality40k, deberta-v3-base-squad2: 95% CI [22.50, 33.30] percentage points.

Paired weighted-EM gain versus quality40k, roberta-base-squad2: 95% CI [19.30, 29.50] percentage points.

## Comparison with the previous 1,024-example benchmark

These cohorts differ. Changes below are descriptive, not paired improvement estimates or evidence of retraining. The same primary thresholds and policies were used.

| Model | Previous EM / F1 | Current EM / F1 | EM / F1 difference (points) |
|---|---|---|---|
| deberta-v3-base-squad2 | 84.38 / 87.72 | 83.00 / 86.69 | -1.38 / -1.04 |
| roberta-base-squad2 | 81.05 / 83.99 | 78.60 / 81.68 | -2.45 / -2.31 |
| quality40k | 60.74 / 62.54 | 58.40 / 61.06 | -2.34 / -1.47 |

## CPU latency and RAM

Every one of the 500 predictions was timed after an untimed warmup, including tokenization, span extraction and the decision. Model loading is excluded. RAM is peak process working set, including Python and dataset overhead. This is one measured run per model, not a speed confidence interval.

| Model | Mean / p95 / max seconds | Peak RAM MiB |
|---|---|---:|
| deberta-v3-base-squad2 | 0.800 / 0.899 / 2.340 | 1268.3 |
| roberta-base-squad2 | 0.504 / 0.585 / 1.436 | 1109.4 |
| quality40k | 0.257 / 0.292 / 0.766 | 931.3 |

## API reliability and concurrency

The previous quality40k bounded-concurrency run recorded one HTTP failure. A direct two-thread tokenizer reproduction produced RuntimeError: Already borrowed. The inference lock was extended to cover tokenization as well as model forward execution, for both production and benchmark engines. This is a thread-safety correction; weights, mathematical span selection and thresholds were not changed. Regression tests cover simultaneous calls.

Tests use the real shared FastAPI service on loopback port 8000 and a random in-memory token. Health, authenticated info with model hash, exact authentication rejection, schema errors, null responses, Unicode and long-context offsets are checked. Each performance mode sends 50 requests; bounded concurrency means two simultaneous clients, not fifty simultaneous clients. All prediction JSONs must have exactly id, answer, start and score. Public samples are repeated for HTTP load only and never tune the models.

| Model | Contract | Sequential valid / total; seconds; max | Concurrent valid / total; seconds; max | Limits pass |
|---|---|---|---|---|
| deberta-v3-base-squad2 | PASS | 50/50; 73.09; 2.764 | 50/50; 55.82; 3.281 | PASS |
| roberta-base-squad2 | PASS | 50/50; 53.01; 2.020 | 50/50; 36.19; 2.131 | PASS |
| quality40k | PASS | 50/50; 40.02; 1.396 | 50/50; 25.05; 1.467 | PASS |

Five seconds per request and 150 seconds per mode are checked independently of accuracy. HTTP response details, including any failure status/body, are retained. Temporary servers were stopped. No public tunnel was opened.

## Per-question predictions and raw artifacts

All 500 individual rows, original expected answers, predicted text/null, offsets, scores, EM/F1 and latency are available locally in `logs/final500/per_question.csv`. Each model also has `predictions.json`, `http.json` and `summary.json`. These files are deliberately excluded from Git along with the dataset and weights; they are not missing generated results. Run the committed scripts with the local artifacts to regenerate the report. The compact public summary is `logs/final500_summary.json`.

CSV SHA-256: `6cc33c208e724effbe0d759187b2fafad45218f8be100b0e09dc16ffbb4c3211`.

The first five rows in the frozen order (not selected for correctness) are shown below. The complete local CSV contains every question and expected answer.

| Question ID | Original references | DeBERTa answer | RoBERTa answer | quality40k answer |
|---|---|---|---|---|
| 5acf98ad77cf76001a6853c6 | [] | null | "academic institutions and programs" | null |
| 5733ffa7d058e614000b6751 | ["attacked the British column, killing and capturing several hundred men, women, children, and slaves.", "attacked the British column", "attacked the British column", "attacked the British", "attacked the British column"] | " attacked the British column" | "attacked the British column" | "under parole" |
| 5a3e2dc0378766001a0024c5 | [] | null | null | null |
| 5ad13b23645df0001a2d12fa | [] | null | null | null |
| 5725c0f289a1e219009abdf6 | ["‘combs’ – groups of cilia", "combs", "the ‘combs’"] | " the ‘combs’" | "the ‘combs" | "combs" |

## Recommendation and remaining limitations

DeBERTa is NOT recommended for activation under the requested all-targets rule. None of the three models fully satisfies all four requested requirements.

The user subsequently approved DeBERTa as a stable baseline; quality40k remains operational as a rollback option. A statistical gain over quality40k does not imply meeting all four absolute targets. No thresholds will be changed after this evaluation. Known domain mismatch, historical context exposure and imperfect no-answer recognition remain relevant. No submission tag was created.
