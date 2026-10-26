# One authorized continuation experiment

Status: the single authorized continuation completed. Measured results appear below.
The user authorized exactly one controlled fine-tuning run in the 2026-10-26 request.

## Diagnosis and alternatives

The prior selection loss decreased 2.1117 -> 2.0203 -> 1.8306 -> 1.7962 at steps 400, 800, 1200 and 1522.
Late training loss was about 1.52–1.77. This is evidence of continued learning, not proof that more epochs will
improve exact match. There is no observed upward validation-loss trend establishing overfitting; the train/validation
gap nevertheless warrants checkpoint selection. The linear schedule was nearly zero by step 1522.
The previous calibration audit found raw-span errors and excessive abstention, but no indexing/overflow defect.
See CALIBRATION_DIAGNOSIS.md for the completed audit; its extra-training proposals were superseded by this authorization.

Training uses the model's standard mean start/end cross-entropy. Impossible questions and answerable windows
without the gold answer target CLS. This is consistent with the extractive objective; it also means window-level
null frequency can exceed example-level impossible frequency. There is no demonstrated loss implementation error.
Separate start/end losses do not directly optimize exact span match or the final null decision. Gold offsets,
shared tokenization, dynamic-padding labels and tail coverage already have tests; they were not changed.

| Alternative | Assessment |
|---|---|
| A: continue with new data | Adds coverage absent from the first 12k; selected as part of the combined strategy |
| B: smaller learning rate | 1e-5 instead of 3e-5 limits destructive updates; restart optimizer/schedule because the old schedule ended |
| C: larger representative sample | Nested 20k preserves prior data and adds 8k; two epochs estimated 44–60 minutes |
| D: LoRA/PEFT | Not implemented: full fine-tuning already fits 4 GiB and meets time budget; adapters add integration and hyperparameter risk without measured superiority here |
| E: span/calibration changes | Preserve validated span search; use calibration with explicit impossible-question safeguard, not another unconstrained abstention reduction |

[Hugging Face PEFT's LoRA guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
describes parameter-efficient updates and adapter handling. This does not establish that LoRA improves this
DistilBERT QA task. No LoRA benchmark or speculative accuracy gain is reported.

## Fixed run configuration

- Start from the project's own trained encoder and QA head in models/subset12k; verify its SHA-256.
- Use a separate models/continued20k export and checkpoints/continued20k; preserve original artifacts.
- Train 20,000 examples: old 12,000 plus 8,000 previously unused official training examples.
- Impossible examples: 6,675 total / 2,670 added; reproducible stratified sample, training seed 42.
- Two additional epochs; 1e-5 learning rate, 5% warmup, linear decay, weight decay .01.
- Quadro P1000, FP32, batch 16, accumulation 1, dynamic padding and length grouping.
- Save resumable checkpoints every 100 steps, select minimum loss on original 512 selection examples;
  evaluate every 400 steps and at the end. Preserve the best checkpoint and recent checkpoints.
- Stop safely at 3,300 seconds if the run is slower than projected; write checkpoint and incomplete status.
- Exactly one run, exclusive process lock; detached launcher prevents loss of chat availability stopping training.

The original measured duration was 1,582.52 seconds for two 12k epochs: proportional estimate is 43.96 minutes
for two 20k epochs, with a 44–60 minute planning range. Data preparation and later CPU evaluation add time.
Original checkpoints are untouched. No second attempt or automatic full-dataset extension is authorized.

## Evaluation protocol fixed before the run

Keep existing selection (512), calibration (512) and old holdout (1,024) IDs separate. The old holdout has been
reported before and is not pristine. Reserve an additional 1,024 validation examples, excluding every recorded
historical validation ID and all contexts associated with those IDs. Reserve without inspecting predictions.

For the continued weights, fit a threshold on calibration maximizing weighted EM (75/25), with weighted F1
as tie-break, subject to impossible calibration EM >=70%. This predeclared safeguard addresses the user's
request not to sacrifice impossible detection. It is not a guarantee of 70% on unseen data.
Compare original, recalibrated and continued profiles on calibration only; select the best eligible profile by
weighted EM, then weighted F1. Freeze and persist that decision before final evaluation. Report all three
profiles on both holdouts, with unchanged weights/settings and the same examples for fair comparisons.
Public questions are only the last functional/accuracy check; do not select a model from public performance.

Evidence and recovery: configs/continued20k.json, logs/continued20k/plan.json, data_manifest.json,
job.json, history_live.json, runtime.json and training_execution.txt. The first three are publishable metadata;
datasets, weights, checkpoints, process logs and prediction caches remain ignored.
To request a safe stop, create logs/continued20k/STOP; the callback saves before exiting. Do not launch another
training while job.json reports running. Any resumption needs review of the saved state and remaining time budget.

## Completed results

The run completed flag is True; actual epochs 2.000, 2534 steps, 42.98 minutes. Best checkpoint: checkpoints/continued20k\checkpoint-1600.

Candidate fixed on calibration before final evaluation: **continued**. Fresh holdout comparison:

| Profile | Overall EM | Overall F1 | Answerable EM | Answerable F1 | Impossible EM | 75/25 EM | 75/25 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 52.25 | 53.50 | 22.35 | 24.87 | 81.91 | 37.24 | 39.13 |
| recalibrated | 37.70 | 41.94 | 47.65 | 56.17 | 27.82 | 42.69 | 49.08 |
| continued | 53.61 | 56.18 | 33.92 | 39.08 | 73.15 | 43.73 | 47.60 |

Public cases: 4/5. CPU 50 requests: 14.08s; max 0.632s. See comparison.json for the previously consulted holdout and paired bootstrap intervals. No configuration was adjusted from these final results.

## Interpretation and previously consulted holdout

The continued candidate was locked in commit bac0cba, before both final comparisons. On the fresh set,
weighted EM gains 6.49 points over the original (paired bootstrap 95% interval 3.99–8.88).
The gain over recalibrated is only 1.04 points (interval -1.70–3.68), so superiority on that average is not
statistically established. The new candidate instead offers a much stronger correct-null rate: 73.15 vs 27.82,
while answerable EM falls from 47.65 to 33.92 relative to the aggressive recalibration.
Compared with original, answerable EM improves 22.35 -> 33.92 but impossible EM declines 81.91 -> 73.15.
These trade-offs are explicit; no profile dominates every metric. We retain the candidate selected using the
predeclared calibration safeguard, not merely because it is newest. Accuracy is still modest and does not
justify a high-grade guarantee. The hidden test uses non-SQuAD texts and an unknown partial-credit judge.

Separate historical holdout (1,024, already consulted; 511 answerable / 513 impossible):

| Profile | EM | F1 | Answerable EM | Impossible EM | 75/25 EM |
|---|---:|---:|---:|---:|---:|
| original | 52.25 | 53.30 | 24.85 | 79.53 | 38.52 |
| recalibrated | 41.11 | 44.68 | 48.73 | 33.53 | 44.93 |
| continued | 56.05 | 57.58 | 36.01 | 76.02 | 46.01 |

The fresh table above uses 510 answerable / 514 impossible examples and must not be mixed with this table.
Historical baseline metrics were corrected for normalized-empty references before this experiment.

## CPU measurements

Same laptop and the same five public inputs repeated ten times; measurements were taken at different times.
The first two measurements are reused evidence, not reruns. Differences do not establish an inference speedup.

| Profile | Mean CPU request (s) | Maximum (s) | 50 HTTP requests (s) |
|---|---:|---:|---:|
| Original, historical | 0.237 | 0.559 | 12.51 |
| Recalibrated, historical | 0.295 | 0.694 | 15.32 |
| Continued, current | 0.271 | 0.632 | 14.08 |

All meet the local 5s/request and 150s/50-request limits. Network timing for the new candidate remains unverified.
47 automated tests and 19 real-model HTTP checks passed. One obsolete default-model assertion was updated
from subset12k to continued20k; its initial failure is retained in pytest_initial.txt. Public result is 4/5:
sample 04 still returns an unsupported GPT-3 quantity for a BERT-base question.
Do not present the project as ready for the official dry run until this quality failure and operational gates
have been reviewed. No more training, threshold changes or public-specific fixes were attempted.
