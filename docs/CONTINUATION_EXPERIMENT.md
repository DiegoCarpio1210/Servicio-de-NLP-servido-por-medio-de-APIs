# One authorized continuation experiment

Status: training launched; results pending. Do not confuse the planned run with completed evidence.
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
