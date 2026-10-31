# Controlled DeBERTa LoRA pilot: actual results

**Completed exactly 200 optimizer updates. No adapter was activated or merged.**
Stable main remains `02a9d34`; work is isolated on `experiment/deberta-lora`. This is development evidence, not an official final evaluation.

## Provenance and overlap audit

Training: **1,024 questions**, 512 answerable AdversarialQA training examples plus 512 genuine SQuAD2 training unanswerables. Development: **128 questions**, 64 AdversarialQA validation answers plus 64 SQuAD2 training unanswerables. A further **128 calibration examples** were unused throughout this pilot; their later separate calibration study is documented in DEBERTA_LORA_CALIBRATION_ANALYSIS.md. Every selected question has a distinct normalized context and normalized question across all three partitions.

AdversarialQA: [UCLNLP dataset](https://huggingface.co/datasets/UCLNLP/adversarial_qa), revision `c2d5f738db1ad21a4126a144dfbb00cb51e0a4a9`, crowdsourced adversarial extractive questions. SQuAD2: [Stanford source](https://rajpurkar.github.io/SQuAD-explorer/). Both are distributed under CC BY-SA 4.0; source file SHA-256 values and local SQuAD fingerprints are frozen in the committed manifest. Original answers and offsets are retained; no artificial null labels were manufactured.

The audit scanned 190 local records, excluding 6963 historical question IDs and 1822 normalized contexts. AdversarialQA train contained 30,000 rows, 51 duplicate context/question pairs and 1,142 historical-overlap exclusions; validation contained 3,000 rows, 3 duplicates and 1,072 historical-overlap exclusions. No invalid or empty answer spans were found. The eligible SQuAD negative pool had 42,470 examples. Selection was frozen before fitting in commit `d03c05b`.

The audit includes the previous 500- and 1,024-question benchmark records and professor public examples. None was used for training, calibration or this development evaluation. The vendor already trained on SQuAD2, and AdversarialQA shares Wikipedia/SQuAD source contexts: this is complementary-question adaptation, not certified unseen data. Exact normalized context/question checks do not exclude every paraphrase or near-duplicate.

**Important design limitation:** answerability is confounded with dataset source in this pilot: all selected AdversarialQA examples have answers and all selected SQuAD examples are impossible. Question-style cues may help classification. Any future quality experiment should include SQuAD positives too and use a separately annotated balanced external set. Do not infer generalization from this mixed-source development cohort.

## Frozen training and inference configuration

Pinned original DeBERTa; rank 4, alpha 8, dropout .05, FP32, microbatch 1, accumulation 8, non-reentrant gradient checkpointing. Only query/value LoRA and the saved QA head train: **148,994 parameters**. LR 2e-5, 10 warmup updates, linear decay through update 200, AdamW weight decay .01, gradient clipping 1.0; seed 20261031. Training windows: 256 tokens, stride 96; questions bounded by the existing shared preprocessing utility. Genuine impossible questions and answer-outside-window cases receive CLS start/end supervision.

Tokenization produced 1111 windows: 524 positive and 587 CLS-supervised windows. The run consumed exactly 1,600 microbatches, using deterministic epoch shuffles over windows; 200 updates is not a claim of 200 epochs or 1,600 distinct source questions.

Development uses the production 384-token/128-stride inference policy, max span 30, length penalty 0, threshold **2.7772940397262573**. No threshold sweep or calibration was performed. Model selection was declared as maximum development 75/25 weighted EM, with ties retaining the earlier state including the original base at step 0. No final test set was consulted.

## Development metrics

| Update | Overall EM | Overall F1 | Answerable EM | Unanswerable EM | Weighted 75/25 EM | Validation loss |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 60.94% | 67.64% | 21.88% | 100.00% | 41.41% | 2.8763 |
| 100 | 60.94% | 68.88% | 21.88% | 100.00% | 41.41% | 2.7005 |
| 200 | 60.94% | 68.88% | 21.88% | 100.00% | 41.41% | 2.6313 |

Selected development state: **step 0**. Overall EM change from the same-cohort base to update 200: +0.00 points; F1 change: +1.24 points.

EM/F1 use the existing official SQuAD normalization and best original reference. Validation loss is the mean of overflow-window losses; metrics are per question. Each class has only 64 questions, so a single class error moves EM by 1.5625 points. This dataset differs from the previous 500-question benchmark; do not subtract its 83.00/86.69 baseline scores from these development results.

## Loss, resources and duration

| Update block | Mean training loss |
|---|---:|
| 1-50 | 3.4339 |
| 51-100 | 2.9302 |
| 101-150 | 3.1632 |
| 151-200 | 3.1019 |

Optimizer-loop time: **461.20 seconds**; mean update **2.306s**. Training-phase wall clock including development evaluation and checkpoints: **523.11s**. Model loading, initial tokenization and the deliberate recovery pause between processes are excluded from this clock; these are not fabricated end-to-end timings.

Maximum recorded PyTorch allocated/reserved peaks across training logs: **797.91 / 846.00 MiB**. Peak recorded process RAM: **1437.30 MiB**. These are allocator/process counters, not total computer usage. Finite losses and gradients were checked on every update; frozen parameters had no gradients. Gradient clipping was applied after checking and before each optimizer step.

## Checkpoints and actual recovery test

The pilot stopped cleanly at update 28 and resumed from `checkpoint-0028`. Verified restoration: cursor 224, 50 optimizer-state entries, scheduler epoch 28, complete history, Python/NumPy/Torch/CUDA RNG state. Updates 1–200 occur exactly once in the final history. No silent restart or extra training budget was used.

Local checkpoints are under `logs/deberta_lora/pilot200/checkpoint-NNNN/`: adapter and QA-head safetensors, trusted local `trainer.pt` (optimizer, scheduler, RNG, cursor, history) and hash metadata. Checkpoints are written to unique partial directories and atomically renamed before updating `latest.json`. Partial directories are never treated as resumable checkpoints. Latest complete checkpoint: `checkpoint-0200`.

From this branch, `PILOTO_LORA.bat` starts only a new frozen run; `PILOTO_LORA.bat --resume` explicitly restores the latest checkpoint. It refuses a silent restart. `DETENER_PILOTO_LORA.bat` creates STOP; the current update finishes and is saved. Remove only `logs/deberta_lora/pilot200/STOP` before explicit resume. A completed 200-update run exits without further updates. Load trainer.pt only from trusted local checkpoints.

For a fresh checkout, recover the pinned public base as documented in STABLE_DEBERTA.md, install isolated PEFT as in DEBERTA_LORA_SMOKE_TEST.md, and restore the exact pinned AdversarialQA files and original cached SQuAD dataset. `python -m experiments.restore_lora_pilot_data` reconstructs the committed ID selection and checks its byte hash; it never resamples after results. Adapter/checkpoint files are deliberately local and must be backed up separately to resume actual training.

## Interpretation and preserved baseline

Decreasing training loss alone does not demonstrate success. Use the table above and its source-confounding/small-sample limitations. This pilot does not establish the experimental targets (overall EM >=85, F1 >=88, answerable EM >=80, unanswerable EM >=95) on an independent final set. No new production metrics replace the stable 83.00/86.69 benchmark.

The stable DeBERTa files, tokenizer, frozen policy and quality40k backup hashes were verified unchanged. API, dashboard, stable requirements and main were not modified. All adapters/data/raw predictions remain ignored; only source code, manifests and compact results are published on experiment/deberta-lora. No automatic activation, public tunnel, submission tag or merge into main occurred.

Evidence: logs/deberta_lora_pilot200_summary.json; full local logs/deberta_lora/pilot200/{training_history,validation_metrics,progress,resume_verification}.json. Calibration was unused during training/checkpoint selection, and no prior final benchmark was rerun.
