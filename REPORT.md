# Project Report — Extractive QA as a Service

Measured draft; official submission is PENDING. Production now uses approved quality40k checkpoint-2000; continued20k is preserved as backup.

## 0. Team

| Field | Value |
|---|---|
| Team id | Pimientero |
| Members | Diego Roberto Carpio Flamenco; Julio Roberto Guardado Quijano; Alfredo Enmanuel Cortez Martínez; Iván Daniel Salguero Esperanza |
| Repository URL | https://github.com/DiegoCarpio1210/Servicio-de-NLP-servido-por-medio-de-APIs |
| `submission` commit hash | PENDING; tag not authorized |
| Public service URL used at the dry run | PENDING official instructor dry run |
| AI tools used (which, and for what) | OpenAI Codex: code, tests, experiment orchestration and documentation; no commercial inference in the service |

## 1. Model choice and justification

We fine-tuned `distilbert/distilbert-base-uncased`, BASE revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, including its QA head and encoder.
Six layers and 66,364,418 parameters suit laptop CPU inference. Twelve-layer
BERT-base is a permitted but costlier alternative, not locally benchmarked.
A downloaded SQuAD-fine-tuned model would not demonstrate our own BASE training.
LoRA reduces trainable parameters but does not guarantee greater accuracy.

## 2. Training

| Run | Training examples | Epochs completed | Run steps | Selected step | Time |
|---|---:|---:|---:|---:|---:|
| subset12k | 12,000 | 2 | 1,522 | 1,522 | 26m22s |
| continued20k | 20,000 | 2 | 2,534 | 1,600 | 42m59s |
| quality40k | 40,000 | 1 | 2,531 | 2,000 | 43m11s |

quality40k continues the preserved checkpoint-1600 with 20,000 previous and
20,000 new SQuAD 2.0 examples, including 13,406 impossible questions. Its selected
checkpoint minimizes loss on 512 selection examples: 1.441129 at epoch 0.7902.
The full run completed epoch 1.0; exported weights match checkpoint-2000.

| Setting | quality40k |
|---|---|
| GPU / peak reserved memory | Quadro P1000 / 2,798 MiB |
| LR / warmup / decay | 2e-5 / 5% / linear; weight decay .01 |
| Batch / accumulation / precision | 16 / 1 / FP32; dynamic padding and length grouping |
| Sequence / stride / seed | 384 / 128 / 42 |
| Checkpoints | Every 200 steps; selection evaluation every 400; resumable optimizer/RNG files |

[Loss curve](logs/quality40k/loss.png), [training metadata](logs/quality40k/training_metadata.json),
and [artifact hashes](logs/quality20261029/artifact_audit.json) provide evidence.
No additional training was performed during recovery.

Both candidates were scored on the same frozen 1,024 examples:

| Profile | EM | F1 | Answerable EM | Answerable F1 | Impossible EM | 75/25 EM |
|---|---:|---:|---:|---:|---:|---:|
| continued20k stable | 57.71 | 60.81 | 36.33 | 42.52 | 79.10 | 47.02 |
| continued20k short15 | 57.81 | 60.83 | 36.33 | 42.36 | 79.30 | 47.07 |
| quality40k + length05 | 61.82 | 65.53 | 48.05 | 55.47 | 75.59 | 54.93 |

The trained candidate gains 7.91 weighted-EM points; paired context-cluster
bootstrap 95% CI is [+5.48,+10.43], using 3,000 resamples. Impossible EM falls
3.52 points. This supports recommending quality40k for approval, not a guaranteed
instructor score. Historical stable EM/F1 53.61/56.18 used different examples.

## 3. No-answer threshold

Stable uses threshold -0.98309326. The experimental candidate scores spans as
start logit + end logit minus `0.5 * log(token_length)`, maximum 30 tokens.
Return null when minimum-window CLS score minus best-span score exceeds
-0.4049425144. Only 512 calibration examples fitted thresholds, maximizing 75/25
weighted EM with a 70% impossible-EM floor. A separate 1,024-example development
set selected the policy. Decisions were committed in `37fc668` before final scoring.
Public and ISS examples never selected parameters. Scores measure threshold
distance, not probability of correctness; the GUI states this explicitly.

## 4. Serving architecture

FastAPI loads one local CPU model, four Torch threads, port 8000. `/health` is
public; `/info` and `/predict` require exact `Bearer` authentication against
`QA_API_TOKEN`, using constant-time comparison. JSON validation enforces the
contract. Overflow windows preserve long contexts; original-string offsets support
Unicode. A separate loopback gateway on 8765 serves the optional GUI and keeps
the API token out of the browser.

Run `INICIAR_PIMIENTERO.bat`; stop with `DETENER_PIMIENTERO.bat`. These now select
quality40k with the frozen length05 policy. For the official session, follow [the secure runbook](docs/RUNBOOK_ES.md)
to set the instructor token locally and start an authorized HTTPS tunnel.
No public service was exposed for this candidate evaluation.

## 5. Results on the public sample set

| Item | quality40k answer | Start | Correct |
|---|---|---:|---|
| sample-01 | 16 | 383 | Yes |
| sample-02 | 3.5 days | 437 | Yes |
| sample-03 | null | null | Yes |
| sample-04 | 10,000 | 263 | No |
| sample-05 | 50,257 | 3157 | Yes |

Both models achieve 4/5. Sample-04 incorrectly attributes a GPT-3 quantity to
BERT. No question-specific rule was added. Mandatory all-public-samples conformance
remains unresolved. [Raw HTTP evidence](logs/quality20261029/cpu_trained/result.json).

## 6. Model size and latency (not graded)

| CPU HTTP profile | 50 calls | Mean | Maximum | Sampled process-tree RAM |
|---|---:|---:|---:|---:|
| Stable | 22.93s | .443s | 1.186s | 826.9 MiB |
| quality40k | 28.54s | .551s | 1.419s | 826.0 MiB |

Both satisfy 5s/request and 150s total. After approved production activation,
50 actual API requests took 13.14s, maximum .552s; outputs matched the locked
candidate exactly. UI gateway answer/null/Unicode/long-context checks passed.
The browser automation timed out, so no new visual-session check is claimed. Calls repeat five public inputs ten times,
not fifty hidden questions. Mean server CPU utilization across all logical cores
was 31.91% versus 32.45%; single-run timings are descriptive. Eighteen real HTTP
contract checks per profile passed, including authentication, Unicode and long
contexts. All temporary servers stopped. Prior stable Cloudflare/mobile and ngrok
checks remain historical evidence, not external tests of quality40k.

## 7. Problems encountered and how you fixed them

1. Full-data training projected approximately 16 hours. Stratified subsets,
   measured FP32 batches and dynamic padding reduced actual time (`020603a`).
2. The API default referenced obsolete `models/final`. Corrected deployment
   configuration and verified hashes/steps through real HTTP (`0e4c678`).
3. ISS speed extraction was too long despite valid offsets and one complete
   window. General policies and a controlled continuation were compared
   (`9a546ec`, `37fc668`); the candidate now extracts the shorter correct speed.
   Unsupported-name extraction remains incorrect. Sixty automated tests pass.

## 8. Decoder investigation

Sample-04 requires recognizing that a number describes another model. A decoder
might explain this mismatch, but that is hypothetical and could hallucinate.
Generated explanations violate our verbatim-span contract unless grounded back to
an offset or null. Autoregressive inference adds sequential compute and memory.
No decoder was trained or benchmarked; the permitted encoder remains the backend.

## 9. Known limitations

All official-validation contexts had been consulted. The new final set therefore
uses reserved official-TRAIN contexts excluded from retained models and other
partitions. An aborted full-data run has unknown consumed IDs; absolute historical
non-exposure cannot be certified. Selection/calibration share 132 contexts despite
disjoint IDs. These local results are not an official unseen test; no retuning
follows final scoring.

The candidate still misses aspirational 65 EM, 70 F1 and 50 answerable EM. Reduced
abstention increases wrong non-null outputs from 19.14% to 23.14% of all examples.
English questions are capped at 64 tokens. Public-sample
conformance, instructor dry run/live service, authorized weights, team defense and
final submission approval remain pending. REPORT.md is retained; no PDF or tag is
created. Details: [quality comparison](docs/QUALITY_RESULTS.md).
