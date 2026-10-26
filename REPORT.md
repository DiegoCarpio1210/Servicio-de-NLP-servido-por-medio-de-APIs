# Project Report — Extractive QA as a Service

Draft with measured baseline and continuation results. Final submission is PENDING.

## 0. Team

| Field | Value |
|---|---|
| Team id | `Pimientero` |
| Members | Diego Roberto Carpio Flamenco; Julio Roberto Guardado Quijano; Alfredo Enmanuel Cortez Martínez; Iván Daniel Salguero Esperanza |
| Repository URL | https://github.com/DiegoCarpio1210/Servicio-de-NLP-servido-por-medio-de-APIs |
| `submission` commit hash | PENDING; tag deliberately not created |
| Public service URL used at the dry run | PENDING instructor dry run |
| AI tools used (which, and for what) | OpenAI Codex: implementation, tests, training orchestration, documentation; no commercial inference in the service |

## 1. Model choice and justification

**Base checkpoint:** `distilbert/distilbert-base-uncased`, revision `12040accade4e8a0f71eabdb258fecc2e7e948be`.
Six encoder layers reduce computation compared with the twelve-layer `google-bert/bert-base-uncased` alternative.
BERT was evaluated architecturally, not fine-tuned or benchmarked here. DistilBERT is a practical CPU candidate;
its quality still depends on sufficient training. The first run initialized the QA head; continuation retained and optimized both head and encoder.

## 2. Training

Initial run: 12,000 SQuAD 2.0 examples, two epochs, 1,522 steps, 26m22s.
Continuation: nested 20,000 examples (12,000 previous + 8,000 new; 6,675 impossible),
2.00 additional epochs, 2534 steps, 42.98 minutes.
Both use the Quadro P1000 4 GiB and train encoder plus QA head. Original weights/checkpoints are preserved.
The continuation export uses the minimum-loss checkpoint at step 1600
(epoch 1.263), which can precede the end of the completed run.

| Setting | Initial / continuation |
|---|---|
| Learning rate / warmup | 3e-5 / 10%; then 1e-5 / 5%; linear decay, weight decay .01 |
| Batch / accumulation / precision | 16 / 1 / FP32; length-grouped dynamic padding |
| max_seq_length / doc_stride / seed | 384 / 128 / 42 |
| Validation | 512 selection, 512 calibration, old 1,024 holdout, fresh 1,024 context-disjoint holdout |

[Continuation loss curve](logs/continued20k/loss.png). The candidate **continued** was fixed using calibration.
Fresh holdout EM/F1: 53.61/56.18; answerable
33.92/39.08; impossible 73.15.
75/25 weighted EM/F1: 43.73/47.60.
The old holdout was already consulted and was not used to optimize this experiment.

Comparable fresh holdout (510 answerable / 514 impossible), percentages:

| Profile | EM | F1 | Answerable EM | Impossible EM | 75/25 EM |
|---|---:|---:|---:|---:|---:|
| Original | 52.25 | 53.50 | 22.35 | 81.91 | 37.24 |
| Recalibrated | 37.70 | 41.94 | 47.65 | 27.82 | 42.69 |
| Continued candidate | 53.61 | 56.18 | 33.92 | 73.15 | 43.73 |

The continuation gains 6.49 weighted EM points over original, while losing 8.75 impossible-class points.
Its 1.04-point weighted gain over recalibrated is not statistically conclusive; it restores correct-null performance
at the cost of answerable accuracy. Candidate selection used calibration only, before these measurements.

## 3. No-answer threshold

Return null when minimum window CLS start-plus-end minus best valid span score exceeds
**-0.98309326**. The continuation threshold maximizes calibration 75/25 weighted EM,
with weighted F1 tie-break, subject to impossible calibration EM >=70%. This is a calibration safeguard,
not a guaranteed test result. The candidate and thresholds were locked before final evaluations.
Lower thresholds increase abstention; higher ones risk unsupported spans. Public examples never train or calibrate the model. EM/F1 are proxies, not the instructor's partial-credit judge.
See docs/CONTINUATION_EXPERIMENT.md for all three profiles and the historical baseline scoring correction.

## 4. Serving architecture

FastAPI and uvicorn use one worker. A background loader reads local weights and calibration once, forces CPU,
warms the model, then changes health from 503 to 200. A lock and inference mode govern prediction. All overlapping
windows are scored; answers are sliced from the original context, preserving character offsets. Exact Bearer
comparison uses only `QA_API_TOKEN`; invalid inputs receive the YAML error format.
Cloudflare was tested historically with the baseline; current-model HTTPS is PENDING. Ngrok is the untested backup. Instructor dry run is PENDING. Optional /info fields expose the loaded weight hash and actual training steps.
Clean-machine PowerShell commands and secure token entry are in README:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:TEAM_ID = 'Pimientero'
$env:MODEL_DIR = 'models/continued20k'
# Set QA_API_TOKEN securely as described in README, after final training/calibration.
.\.venv\Scripts\python.exe -m uvicorn qa.api:app --host 0.0.0.0 --port 8000 --workers 1
.\tools\cloudflared.exe tunnel --url http://localhost:8000
```

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | 16 / 16 attention heads | 16 | 383 | Yes |
| sample-02 | 3.5 days | 3.5 days | 437 | Yes |
| sample-03 | null | null | null | Yes |
| sample-04 | null | 10,000 times and the GPU memory requirement by 3 times | 263 | No |
| sample-05 | 50,257 / 50,257 tokens | 50,257 | 3157 | Yes |

Real HTTP: 4/5. Sample 04 incorrectly extracts GPT-3 reduction figures for a BERT-base question: a semantic attribution error. Results were observed only after candidate locking; no case-specific fixes.

## 6. Model size and latency (not graded)

66,364,418 parameters; 254.10 MiB including tokenizer/metadata.
CPU mean 0.271s, maximum 0.632s; 50 sequential calls 14.08s.
This repeats five public cases ten times, not fifty hidden questions. Laptop: Intel family 6/model 158,
31.72 GiB RAM. Current-policy HTTPS and independent-network rehearsal remain PENDING;
historical baseline HTTPS took 20.47s and does not validate changed weights/settings.

## 7. Problems encountered and how you fixed them

1. **Problem:** the full-data schedule projected about 16 hours on this laptop. **Cause:** 130,319 examples,
   three epochs and fixed padding. **Fix:** user-authorized stratified 12k subset, two epochs, measured FP32
   batch 16, length grouping and dynamic padding. Actual duration: 26m22s. **Commit:** `020603a`.
2. **Problem:** the API default still referenced the obsolete `models/final` directory. **Cause:** deployment
   configuration lagged behind the selected checkpoint. **Fix:** default to `models/subset12k`, expose its SHA-256
   and training-step count, and verify the real HTTP model identity. **Commit:** `0e4c678`.

## 8. Decoder investigation

The original model extracted a question count for payment in sample 03; the continuation now returns null. Sample 04 asks about BERT-base,
but the paragraph's numbers describe GPT-3. A decoder could generate an explicit explanation that those facts
do not answer the question. That is a hypothetical capability, not a measured fix: it could still hallucinate.
Generated explanations would violate this API's exact-span contract and need grounding back to offsets or null.
Autoregressive generation adds sequential compute; larger context models add memory costs. No decoder was
implemented or benchmarked. Our current encoder already covers the long public context with windows.

## 9. Known limitations

Model accuracy remains dataset-dependent; weighted scores assume the stated 75/25 mix transfers.
The impossible-class calibration floor is not a deployment guarantee. Confidence is not probability-calibrated;
questions cap at 64 tokens, spans at 30. Training is English-only. The new holdout is now observed and must
not become a tuning set. Instructor dry run, backup tunnel, independent-network checks, weight sharing and
final review remain PENDING. No second continuation or submission tag was created.

