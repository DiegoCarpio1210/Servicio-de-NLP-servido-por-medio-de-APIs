# Project Report — Extractive QA as a Service

Draft with completed 12,000-example training and independent validation. Final submission is PENDING.

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
its quality still depends on sufficient training. Both encoder and newly initialized QA head are optimized.

## 2. Training

| Item | Value |
|---|---|
| Training data | 12,000 SQuAD 2.0 train examples; 4005 unanswerable; seed 42 |
| Validation data | 512 checkpoint selection, 512 calibration, 1,024 independent holdout; disjoint IDs |
| Epochs | 2; 1522 optimizer steps |
| Learning rate and schedule (warmup?) | 3e-5; linear decay; 10% warmup; weight decay .01 |
| Batch size | 16; accumulation 1; FP32; length-grouped dynamic padding |
| max_seq_length / doc_stride | 384 / 128 |
| Hardware and training time | Quadro P1000 4 GiB; 26.38 minutes |

[Real loss curve](logs/subset12k/loss.png). Logs and metadata: `logs/subset12k/`.
Holdout EM/F1: 41.11/44.68; answerable EM/F1:
48.73/55.88; unanswerable EM/F1:
33.53/33.53. The earlier full-data run was stopped before its first checkpoint;
its logs and all pilot artifacts were preserved. No full-data completion is claimed.

## 3. No-answer threshold

Return null when minimum window CLS start-plus-end minus the best valid span score exceeds
**1.78540611**. The exact sweep maximizes 75/25 prior-weighted EM, with weighted F1 as a tie-break.
The 512 calibration examples are separate from checkpoint selection and the 1,024-example final holdout.
Public cases are never used to fit the model or threshold. Lower thresholds increase abstention;
higher thresholds increase unsupported-span risk. The threshold-sensitivity sweep is saved in evaluation.json.

## 4. Serving architecture

FastAPI and uvicorn use one worker. A background loader reads local weights and calibration once, forces CPU,
warms the model, then changes health from 503 to 200. A lock and inference mode govern prediction. All overlapping
windows are scored; answers are sliced from the original context, preserving character offsets. Exact Bearer
comparison uses only `QA_API_TOKEN`; invalid inputs receive the YAML error format.
Cloudflare HTTPS was tested from this host; ngrok is the untested backup. Instructor dry run is PENDING. Optional /info fields expose the loaded weight hash and 1,522 training steps.
Clean-machine PowerShell commands and secure token entry are in README:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:TEAM_ID = 'Pimientero'
$env:MODEL_DIR = 'models/subset12k'
# Set QA_API_TOKEN securely as described in README, after final training/calibration.
.\.venv\Scripts\python.exe -m uvicorn qa.api:app --host 0.0.0.0 --port 8000 --workers 1
.\tools\cloudflared.exe tunnel --url http://localhost:8000
```

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | 16 / 16 attention heads | 16 | 383 | Yes |
| sample-02 | 3.5 days | 3.5 days | 437 | Yes |
| sample-03 | null | 50,000 | 169 | No |
| sample-04 | null | 10,000 times and the GPU memory requirement by 3 times | 263 | No |
| sample-05 | 50,257 / 50,257 tokens | 50,257 | 3157 | Yes |

Real HTTP result: 3/5. Wrong cases remain prediction failures; no public-answer tuning was performed. Full contexts, including the last window, were processed.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 |
| Size on disk | 254.10 MiB including tokenizer and metadata |
| Mean CPU latency | 0.295 s |
| Maximum CPU latency / 50 sequential requests | 0.694 s / 15.32 s |
| Laptop CPU and RAM | Intel family 6/model 158; 31.72 GiB |

This repeats five cases ten times, not the hidden test. The previous-threshold HTTPS test took 20.47 seconds from this host. The adjusted policy has not been retested
through HTTPS; an independent-network rehearsal remains PENDING.

## 7. Problems encountered and how you fixed them

1. **Problem:** the full-data schedule projected about 16 hours on this laptop. **Cause:** 130,319 examples,
   three epochs and fixed padding. **Fix:** user-authorized stratified 12k subset, two epochs, measured FP32
   batch 16, length grouping and dynamic padding. Actual duration: 26m22s. **Commit:** `020603a`.
2. **Problem:** the API default still referenced the obsolete `models/final` directory. **Cause:** deployment
   configuration lagged behind the selected checkpoint. **Fix:** default to `models/subset12k`, expose its SHA-256
   and training-step count, and verify the real HTTP model identity. **Commit:** `0e4c678`.

## 8. Decoder investigation

Public sample 03 asks about payment, but our model extracts a question count; sample 04 asks about BERT-base,
but the paragraph's numbers describe GPT-3. A decoder could generate an explicit explanation that those facts
do not answer the question. That is a hypothetical capability, not a measured fix: it could still hallucinate.
Generated explanations would violate this API's exact-span contract and need grounding back to offsets or null.
Autoregressive generation adds sequential compute; larger context models add memory costs. No decoder was
implemented or benchmarked. Our current encoder already covers the long public context with windows.

## 9. Known limitations

Recalibration raised 75/25 weighted EM from 38.52 to 44.93 but reduced impossible EM from 79.53 to 33.53.
A subsequent user-authorized 20k continuation is in progress; its outcomes are pending. See docs/CONTINUATION_EXPERIMENT.md.
Instructor dry run, backup tunnel and final report review remain PENDING. Questions are capped internally at
64 tokens, answers at 30 tokens; confidence is not probability-calibrated. Training is English-only.
Malformed inputs fail validation. The final submission tag has not been created.

