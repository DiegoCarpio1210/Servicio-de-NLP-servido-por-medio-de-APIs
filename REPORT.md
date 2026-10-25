# Project Report — Extractive QA as a Service

Draft: the full-data run was stopped; the authorized 12,000-example run is in progress. Final submission is PENDING.

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
| Training data (full SQuAD 2.0 train or subset; how many examples) | Pilot: 1,024 official train examples; 342 unanswerable; seed 42 |
| Validation data (what you validated on, how many examples) | 128 official validation examples; 64 unanswerable |
| Epochs | 2; 130 optimizer steps |
| Learning rate and schedule (warmup?) | 3e-5, linear decay, 10% warmup, weight decay .01 |
| Batch size | 4, accumulated over 4 steps; effective 16 |
| `max_seq_length` / `doc_stride` | 384 / 128; question cap 64 tokens |
| Hardware and training time | Quadro P1000, 4 GiB VRAM; 325.61 seconds |

**Loss curve:** [real pilot loss](logs/pilot/loss.png); exact history and metadata are alongside it.
**Validation results:** threshold holdout (64 examples): EM 50.00, F1 50.00;
answerable EM/F1 0/0, unanswerable EM/F1 100/100. This null-heavy pilot is inadequate.
The full-data run was stopped. The replacement uses 12,000 examples and two epochs; its metrics are PENDING.

## 3. No-answer threshold

`qa.spans.aggregate` compares minimum window CLS start-plus-end against the best valid context span.
Return null when null-minus-span exceeds **1.94921350** for this pilot only.
An exact sorted sweep maximizes F1 on 64 calibration examples, using EM as a tie-break. The other 64 are
held out from threshold fitting, but validation loss was used for checkpoint selection. Public cases are excluded.
Lower thresholds increase abstention; higher thresholds risk unsupported answers. Full-model calibration is PENDING.

## 4. Serving architecture

FastAPI and uvicorn use one worker. A background loader reads local weights and calibration once, forces CPU,
warms the model, then changes health from 503 to 200. A lock and inference mode govern prediction. All overlapping
windows are scored; answers are sliced from the original context, preserving character offsets. Exact Bearer
comparison uses only `QA_API_TOKEN`; invalid inputs receive the YAML error format.
Cloudflare HTTPS was tested from this host; ngrok is the untested backup. Instructor dry run is PENDING.
Clean-machine PowerShell commands and secure token entry are in README:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:TEAM_ID = 'Pimientero'
$env:MODEL_DIR = 'models/final'
# Set QA_API_TOKEN securely as described in README, after final training/calibration.
.\.venv\Scripts\python.exe -m uvicorn qa.api:app --host 0.0.0.0 --port 8000 --workers 1
.\tools\cloudflared.exe tunnel --url http://localhost:8000
```

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | 16 / 16 attention heads | null | null | No |
| sample-02 | 3.5 days | null | null | No |
| sample-03 | `null` | null | null | Yes |
| sample-04 | `null` | null | null | Yes |
| sample-05 | 50,257 / 50,257 tokens | null | null | No |

Real HTTP pilot results: 2/5; every response was contract-valid. The model is undertrained and strongly favors
abstention. The long case is processed across windows; null is a prediction failure, not context truncation.
No public answer was memorized and no threshold was tuned on these examples.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,364,418 |
| Size on disk | 254.09 MiB including tokenizer/metadata |
| Mean latency per `/predict` request on your laptop (CPU) | 0.345 s; maximum 0.930 s; 50 sequential requests 18.06 s |
| Laptop CPU and RAM | Intel family 6/model 158; 31.72 GiB RAM |

Measurements repeat the five public examples ten times; they do not predict hidden-set accuracy.
HTTPS: 50 calls in 32.21 s, maximum 1.145 s, from this host. Independent-network rehearsal is PENDING.

## 7. Problems encountered and how you fixed them

1. **Problem:** cached runs retried blocked Hub connections. **Cause:** automatic metadata HEAD requests.
   **Fix:** explicit offline mode and persistent local dataset snapshots; offline evaluation succeeded.
   **Commit:** `06b6148`.
2. **Problem:** reading the guide failed at console output. **Cause:** Windows cp1252 could not encode its
   approximation symbol. **Fix:** save extraction as UTF-8 and read explicitly as UTF-8; all twelve pages reviewed.
   **Commit:** `8955d78` preserves the verified extraction.

## 8. Decoder investigation

The observed pilot null failures need more supervised learning; a decoder would not automatically correct
undertraining. A trained decoder could explain unsupported answers or combine non-contiguous facts, while our
span-only output cannot. However, generated explanations would violate this exact-span API and require grounding
back to offsets. Autoregressive generation also adds sequential compute and evaluation complexity. A larger
context-capable decoder could reduce windowing, at a memory cost; no decoder speed or accuracy was measured.
Decoder models are excluded from the implementation.

## 9. Known limitations

Pilot answerable accuracy is unacceptable. The 12,000-example run, its calibration, candidate metrics and the
instructor dry run remain PENDING. Questions beyond 64 tokens lose their suffix internally; spans above 30 tokens
cannot be returned. Confidence is not probability-calibrated. English-only training limits other languages.
Malformed or oversized requests fail validation. Tunnels require Internet, AC power and disabled sleep.
