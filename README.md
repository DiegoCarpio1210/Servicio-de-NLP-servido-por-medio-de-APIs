# Pimientero AI — Extractive Question Answering API

**Submission model:** `microsoft/deberta-v3-base` (base checkpoint, not SQuAD fine-tuned), **fully fine-tuned by our team** on 94,000 SQuAD 2.0 questions with a newly initialized QA head. Training ran for **5,929 optimizer steps** (1 epoch, 38.9 min on an RTX 4050 Laptop GPU). The branch `feature/full-finetune` serves this model by default. Earlier experiments in the repository (including our LoRA rank-4 model) are historical and are not the submission model.

**Official report and evidence:** [REPORT.md](REPORT.md) · [Training loss curve](logs/deberta_full_fast/loss.png) · [Training history](logs/deberta_full_fast/training_history.json) · [Validation results](logs/deberta_full_fast/results.json) · [Dry-run screenshots](docs/dry_run/).

## Results and limitations

| Evaluation | Exact Match | Token F1 | Note |
|---|---:|---:|---|
| SQuAD 2.0 validation, 1,500 held-out items (727 answerable / 773 unanswerable) | 82.93% | 85.38% | Articles not used for threshold calibration |
| Same items, reweighted to 75% answerable / 25% unanswerable | **83.43%** | **87.22%** | Mix announced for the hidden test set |
| Previous LoRA rank-4 model on the same items, 75/25 | 65.00% | 71.10% | Replaced by this model |
| Instructor's five public examples (local CPU, HTTP) | 4/5 correct | — | `sample-04` false positive |

On the serving laptop (Intel Core i7-12650H, CPU only, 4 threads), each `/predict` request took **0.42–1.27 s** over the five public examples. The official HTTPS dry run used our previous model with 0.93–2.28 s round trips; tunnel latency is added to every request. These measurements are not a guarantee of performance on unseen questions or under different network conditions. A response's `score` is not a calibrated probability of correctness.

## Download the trained model (required on a fresh clone)

Git intentionally excludes model weights and datasets. Download **`Pimientero-AI-DeBERTa-full.zip`** from the [v2.0.0-full Release](https://github.com/DiegoCarpio1210/Servicio-de-NLP-servido-por-medio-de-APIs/releases/tag/v2.0.0-full).

Extract the ZIP to the repository's **`models/`** directory, yielding:

```text
models/
  deberta-full-fast/
    model.safetensors
    config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    spm.model
    added_tokens.json
    calibration.json
    training_metadata.json
```

`model.safetensors` SHA-256: `80cdbc06fc1d77c36a45a73a35187cfb9eb2f1b2ec8b76485851722d060f76d5`.

The API and the launcher validate every configured file hash against [`configs/active_model.json`](configs/active_model.json) before serving. **Do not run `scripts.restore_deberta`** for the submission: that older script restores a historical specialized model instead.

To rebuild the weights instead of downloading them, see [Training records and reproducibility](#training-records-and-reproducibility).

## Set up on Windows (Python virtual environment)

From the repository root in PowerShell. Use **Python 3.12**: the pinned `torch==2.7.1` has no wheels for Python 3.14.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q tests
```

The tests should report `62 passed, 3 skipped`. Among other checks, they fail if the active model is not trained from `microsoft/deberta-v3-base`.

### Option A: Local graphical interface

Run `INICIAR_PIMIENTERO.bat`. Once loaded, open **http://127.0.0.1:8765**. The launcher runs a loopback-only FastAPI process on port **8000**, the local UI gateway on port **8765**, and generates a temporary private API token for its child processes. Stop via `DETENER_PIMIENTERO.bat` or Ctrl+C in the launcher console.

This mode is local only. **Do not publicly expose the UI gateway (`8765`).**

### Option B: Instructor-facing API with a known Bearer token

Stop the graphical launcher first (it also uses port 8000). Start an independent PowerShell session in the repository root. Use a private development token, not a token committed to source control:

```powershell
$env:TEAM_ID = 'Pimientero'
$secret = Read-Host 'QA_API_TOKEN' -AsSecureString
$env:QA_API_TOKEN = [System.Net.NetworkCredential]::new('', $secret).Password
Remove-Variable secret
.\.venv\Scripts\python.exe -m uvicorn qa.api:app --host 127.0.0.1 --port 8000 --workers 1
```

No `MODEL_DIR` is needed: [`configs/active_model.json`](configs/active_model.json) selects `models/deberta-full-fast`. Inference uses 4 CPU threads unless `TORCH_NUM_THREADS` is set. The API binds to `127.0.0.1`; a tunnel securely proxies requests from a public HTTPS address to it.

In a **separate** terminal (with Cloudflare Tunnel's executable installed locally), run:

```powershell
.\tools\cloudflared.exe tunnel --url http://127.0.0.1:8000
```

Use `127.0.0.1` rather than `localhost`: on Windows, `localhost` can first try IPv6 and add about 2 s to the first connection. Tested backup: `ngrok http 8000`.

Cloudflare Quick Tunnel generates a **temporary** `https://...trycloudflare.com` URL. It has no uptime guarantee, and the laptop, Uvicorn and `cloudflared` must remain running. Share the current URL with the instructor for the evaluation. The professor-provided presentation token can replace the development token by **restarting only the instructor-facing API** with the new `QA_API_TOKEN` environment variable; do not hardcode it.

## API contract

All response bodies use JSON. `/health` is public; `/info` and `/predict` require `Authorization: Bearer <QA_API_TOKEN>`, compared exactly and case-sensitively.

| Method | Endpoint | Authentication | Purpose |
|---|---|---|---|
| GET | `/health` | None | `200 {"status":"ok"}` when loaded; `503 {"status":"loading"}` while loading |
| GET | `/info` | Bearer | Team, original model ID and parameter count (`401` without a valid token) |
| POST | `/predict` | Bearer | Extractive answer or `null` (`401` without a valid token, `422` for an invalid body) |

Example JSON request for `/predict`:

```json
{
  "id": "demo-01",
  "context": "The robotics laboratory was founded in 2021 by Professor James Wilson.",
  "question": "Who founded the robotics laboratory?"
}
```

Response keys: `id`, `answer`, `start`, `score`. `answer` is an exact substring of `context` with surrounding whitespace trimmed, or `null`; `start` is the zero-based character index, or `null` with a null answer, so `context[start:start+len(answer)] == answer` always holds. The service is designed for English questions, including questions with no supported answer and contexts longer than one 384-token window (overlapping windows, stride 128).

## Training records and reproducibility

- [Training metadata](logs/deberta_full_fast/training_metadata.json): 94,000 SQuAD 2.0 training examples (seed 20261101), 94,857 token windows; full fine-tuning with frozen word embeddings (85.5M trainable parameters), AdamW with LR `3e-5`, linear decay, 10% warmup, effective batch 16, bf16, 384-token windows / stride 128 (same as serving).
- [Training history](logs/deberta_full_fast/training_history.json), [loss curve](logs/deberta_full_fast/loss.png), [validation results](logs/deberta_full_fast/results.json) and [calibration](logs/deberta_full_fast/calibration.json).
- No-answer threshold **`0.8262314796447754`**, chosen on 1,500 validation questions from articles disjoint from the 1,500 reported ones.
- [Public-sample run on CPU](logs/deberta_full_fast/public_cpu.json) and [official dry-run evidence](docs/dry_run/).

To reproduce on a CUDA GPU (about 40 minutes on an RTX 4050), install the CUDA build of torch in a separate environment, then run from the repository root:

```powershell
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-gpu\Scripts\python.exe -m experiments.train_fast --train-examples 94000
.\.venv\Scripts\python.exe -m scripts.activate_full_fast
```

`train_fast` downloads `microsoft/deberta-v3-base` at the pinned revision and SQuAD 2.0, trains, calibrates the threshold and writes `models/deberta-full-fast`; `activate_full_fast` pins it in `configs/active_model.json` and copies the evidence to `logs/deberta_full_fast/`. GPU nondeterminism can change the last digits of the metrics and the weight hash.

The instructor's five public examples were used only for testing, never for training or threshold selection. One error remains: `sample-04` extracts `3 times`, which the context states for GPT-3 rather than BERT-base, instead of returning `null`.

## Submission

The instructor requested a Git tag named **`submission`** on the final reviewed commit, with history preserved, no weights in Git, and a completed root-level [`REPORT.md`](REPORT.md). The supporting screenshots are in [`docs/dry_run/`](docs/dry_run/). The trained weight package is an asset of [v2.0.0-full](https://github.com/DiegoCarpio1210/Servicio-de-NLP-servido-por-medio-de-APIs/releases/tag/v2.0.0-full); [v1.0.0-student](https://github.com/DiegoCarpio1210/Servicio-de-NLP-servido-por-medio-de-APIs/releases/tag/v1.0.0-student) holds the previous LoRA model.

This README describes the **fully fine-tuned DeBERTa submission**. Previous model comparisons and benchmark documentation elsewhere in the repository are historical and are not instructions to switch the active model.
