# Pimientero AI — Extractive Question Answering API

**Submission model:** `microsoft/deberta-v3-base` (base checkpoint, not SQuAD fine-tuned), trained by our team with **LoRA rank 4 and a newly initialized QA head**. Training ran for **5,500 optimizer steps**; the selected model is **checkpoint 3,500**. This branch, `feature/pimientero-student-primary`, serves our own trained model. Historical experiments in the repository are not the submission model.

**Official report and evidence:** [REPORT.md](REPORT.md) · [Dry-run screenshots](docs/dry_run/) · [Training loss curve](logs/deberta_base_training/train/learning_curves.png) · [Training history](logs/deberta_base_training/train/training_history.json).

## Results and limitations

| Evaluation | Exact Match | Token F1 | Note |
|---|---:|---:|---|
| SQuAD 2.0 validation (500 items; 250 answerable / 250 unanswerable) | 68.40% | 74.65% | Balanced cohort |
| SQuAD 2.0 validation (1,000 items; 750 answerable / 250 unanswerable) | 66.30% | 72.87% | Different answerability mix |
| Instructor's five public examples via HTTPS | 3/5 correct | — | `sample-04` false positive; `sample-05` false negative |

The five public examples took **6.45 s in total**, with observed individual HTTPS round trips of **0.93–2.28 s**. These measurements are not a guarantee of performance on unseen questions or under different network conditions. A response's `score` is not a calibrated probability of correctness.

## Download the trained model (required on a fresh clone)

Git intentionally excludes model weights and datasets. Download **`Pimientero-AI-DeBERTa-3500.zip`** from the [v1.0.0-student Release](https://github.com/DiegoCarpio1210/Servicio-de-NLP-servido-por-medio-de-APIs/releases/tag/v1.0.0-student).

Extract the ZIP to the repository's **`models/`** directory, yielding:

```text
models/
  deberta-base-student/
    model.safetensors
    config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    spm.model
    calibration.json
    training_metadata.json
    added_tokens.json
```

ZIP SHA-256: `8c1381b071e321a58979a38d92457e9691476b68123e04a96e7cf2fa979ac43e`.

`model.safetensors` SHA-256: `1cd3fa25675e3cd5b42f26d0ab8ade8368fceb7971e3aaa3f1df1aaa12d36cce`.

The model launcher validates configured file hashes against [`configs/active_model.json`](configs/active_model.json). **Do not run `scripts.restore_deberta`** for the submission: that older script restores a historical specialized model instead.

## Set up on Windows (Python virtual environment)

From the repository root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Option A: Local graphical interface

Run `INICIAR_PIMIENTERO.bat`. Once loaded, open **http://127.0.0.1:8765**. The launcher runs a loopback-only FastAPI process on port **8000**, the local UI gateway on port **8765**, and generates a temporary private API token for its child processes. Stop via `DETENER_PIMIENTERO.bat` or Ctrl+C in the launcher console.

This mode is local only. **Do not publicly expose the UI gateway (`8765`).**

### Option B: Instructor-facing API with a known Bearer token

Start an independent PowerShell session in the repository root. Use a private development token, not a token committed to source control:

```powershell
$env:TEAM_ID = 'Pimientero'
$env:MODEL_DIR = (Resolve-Path 'models\deberta-base-student').Path
$env:TORCH_NUM_THREADS = '2'
$secret = Read-Host 'QA_API_TOKEN' -AsSecureString
$env:QA_API_TOKEN = [System.Net.NetworkCredential]::new('', $secret).Password
Remove-Variable secret
.\.venv\Scripts\python.exe -m uvicorn qa.api:app --host 127.0.0.1 --port 8002 --workers 1
```

The API binds to `127.0.0.1`; a tunnel securely proxies requests from a public HTTPS address to it. If you want to use port **8000** as in the brief, stop the local graphical launcher first and replace `8002` with `8000`.

In a **separate** terminal (with Cloudflare Tunnel's executable installed locally), run:

```powershell
.\tools\cloudflared.exe tunnel --url http://127.0.0.1:8002
```

Cloudflare Quick Tunnel generates a **temporary** `https://...trycloudflare.com` URL. It has no uptime guarantee, and the laptop, Uvicorn and `cloudflared` must remain running. Share the current URL with the instructor for the evaluation. The professor-provided presentation token can replace the development token by **restarting only the instructor-facing API** with the new `QA_API_TOKEN` environment variable; do not hardcode it.

## API contract

All response bodies use JSON. `/health` is public; `/info` and `/predict` require `Authorization: Bearer <QA_API_TOKEN>`.

| Method | Endpoint | Authentication | Purpose |
|---|---|---|---|
| GET | `/health` | None | `200 {"status":"ok"}` when loaded; `503` while loading |
| GET | `/info` | Bearer | Team, original model ID and parameter count |
| POST | `/predict` | Bearer | Extractive answer or `null` |

Example JSON request for `/predict`:

```json
{
  "id": "demo-01",
  "context": "The robotics laboratory was founded in 2021 by Professor James Wilson.",
  "question": "Who founded the robotics laboratory?"
}
```

Response keys: `id`, `answer`, `start`, `score`. `answer` must be an exact substring of `context`, or `null`; `start` is the zero-based character index, or `null` with a null answer. The service is designed for English questions, including questions with no supported answer and contexts requiring overlapping windows.

## Training records and reproducibility

- [Training configuration](logs/deberta_base_training/train/config.json): 20,000 SQuAD 2.0 training examples, 21,715 token windows; rank 4, alpha 8, LoRA LR `2e-4`, QA-head LR `5e-4`, microbatch 1, gradient accumulation 8, FP32, 256-token windows / stride 96.
- [Training history](logs/deberta_base_training/train/training_history.json), [learning curves](logs/deberta_base_training/train/learning_curves.png), [final summary](logs/deberta_base_training/train/final_summary.json), and [validation metrics](logs/deberta_base_training/train/validation_metrics.json).
- 5,500 optimizer updates, with checkpoint 3,500 selected based on validation loss and final no-answer threshold **`-1.061314582824707`**.
- [1,000-item summary](logs/pimientero_evaluacion/report_1000.json) and [official dry-run evidence](docs/dry_run/) document separate quality checks.

The instructor's five public examples were not used to claim improved accuracy. Two errors remained: `sample-04` incorrectly extracted `3 times` instead of returning `null`, and `sample-05` returned `null` rather than `50,257` in the long context.

## Submission

The instructor requested a Git tag named **`submission`** on the final reviewed commit, with history preserved, no weights in Git, and a completed root-level [`REPORT.md`](REPORT.md). The supporting screenshots are in [`docs/dry_run/`](docs/dry_run/). The trained weight package is an asset of [v1.0.0-student](https://github.com/DiegoCarpio1210/Servicio-de-NLP-servido-por-medio-de-APIs/releases/tag/v1.0.0-student).

This README describes the **student-trained DeBERTa submission**. Previous model comparisons and benchmark documentation elsewhere in the repository are historical and are not instructions to switch the active model.
