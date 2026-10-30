# Extractive QA as a local API

University AI Engineering project. DistilBERT BASE is fine-tuned by this repository on SQuAD 2.0.
Production currently uses our own fine-tuned DistilBERT; all inference runs locally on CPU. The user reports that pretrained QA models are now permitted, and a separate benchmark evaluates them without changing production.
**Status: quality40k checkpoint-2000 is now the approved production default; continued20k is preserved as backup. On the same reserved 1,024 examples, weighted EM improves from 47.02% to 54.93% (95% paired interval for gain +5.48 to +10.43 points). Public accuracy remains 4/5. NOT ready for final submission.**
See [measured comparison and independence limitations](docs/QUALITY_RESULTS.md) and
[candidate activation review](docs/CANDIDATE_REVIEW_ES.md). The dashboard displays the verified results of the active quality40k model. The approved threshold and length05 extraction policy are unchanged from evaluation.
See `docs/STATUS.md` for measured results and `docs/COMPLIANCE.md` for every academic/API requirement. Never present smoke or pilot accuracy as full training.

## PIMIENTERO AI — local graphical application

On Windows, double-click **`INICIAR_PIMIENTERO.bat`**. It checks dependencies and the selected model hashes,
starts the CPU API and a separate local UI gateway, waits for readiness, and opens **http://127.0.0.1:8765**
in your default browser. Stop both services with **`DETENER_PIMIENTERO.bat`** or Ctrl+C in the launcher.
No public tunnel, training, dataset download, frontend build or manually entered API token is needed.
The model files must already exist in the ignored `models/quality40k/` directory.

Open **http://127.0.0.1:8765** after the launcher reports readiness. The academic
API is at **http://127.0.0.1:8000**; its protected endpoints require authentication.
The source repository deliberately excludes trained weights, datasets and secrets;
a fresh clone needs the dependencies and separately supplied local model artifacts.

### Source files for the instructor

| Component | Published source |
|---|---|
| Interface | [HTML](frontend/index.html), [CSS](frontend/style.css), [JavaScript](frontend/app.js) |
| Local interface backend | [UI gateway](qa/ui.py) |
| Academic API | [FastAPI service](qa/api.py), [CPU inference](qa/inference.py) |
| Start / stop | [INICIAR_PIMIENTERO.bat](INICIAR_PIMIENTERO.bat), [DETENER_PIMIENTERO.bat](DETENER_PIMIENTERO.bat) |
| Process management | [Python launcher and shutdown](scripts/local_app.py) |
| Verified candidate metrics | [Final comparison](logs/quality20261029/final_comparison.json), [methodology](docs/QUALITY_RESULTS.md) |

The interface includes context-based questions, session history, copy, null-answer handling, light/dark modes,
responsive navigation and a dashboard separating historical evaluation from current session timings.
The bearer token stays in the server processes; the browser talks to the loopback-only gateway.
The academic API on port 8000 and its authentication contract remain unchanged. Do not expose the UI port 8765 publicly.

**[English setup, architecture, security and test evidence](docs/INTERFAZ_LOCAL.md)**.
Evidence: `logs/ui_pytest.txt` (58 passed) and `logs/ui_http_verification.json` (real selected-model smoke checks).
The GUI is an optional local demonstration; it does not change the documented model accuracy or outstanding academic gates.

Historical screenshots in `docs/screenshots/` predate English localization; the current design is preserved.

## Repository map

```text
originals/             Professor's PDF, immutable YAML, samples, report template, master brief
configs/               Reproducible training configurations, including continued20k and quality40k
qa/api.py              Strict contract, exact Bearer authentication, background loading
qa/ui.py               Loopback UI gateway; keeps the API token server-side
frontend/              Local HTML/CSS/JavaScript application; no build or CDN
scripts/local_app.py   Windows startup, model verification, process ownership and shutdown
qa/features.py         Shared overflow windows and CLS/span labels
qa/spans.py            Valid context spans, character offsets, minimum-null aggregation
qa/inference.py        Local calibrated checkpoint, CPU, load once, inference lock
qa/data.py             Seeded stratified subsets and local dataset snapshot
qa/metrics.py          SQuAD-normalized EM/F1, multiple references
scripts/train.py       Real fine-tuning, resume, checkpoints, logs and loss curve
scripts/evaluate.py    Validation-only calibration; separate threshold holdout
scripts/evaluate_public.py  Real HTTP sample evaluation and sequential latency
scripts/verify_local.py     Ephemeral test token; real CPU server lifecycle
scripts/prepare_data.py     Download official data once and preserve local splits
scripts/offline.ps1         Avoid redundant Hub calls after downloading assets
scripts/audit.py            Final reproducibility and submission checks
scripts/smoke.ps1           Five local or external curl checks
models/                Ignored local weights, tokenizer, metadata, calibration
checkpoints/           Ignored resumable optimizer and model checkpoints
logs/                  Real training/evaluation/test evidence (no token)
tests/                 Contract, span, metrics and real tokenizer tests
REPORT.md              Required headings 0-9, honest draft
 docs/                 Architecture, defense, pitch, requirements, issues and status
```

## Windows PowerShell setup

Run in the project directory. Python 3.13 was tested; use a compatible 64-bit Python.
The development environment reused the machine's installed CUDA PyTorch through a venv with system-site-packages.
For a clean installation use the isolated venv below; `requirements.txt` fixes direct dependency versions.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.prepare_data
```

GPU training is optional; serving always forces CPU. To use CUDA, install the PyTorch 2.7.1 wheel matching your
NVIDIA driver from the official PyTorch installation instructions, then confirm:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The measured machine has a Quadro P1000 (4 GiB VRAM), approximately 31.72 GiB system RAM,
and an Intel family 6/model 158 CPU. The QA model has 66,364,418 parameters: approximately 253 MiB for FP32
weights alone. Optimizer checkpoints, base cache and the dataset require additional disk; keep several GiB free.
No unmeasured memory or latency guarantee is made.

## Train and calibrate

```powershell
# Mechanical verification only; not suitable for the presentation:
.\.venv\Scripts\python.exe -m scripts.train --config configs/smoke.json
.\.venv\Scripts\python.exe -m scripts.evaluate --model-dir models/smoke --limit 64

# Historical pilot (retained for reproducibility):
.\.venv\Scripts\python.exe -m scripts.train --config configs/pilot.json
.\.venv\Scripts\python.exe -m scripts.evaluate --model-dir models/pilot
# Historical completed 12k workflow; do not launch again:
.\.venv\Scripts\python.exe -m scripts.run_subset

# CPU training override, much slower:
.\.venv\Scripts\python.exe -m scripts.train --config configs/pilot.json --cpu
# Resume: replace checkpoint-N with an existing saved directory:
.\.venv\Scripts\python.exe -m scripts.train --config configs/subset12k.json --resume checkpoints/subset12k/checkpoint-N
```

A zero size in configuration means the full split. Subsets preserve the answerability ratio with seed 42;
exact counts and fingerprints are recorded. Train and validation remain separate. Questions are capped at 64
tokens; contexts are never silently cut. Windows use 384 tokens and overlap 128; answers outside a window
use CLS supervision. All encoder weights and the newly initialized QA head are trained.
The current schedule is 2 epochs on 12,000 examples, learning rate 3e-5, linear decay, 10% warmup, weight decay .01,
FP32 batch 16 and accumulation 1. Dynamic padding and length grouping remove padded computation without cutting real tokens.

The best checkpoint is selected on 512 validation examples. A separate 512-example partition calibrates
abstention; 1,024 different examples provide independent final scores. The pipeline preserves all IDs to verify
separation. Old pilot evidence used a different split and remains labeled historical. Public samples are excluded
from all learning decisions. No automatic extension to more examples or epochs is allowed.

After downloading once, avoid redundant network access:

```powershell
. .\scripts\offline.ps1
```

To download new assets later, remove `HF_HUB_OFFLINE` and `HF_DATASETS_OFFLINE` from the environment.
Training saves `training_metadata.json`; evaluation creates `calibration.json`. Serving requires both.
Retraining invalidates previous calibration. Never copy a threshold from a different model or tune on public samples.

## Start the actual service

Do not use `models/smoke` for the professor's evaluation. Use the selected `models/quality40k` export; training and calibration are already complete.
Set the registered team ID and request the instructor's development token. Do not paste tokens into source files,
chat, logs or tracked files. This project does not automatically read `.env`.

```powershell
$env:TEAM_ID = 'Pimientero'
$env:MODEL_DIR = (Resolve-Path 'models/quality40k').Path
$env:TORCH_NUM_THREADS = '4'
$secret = Read-Host 'Instructor API token' -AsSecureString
$env:QA_API_TOKEN = [System.Net.NetworkCredential]::new('', $secret).Password
Remove-Variable secret
.\.venv\Scripts\python.exe -m uvicorn qa.api:app --host 0.0.0.0 --port 8000 --workers 1
```

On presentation day stop the server, set the new token, restart, and repeat access checks.
The token is captured at startup, compared exactly including `Bearer`, and never logged.
`/health` is public. Missing configuration/checkpoint keeps health at 503; the server will not pretend to be ready.
Protected predictions during startup return the contract's internal-error shape; wait for health 200 first.
No reload or multiple model workers during assessment.

## Five required curl checks

Run from a second PowerShell terminal with the same token environment, or set it securely again there.
`request.json` contains only the request fields of public sample 01.

```powershell
$base = 'http://localhost:8000'
curl.exe -i "$base/health"
curl.exe -i "$base/info"
curl.exe -i -H "Authorization: Bearer wrong" "$base/info"
curl.exe -i -H "Authorization: Bearer $env:QA_API_TOKEN" "$base/info"
curl.exe -i -X POST "$base/predict" -H "Authorization: Bearer $env:QA_API_TOKEN" -H "Content-Type: application/json" -d "@request.json"
.\.venv\Scripts\python.exe -m scripts.evaluate_public --url $base
.\.venv\Scripts\python.exe -m scripts.evaluate_public --url $base --repeat 10 --output logs/latency-local.json
```

Expected status sequence: 200, 401, 401, 200, 200. The evaluator checks all five answers and offsets;
its nonzero exit status means some sample failed. Repeating 10 times measures 50 sequential calls, not
50 independent hidden questions. Use `scripts.verify_local` for an isolated real-model check with a temporary
random token; it uses the registered team ID `Pimientero`. Historical pilot logs made before team identification retain their original test ID.

## HTTPS tunnel and external rehearsal

Cloudflare is the primary tunnel; ngrok 3.39.11 is installed but requires local account configuration and a backup tunnel test. Consult `docs/SOURCES.md`.
Authenticate ngrok using your account's authtoken in its official configuration flow; this is different from
`QA_API_TOKEN`. Then, in another terminal:

```powershell
ngrok http 8000
# Backup, run separately:
.\tools\cloudflared.exe tunnel --url http://localhost:8000
```

Copy the printed HTTPS URL. Repeat all five curl tests and the HTTP evaluator against that URL **from another
network** (e.g. mobile tethering). Do not add interactive authentication that would block the professor's harness.
Measure the full round trip: <5 seconds each, <150 seconds for 50 sequential requests. Cloudflare was installed and measured through HTTPS from this host; ngrok and
an independent-network test remain manual gates. A Quick Tunnel URL changes when restarted.
Send the current URL to the instructor before the dry run and presentation; resend after changes.
Keep the laptop on AC, disable sleep through Windows power settings for the session, and leave server and tunnel
running. Do not deploy the model to cloud hosting. If both tunnels fail, ask the instructor about the documented
local encrypted-harness fallback.

## Final submission gates

1. Finish substantial training; inspect real loss, validation EM/F1, subgroup scores and threshold sensitivity.
2. Measure public accuracy and CPU/tunnel latency. Fix timeouts; never hard-code public answers.
3. Fill the team, members, repository URL, public URL and real final evidence in REPORT.md.
4. Resolve every PENDING/FAILED item in `docs/STATUS.md`; rehearse pitch and individual Q&A without AI.
5. Run `python -m scripts.audit` and inspect Git history and ignored weights. Share weights in an authorized
   release/shared folder separately. Invite the instructor if the repository is private.
6. Create the final tag only after these gates pass:

```powershell
git add README.md REPORT.md docs qa scripts tests configs logs requirements.txt .gitignore
git commit -m "Record validated final training, API and tunnel results"
git rev-parse HEAD
git tag submission
# Only after choosing and authorizing your remote:
git remote -v  # origin is already configured
git push -u origin main
git push origin submission
```

A commit cannot contain its own hash without changing the hash. Record the final reviewed hash in the handoff
or resolve `submission^{commit}`; do not fabricate a self-referential hash in REPORT.md.
The user authorized publication of development progress to the official repository. The final submission tag remains withheld until validation is complete.

## Team and historical 12k run

Team: **Pimientero**. Members and official repository are recorded in `configs/team.json` and REPORT.md.
The full-data job was stopped. The 12,000-example, two-epoch job completed; measured results are in logs/subset12k/summary.json. Read `docs/STATUS.md` before launching another training process.
The job writes `logs/subset12k/job.json`; model weights and live console logs are not published.

The pre-run estimate and FP32/FP16 batch benchmarks are in `logs/subset12k/`. To stop safely or resume, follow
`docs/STATUS.md`. Do not use the historical `run_full.py` workflow; it has been superseded by `run_subset.py`.

## Completed-model verification

`python -m scripts.verify_completed` is the historical 12k verifier. Current export identity was checked by
`scripts.verify_continuation_artifacts`; results are saved under logs/continued20k.
`python -m scripts.verify_local --model-dir models/quality40k --repeat 10` verifies real HTTP behavior
and timing. `/info` adds `checkpoint_sha256` and `training_steps` to prove the loaded artifact; the original YAML
allows extra response fields. Public evaluation exits 1 when answers are wrong even if the API contract passes.
The independent evaluation already completed; do not retrain merely to start or verify this service.

## Selected continuation: completed, do not retrain

For step-by-step team operation, see [the English runbook](docs/RUNBOOK_ES.md). The prepared HTTPS rehearsal
is `python -m scripts.rehearse --url <HTTPS-URL> --output logs/external_network_verification.json`.
Run it from the other network with `--external-network --network-note "actual device/network description"`;
it prompts for the token without saving it. Current-model HTTPS from the origin laptop passed; a genuinely separate-network rehearsal remains pending.

The run used 20k examples (12k previous + 8k new), two additional epochs, 1e-5 learning rate and a fresh schedule.
It took 42m59s. Exported checkpoint-1600 had the best selection loss; the run itself completed 2,534 steps.
The original models/subset12k and its checkpoints remain intact. /info reports both step counts and the weight hash.
The threshold is -0.98309326171875, fitted only on 512 calibration examples with a 70% impossible-class floor.
The candidate was locked in commit bac0cba before evaluating old and fresh holdouts. See
docs/CONTINUATION_EXPERIMENT.md for the full comparison and its uncertainty. The fresh holdout is now observed;
do not tune on it. Generic scripts.evaluate refuses this manifest-based experiment to avoid using selection data
as calibration. Existing evaluation results are complete and do not need rerunning.

Historical continued20k connectivity evidence (not a new quality40k external test): an authorized temporary Cloudflare test passed: 50 calls in 19.25s, max .755s. The tunnel was closed and three probes returned 530. Another-network testing remains pending. Follow docs/FINAL_REHEARSAL.md for the remaining authorized human-operated rehearsal.

## Autonomous pretrained benchmark

Run `RUN_PRETRAINED_BENCHMARK.bat` once. The independent CPU process evaluates
Deepset DeBERTa-v3-base-SQuAD2, then RoBERTa-base-SQuAD2, plus unchanged quality40k.
Follow `logs/pretrained_benchmark/progress.json`, `pid.json`, `execution.log` and
candidate progress files. Restarting skips completed work. No training, model
activation or public tunnel is performed. See [protocol, licenses and limitations](docs/PRETRAINED_BENCHMARK.md).
