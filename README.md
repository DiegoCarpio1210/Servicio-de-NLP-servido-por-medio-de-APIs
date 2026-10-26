# Extractive QA as a local API

University AI Engineering project. DistilBERT BASE is fine-tuned by this repository on SQuAD 2.0.
No commercial inference API or pre-fine-tuned QA checkpoint is used. All inference runs locally on CPU.
**Status: 12,000-example / two-epoch model evaluated. Recalibrated EM 41.11%, F1 44.68%; 75/25 weighted EM 44.93%; public 3/5. NOT yet ready for submission.**
See `docs/STATUS.md` for measured results and `docs/COMPLIANCE.md` for every academic/API requirement. Never present smoke or pilot accuracy as full training.

## Repository map

```text
originals/             Professor's PDF, immutable YAML, samples, report template, master brief
configs/               subset12k (current), smoke/pilot/extended/full (historical)
qa/api.py              Strict contract, exact Bearer authentication, background loading
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
# Current authorized run (do not start twice):
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

Do not use `models/smoke` for the professor's evaluation. Use `models/subset12k` after training and calibration.
Set the registered team ID and request the instructor's development token. Do not paste tokens into source files,
chat, logs or tracked files. This project does not automatically read `.env`.

```powershell
$env:TEAM_ID = 'Pimientero'
$env:MODEL_DIR = (Resolve-Path 'models/subset12k').Path
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

Install ngrok from its official installer and Cloudflare's cloudflared as the backup. Consult `docs/SOURCES.md`.
Authenticate ngrok using your account's authtoken in its official configuration flow; this is different from
`QA_API_TOKEN`. Then, in another terminal:

```powershell
ngrok http 8000
# Backup, run separately:
cloudflared tunnel --url http://localhost:8000
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
git remote add origin https://github.com/DiegoCarpio1210/Servicio-de-NLP-servido-por-medio-de-APIs.git
git push -u origin main
git push origin submission
```

A commit cannot contain its own hash without changing the hash. Record the final reviewed hash in the handoff
or resolve `submission^{commit}`; do not fabricate a self-referential hash in REPORT.md.
The user authorized publication of development progress to the official repository. The final submission tag remains withheld until validation is complete.

## Team and current 12k run

Team: **Pimientero**. Members and official repository are recorded in `configs/team.json` and REPORT.md.
The full-data job was stopped. The 12,000-example, two-epoch job completed; measured results are in logs/subset12k/summary.json. Read `docs/STATUS.md` before launching another training process.
The job writes `logs/subset12k/job.json`; model weights and live console logs are not published.

The pre-run estimate and FP32/FP16 batch benchmarks are in `logs/subset12k/`. To stop safely or resume, follow
`docs/STATUS.md`. Do not use the historical `run_full.py` workflow; it has been superseded by `run_subset.py`.

## Completed-model verification

`python -m scripts.verify_completed` checks weight identity, all saved calibration/holdout metrics and 32 fresh
predictions. `python -m scripts.verify_local --model-dir models/subset12k --repeat 10` verifies real HTTP behavior
and timing. `/info` adds `checkpoint_sha256` and `training_steps` to prove the loaded artifact; the original YAML
allows extra response fields. Public evaluation exits 1 when answers are wrong even if the API contract passes.
The independent evaluation already completed; do not retrain merely to start or verify this service.
