# Operating runbook — Team Pimientero

The application is English-only. quality40k checkpoint-2000 is the active model.
The autonomous pretrained benchmark is separate and never promotes a candidate.
No new tunnel is authorized by the current benchmark task.

## 1. Local desktop application

Double-click `INICIAR_PIMIENTERO.bat`. Open http://127.0.0.1:8765 when ready.
Use `DETENER_PIMIENTERO.bat` to stop the owned services. This route generates a
random in-memory token automatically; it needs no instructor token.

## 2. Academic API with an instructor token

Only for a coordinated official session, open PowerShell in the project folder:

```powershell
$env:TEAM_ID = 'Pimientero'
$env:MODEL_DIR = (Resolve-Path 'models/quality40k').Path
$env:TORCH_NUM_THREADS = '4'
$qaSecret = Read-Host 'Instructor QA token' -AsSecureString
$env:QA_API_TOKEN = [System.Net.NetworkCredential]::new('', $qaSecret).Password
Remove-Variable qaSecret
.\.venv\Scripts\python.exe -m uvicorn qa.api:app --host 127.0.0.1 --port 8000 --workers 1
```

The value is entered through a hidden prompt, not pasted into command history.
Keep this window open. Do not use reload or multiple workers. In another window,
`curl.exe -i http://127.0.0.1:8000/health` should return 200 and `{"status":"ok"}`
once ready; 503 while loading is expected. The protected `/info` should identify
step 2000, run steps 2531, threshold -0.40494251441686124 and the quality40k hash
in `configs/active_model.json`. Never send the token in chat or commit it.

## 3. HTTPS tunnel — only after separate authorization

With the academic API ready, run one authorized tool in another terminal:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
# Alternative, using the locally configured ngrok account:
ngrok http 8000
```

Keep both terminals open. Share the current HTTPS URL only for the coordinated
session. Quick Tunnel URLs change between runs. Do not expose the UI gateway.
Never assume a successful local request proves external connectivity.

Cloudflare previously passed five mobile functional checks; ngrok passed four
HTTPS checks from the laptop. Those are historical continued20k results, not new
external checks of a pretrained candidate. All those tunnels were closed.

## 4. Secure ngrok setup, if required on another machine

Use the official ngrok dashboard and copy only the account authtoken. Do not post
it in chat. Read it through a PowerShell secure prompt and configure it locally:

```powershell
$ngSecret = Read-Host 'ngrok authtoken' -AsSecureString
$ngValue = [System.Net.NetworkCredential]::new('', $ngSecret).Password
ngrok config add-authtoken $ngValue
Remove-Variable ngSecret, ngValue
Set-Clipboard -Value ' '
```

The command history contains variable names, not the literal credential. ngrok
stores its configuration outside the repository. `ngrok config check` validates
syntax; an actual authorized connection is needed to verify account acceptance.

## 5. External phone check and official rehearsal

Follow `PHONE_TEST_ES.md` (English instructions, historical filename retained).
Turn off phone Wi-Fi, use mobile data and ensure the laptop is not on its hotspot.
Verify health, missing/invalid-token rejection, authenticated info and POST predict.
A browser alone cannot conveniently set the Authorization header; use the approved
HTTP Request Shortcuts procedure on Android. Record actual responses and network
confirmation. Never claim that an unperformed mobile test passed.

The `scripts.rehearse` runner checks the supplied endpoint, schemas, authentication
and deadlines. Consult `--help` for its arguments. Fifty sequential requests must
meet 5 seconds per request and 150 seconds overall. Repeating public samples is
not an independent hidden-set evaluation. Public sample-04 remains incorrect.

## 6. Safe shutdown

Stop the tunnel with Ctrl+C, then the manual uvicorn process with Ctrl+C. Remove
the token from that PowerShell session with `Remove-Item Env:QA_API_TOKEN`.
For the desktop launcher, use the stop BAT instead. Confirm the public URL no
longer reaches the API and local health refuses connections. Stop only services
you own. Do not delete models, checkpoints or previous evidence.

## 7. Autonomous benchmark

Run `RUN_PRETRAINED_BENCHMARK.bat` once. It launches a hidden local Python process.
Follow `logs/pretrained_benchmark/progress.json`, `pid.json`, `execution.log` and
candidate-specific progress files. The operating system must remain awake and the
process alive; Codex credits are not required after launch. See PRETRAINED_BENCHMARK.md.
