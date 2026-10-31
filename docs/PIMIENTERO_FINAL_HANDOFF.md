# Pimientero AI — final development handoff

Audit date: **31 October 2026**. Development and model research are closed for this handoff. Official delivery remains pending external connectivity and academic actions below.

## Selected implementation and evidence

The local extractive QA service uses **deepset/deberta-v3-base-squad2**, CPU, one worker, two Torch threads, 384-token windows, stride 128, maximum answer length 30 and no length penalty. The frozen threshold is **2.7772940397262573**. The exact source revision is `eea39c60cc305c2e4a9504f5ff117294bebb42db`; weights SHA-256 is `aff8759009b728d7d35c1163cd3578b911a3e3e1fbd2522c2b4af398e523f92a`. The source/config/tokenizer hashes are in `configs/active_model.json`. No local DeBERTa fine-tuning is claimed. quality40k remains the backup. LoRA is not active.

| Evidence | Result and scope |
|---|---|
| Historical balanced SQuAD2 benchmark, 500 questions | EM 83.00%, F1 86.69%, answerable EM 77.60%, impossible EM 88.40%; all contexts had prior local exposure. Three requested accuracy targets remain unmet. |
| Synthetic development, 50 questions | EM 46/50 (92%), F1 98.08%; 15/15 correct nulls |
| Separate synthetic final, 50 questions | EM 40/50 (80%), F1 95.29%; answerable EM 73.68%; 12/12 correct nulls, no incorrect nulls |
| Local semantic review of saved final responses | 40 exact + 9 additional correct + 1 ambiguous; conservative 49/50, possible 50/50. **Not Claude, not independent grading, not an instructor score.** |
| Final sequential HTTP | 50/50 valid; 47.10s total, 0.942s mean, 2.317s maximum |
| Final bounded concurrency, two clients | 50/50 valid; 46.18s total, 1.831s mean, 3.032s maximum |
| Total previous readiness traffic | 200 evaluated POSTs, 100 distinct questions, 278 HTTP requests including expected rejection/loading checks |
| Current final audit | `logs/final_audit.json`: protected hashes, saved dataset/results integrity and minimal live HTTP/UI checks; no repeat benchmark |
| Regression and dependencies | 77 experimental-branch tests passed; main has its own smaller production suite. Separate logs identify each run. `pip check` passed. Direct dependency versions are pinned; a fresh isolated installation was not repeated. |

See [semantic decisions](FINAL_SEMANTIC_REVIEW.md), [HTTP methodology](CLAUDE_50_QUESTION_READINESS.md), [500-question limitations](FINAL_500_EVALUATION.md) and [recovery/rollback](STABLE_DEBERTA.md). Raw authored datasets and predictions remain local. Main contains delivery documentation and compact evidence; evaluation tools and research remain on `experiment/deberta-lora`.

## Contract and known limits

The original `originals/qa_api_contract.yaml` is unchanged. `/health` is public (503 while loading, 200 when ready). `/info` and `/predict` require exact, case-sensitive `Authorization: Bearer <token>`. Invalid credentials return 401. Invalid request JSON returns 422 with `bad_request`; internal failures return 500 with `internal_error`. POST requires only `id`, `context`, `question` within 64/5000/500 character limits; extra input fields are rejected. Output requires `id`, `answer`, `start`, `score`; response extras are permitted. Answers copy exact source substrings with zero-based Unicode character offsets; answer/start are null together.

Real HTTP and regression checks cover authentication, schema, nulls, Unicode, context overflow and concurrency. Tokenization and forward inference share one lock. There is no demonstrated defect requiring an inference change. Questions beyond the existing 64-token preprocessing cap remain a disclosed quality limitation despite the 500-character API input limit. CPU timings do not guarantee HTTPS round-trip times or accuracy on unseen private questions.

## Local setup and graphical application

Use the existing environment on this laptop. For a fresh clone only, create a compatible 64-bit Python environment, install the pinned dependencies and restore the public checkpoint:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scripts.restore_deberta
```

Restoration downloads the exact public revision and verifies all manifest hashes. No dataset or training is required for serving. Model weights, credentials and large evaluation data are intentionally absent from GitHub. The current environment was tested, but dependency pins are not a complete transitive lock or a certification of every clean Windows installation.

From the project folder:

```powershell
.\INICIAR_PIMIENTERO.bat
# Browser: http://127.0.0.1:8765
# Readiness: http://127.0.0.1:8000/health
```

Wait for readiness. The launcher verifies hashes, generates a local temporary token and starts API and UI. The token stays server-side. Do not try to reuse or reveal it. The English dashboard intentionally shows the historical 500-question metrics, not the local semantic estimate. Stop using `DETENER_PIMIENTERO.bat` or Ctrl+C in the launcher. The UI gateway must never be exposed publicly.

## Manual checklist

1. Launch the BAT; open `http://127.0.0.1:8765` and confirm DeBERTa and the 500-question dashboard label.
2. Enter context `Paris is the capital of France.` and ask `What is the capital of France?`. Check the highlighted span and source offset. Leading whitespace may be part of a valid extracted span.
3. Enter `The observatory opened in 2008. It moved to the northern hill in 2016 because city lights obstructed observations.` Ask `Why did the observatory move?`. Check that the returned span expresses the supported reason. This is a manual check, not a promised unseen prediction.
4. With the Paris context, ask `Who is the mayor of Paris?`; verify null. The final audit already confirmed this simple live case.
5. Open `/health` and `/info` in the browser: expect 200/ok and 401/unauthorized respectively. Use the separate direct-API procedure below to check valid authentication.
6. Run `DETENER_PIMIENTERO.bat`; confirm ports 8000 and 8765 stop responding. Report any discrepancy instead of changing calibration.

## Direct API, secure token and HTTPS delivery

Stop the graphical launcher before using port 8000 for this separate session. In PowerShell terminal A, inside the project folder:

```powershell
Remove-Item Env:MODEL_DIR -ErrorAction SilentlyContinue
$env:TEAM_ID = 'Pimientero'
$env:TORCH_NUM_THREADS = '2'
$secret = Read-Host 'API token for this session' -AsSecureString
$env:QA_API_TOKEN = [System.Net.NetworkCredential]::new('', $secret).Password
Remove-Variable secret
.\.venv\Scripts\python.exe -m uvicorn qa.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Use the professor-provided token only for the official session, entered locally; for rehearsal use a new random token. Secure input prevents its literal value entering command history; the API still needs plaintext in its process environment. Never paste it into chat, Git, screenshots or the URL. Do not run reload or multiple workers.

In terminal B, enter the same token through `Read-Host -AsSecureString`, then test:

```powershell
$base = 'http://127.0.0.1:8000'
$secret = Read-Host 'Same API token' -AsSecureString
$headers = @{ Authorization = 'Bearer ' + [System.Net.NetworkCredential]::new('', $secret).Password }
Remove-Variable secret
Invoke-RestMethod "$base/health"
curl.exe -i "$base/info" # expected 401
Invoke-RestMethod "$base/info" -Headers $headers
$body = @{id='manual-01';context='Paris is the capital of France.';question='What is the capital of France?'} | ConvertTo-Json
Invoke-RestMethod "$base/predict" -Method Post -Headers $headers -ContentType 'application/json' -Body $body
Remove-Variable headers,body
```

Check `/info`: team Pimientero, model deepset/deberta-v3-base-squad2, params 183833090, hash and threshold above, zero local DeBERTa training steps. `/predict` must include id/answer/start/score and satisfy context slicing; a non-null answer is not automatically factually correct.

**No live public HTTPS endpoint was found or opened during this audit.** After the user authorizes exposure for the grading session, start a separate terminal C:

```powershell
.\tools\cloudflared.exe tunnel --url http://127.0.0.1:8000
# Alternatively, with ngrok installed and its account configured locally:
# ngrok http 8000
```

Use the actual printed HTTPS URL; old Quick Tunnel URLs are not reusable. Do not expose 8765, disable authentication or change firewall/antivirus settings. A ngrok account authtoken is distinct from the API bearer token and stays in its local configuration. Existing historical tunnel tests used earlier models; they do not establish a current DeBERTa endpoint.

From the Samsung on mobile data with Wi-Fi off (laptop not on its hotspot), test health, missing/wrong/valid-token info and authenticated POST. HTTP Request Shortcuts can reuse a Password Input variable with Remember value disabled, header `Authorization: Bearer {api_token}`, and `Content-Type: application/json`. Record actual status/response/time without tokens. The 50-request HTTPS rehearsal must measure <=5s each and <=150s total; local CPU results are not a substitute. Do not use the historical `scripts.rehearse` model-identity assertion unchanged: it reads continued20k evidence. Use the contract-aware evaluation tools on the experimental branch for a separately authorized current-model rehearsal or review the external harness with the instructor.

Keep the laptop on AC, stable Wi-Fi, awake and free of training/other inference workloads during grading. Keep terminals A/C open. Close tunnel C with Ctrl+C first, check the HTTPS URL no longer reaches the API, then stop A with Ctrl+C and run `Remove-Item Env:QA_API_TOKEN`. Close test terminals and discard the temporary token; a stopped API no longer accepts it. Restart only with a new rehearsal token or the authorized official token.

## Git consolidation and remaining gates

Starting verified tips: main `02a9d344af19db8db6479e65be933c6455b6524e`, experiment `6642be436d57e395f3218f90f991309b8b561ab1`. This consolidation copies only README/REPORT/compliance corrections, final review/handoff, readiness report and compact evidence to main. It does not merge research scripts, adapters, experimental thresholds or runtime changes. Final branch tips are verified with `git rev-parse HEAD` and `git ls-remote --heads origin main experiment/deberta-lora`; the publication response records the exact new commits to avoid a self-referential commit hash inside this file.

**Local application: ready. API contract: verified. Live external HTTPS: pending. Official submission: not yet ready.** Remaining actions:

- Complete the user's manual launch/stop/UI checklist and a current-model external-network HTTPS rehearsal; provide the live URL/token privately to the instructor for the official window.
- Public-sample dry-run gate remains unresolved: the saved DeBERTa response to sample-04 is ` 3 times` at offset 309 instead of null. The other four match under normalized scoring; no rerun or question-specific correction was made. Obtain the instructor's actual dry-run outcome; do not claim all public samples pass.
- Confirm the reported instructor exception allowing a pretrained QA checkpoint is accepted for the final deliverable; original instructions require own fine-tuning. Historical DistilBERT training is preserved but must not be represented as training this DeBERTa.
- Preserve model-card attribution and pinned recovery instructions; arrange any separately requested weight delivery without committing weights.
- Complete official dry run, four-person defense/pitch and submission fields. Local semantic review is not an official grade; independent judgment remains separate.
- Create `submission` only after explicit authorization and final acceptance. No tag, public tunnel, research restart or model change was performed here.
