# Requirement-by-requirement compliance review

Authority: professor OpenAPI YAML overrides the PDF summary. All four original files are byte-identical to
those in the original ZIP (`logs/source_integrity.json`). This review covers PDF sections I-XV, YAML schemas
and endpoints, the ten report sections, and the master brief. PASSED means the stated check was executed or
its artifact inspected; it never means a guaranteed grade. FAILED distinguishes an observed failure from PENDING.
All evidence paths below are repository-relative. The single authorized continuation is complete; current evidence is logs/continued20k/. Historical 12k evidence remains preserved.

| Source / requirement | Status | Evidence / limitation |
|---|---|---|
| PDF I, III: extractive QA with abstention | PASSED implementation; limited quality | Real CPU predictions, verification.json; group metrics in evaluation.json |
| II: PBL group of 4-5, English work | PASSED | Four full names, Pimientero, English REPORT/README/code |
| II, IV, V.1: own fine-tuning from BASE encoder | PASSED | 12k baseline plus nested 20k continuation; own QA head retained; base revision and source hash recorded |
| IV: encoder choice and alternative justification | PASSED | REPORT §1: DistilBERT versus BERT-base architecture; no unexecuted benchmark claimed |
| V.1, XIII.2: genuine training logs/loss curve | PASSED | logs/continued20k/history.json and loss.png; 42m59s, 2534 run steps, selected checkpoint-1600 |
| V.2: no commercial inference or decoder backend | PASSED | qa/inference.py loads local encoder files only; no inference API calls |
| V.3: exact fixed API | PASSED technical checks | 47 pytest tests and 19 real-model HTTP checks; logs/continued20k/ |
| V.4, VIII: token from environment only | PASSED | qa/api.py; exact hmac comparison; generated ephemeral test tokens; no instructor token stored |
| V.5: retain authentic Git history | PASSED | Incremental commits including 020603a and 0e4c678; no squash |
| VI: permitted reproducible SQuAD 2.0 subset | PASSED | 20,000 unique official train examples, including 8,000 new; 6,675 impossible; manifest and seed recorded |
| VI: validation not used for gradient training | PASSED | Disjoint train/selection/calibration/holdout IDs; verification.json |
| Master 3: independent calibration and evaluation | PASSED | 512 selection, 512 calibration, old 1024 plus fresh 1024 excluding historical contexts; policy locked in bac0cba |
| Master 3: genuine EM/F1 and subgroup results | PASSED measurement | Fresh holdout EM/F1 53.61/56.18; answerable 33.92/39.08; impossible 73.15; weighted EM 43.73 |
| Master 3: no public-example training/tuning | PASSED | Learning uses official train/validation IDs only; threshold recomputed on 512 calibration rows |
| Master 3: correct CLS and overflow labels | PASSED | training_features tests: impossible/outside-window labels, final-window answer, Unicode |
| Master 3: resumable checkpoints/seed/schedule | PASSED | Checkpoints contain optimizer/scheduler/RNG; fixed settings; STOP/locking tests |
| VII general: UTF-8 JSON on port 8000 | PASSED | Real uvicorn HTTP; application/json; strict Pydantic schema |
| VII health public, loading 503, ready 200 | PASSED | Loading/failure lifecycle tests and real server readiness |
| VII info protected, true team/base/parameters | PASSED | continued20k/info.json: Pimientero, 66,364,418 parameters, correct hash, selected steps 1600 / run steps 2534 |
| VII auth exact scheme/token; invalid -> 401 | PASSED | Unit and real HTTP tests, wrong/lowercase/double-space schemes; WWW-Authenticate |
| VII predict strict required fields, no extras | PASSED | Original YAML validation; missing/extra/type/invalid JSON tests |
| VII predict lengths 1-64 / 1-5000 / 1-500 | PASSED | Unit bounds and real 5000-char context/500-char question/64-char id |
| VII errors bad_request/detail, internal_error | PASSED | Generic validation handler and simulated failure tests; no public traceback |
| VII echo id, required answer/start/score | PASSED | Real response schema checks across 50 sequential calls |
| VII offsets are original 0-based characters | PASSED | Unicode/repeated spans, all 1536 validation candidate offsets, public offsets |
| VII paired null and finite [0,1] score | PASSED | Unit checks and real held-out null prediction |
| Master 4: every context window, including tail | PASSED | Overflow tests and actual sample-05 answer at character 3157 |
| Master 4: CPU, one model/worker, eval/inference mode | PASSED | Engine forces CPU; one load; lock; torch.inference_mode; real hash verified |
| Master 4: serving selected 12k model, not pilot | PASSED | Current selected continuation, not pilot: artifact_verification.json and HTTP info.json match checkpoint-1600 |
| VI, XI: five public answers all correct | FAILED | 4/5; sample-04 should be null but returns GPT-3 reduction figures |
| XII: 5-second request / 150-second global cap | PASSED locally | 50 sequential requests 14.08 s, max .632 s; current candidate, public cases repeated ten times |
| IX: HTTPS tunnel to own laptop | PASSED from this host | Current candidate: 50 valid HTTPS calls in 19.25 s, max .755 s; logs/continued20k/tunnel-summary.json; tunnel closed, three 530 probes |
| IX: second tunnel installed and tested | PARTIAL | ngrok 3.39.11 installed/version verified; missing account configuration; no backup tunnel test yet; logs/ngrok_setup.json |
| IX: check from another network | PASSED five manual checks | Samsung mobile data, independent laptop Wi-Fi; user reports plus server logs; logs/continued20k/mobile_connectivity.json. No 50-request mobile timing claimed. |
| IX, X, XI: instructor dry run and send current URL | PENDING | User must arrange slot/send URL; no message sent on user's behalf |
| IX, XII: keep power/no sleep/tunnel active | PENDING presentation operation | README runbook; temporary tests shut down after completion |
| XI, XIII.1: repository shared, weights outside Git | PASSED implementation; publication checked at close | Authentic main history; ignored weights/checkpoints/data/venv/env; current documentation synchronized with origin |
| XI: separate authorized weight distribution | PENDING | Weights remain local; choose release/shared-folder destination before sharing |
| XI, XIII: final submission tag and tagged report | PENDING by explicit user instruction | No submission tag created |
| XIII.3: running service for the live evaluation | PENDING session operation | Test service works but is stopped; start with instructor token for rehearsal |
| Template 0: team/repository/AI disclosure | PASSED; final hash/dry-run URL pending | REPORT §0 accurately identifies four members and Codex use |
| Template 1: model and alternative trade-off | PASSED | REPORT §1; architecture-only BERT alternative is labeled |
| Template 2: real setup/loss/results/hardware | PASSED | REPORT §2 and linked machine-generated evidence |
| Template 3: no-answer score and threshold | PASSED | REPORT section 3; threshold -0.98309326; calibration-only weighted EM with >=70% impossible calibration floor |
| Template 4: architecture/token/tunnel/run commands | PASSED documentation | REPORT §4 and README; rehearsal still pending |
| Template 5: all public outcomes and failure analysis | PASSED documentation; FAILED 5/5 goal | REPORT section 5 records all current answers and semantic attribution error in sample-04 |
| Template 6: real size and latency | PASSED | REPORT section 6; 66,364,418 parameters, CPU HTTP measurements |
| Template 7: two actual problems and fix commits | PASSED | REPORT section 7 retains 020603a and 0e4c678; continuation protocol fixed before final evaluation |
| Template 8: specific decoder investigation | PASSED | Tied to incorrect numeric spans in samples 03/04; costs/grounding, no invented benchmark |
| Template 9: known limitations | PASSED | Accuracy, question cap, span length, confidence, pending operational gates |
| Template: exact headings 0-9, nonempty, English, ~1000 words | PASSED review | Original heading comparison; Markdown table spacing corrected |
| XIV: 3-minute pitch, four rubric criteria | PASSED materials; rehearsal pending | docs/PITCH.md includes actual quality/latency and limitations |
| XIV: individual Q&A without AI, attendance | PENDING human requirement | docs/DEFENSE.md; all members must prepare and attend |
| XV: reading references | PASSED resources | docs/SOURCES.md; team must understand cited methods |
| Master: no fabricated metrics, no unrequested expansion | PASSED | Only completed run reported; no additional epoch or training launched |

## Required user actions before submission

1. Review the measured accuracy: current public result is 4/5; fresh answerable EM is 33.92 and impossible EM is 73.15. The one authorized continuation is complete. No additional training is authorized automatically.
2. Obtain the instructor token and enter it locally with the secure README prompt; do not send it in chat.
3. Configure the already installed ngrok as backup, then test that backup from another network. The five Cloudflare mobile functional checks passed; a 50-request mobile benchmark was not performed.
4. Start the candidate on CPU, send the active HTTPS URL to the instructor, and perform the official dry run.
5. Choose an authorized location to share weights; the public Git repository contains code/evidence only.
6. Rehearse the pitch and individual Q&A. After all gates pass, review the final report and authorize submission tagging.

Dependency check: `pip check` passed in the existing venv. The instructions are provided for clean setup, but a second clean machine was not independently tested.

Preparation evidence: `logs/continued20k/sample04_diagnosis.json` confirms the selected hash, CPU execution,
full single-window coverage, finite logits and valid offsets. This diagnoses the public failure without fitting
any parameter. `logs/rehearsal_tests.txt` records four mocked runner tests; 47 prior model/API tests and 19 real
HTTP checks are reused unchanged. The new runner enforces per-request and global wall-clock deadlines;
neither a mocked test nor a generated file is proof of real external connectivity. RUNBOOK_ES.md covers secure
token entry, startup, explicitly authorized tunnels, external test, ngrok backup and shutdown.
