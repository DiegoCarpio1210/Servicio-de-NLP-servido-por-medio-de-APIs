# Current status — Pimientero

One continuation completed; no additional run or submission tag. Candidate: **continued**, `models/continued20k`.
Selection was fixed on calibration before either final holdout was evaluated. All metrics below use the fresh 1,024.

| Profile | Overall EM | Overall F1 | Answerable EM | Answerable F1 | Impossible EM | 75/25 EM | 75/25 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 52.25 | 53.50 | 22.35 | 24.87 | 81.91 | 37.24 | 39.13 |
| recalibrated | 37.70 | 41.94 | 47.65 | 56.17 | 27.82 | 42.69 | 49.08 |
| continued | 53.61 | 56.18 | 33.92 | 39.08 | 73.15 | 43.73 | 47.60 |

Public accuracy: 4/5. Fifty local CPU requests: 14.08s; maximum 0.632s.
Training: original 26m22s plus continuation 42.98 minutes. Original weights are preserved.
See CONTINUATION_EXPERIMENT.md, logs/continued20k/comparison.json and COMPLIANCE.md.

Pending: instructor token for the eventual official session, another-network test, ngrok account setup/backup tunnel test,
official dry run and active URL, authorized separate weight distribution, pitch/Q&A rehearsal and final review.
The API test process was stopped after verification. Start it with the README commands and the candidate model path.
No further training is authorized automatically. Submission is not declared complete.

Preparation update: sample-04 was inspected once on CPU without changing weights/settings. Its entire context
is present; a valid but semantically unsupported span outranks CLS. No general inference bug was demonstrated.
See SAMPLE04_DIAGNOSIS.md. The original 47 tests and 19 real HTTP checks remain valid; four new mocked tests
cover the rehearsal runner. This is not evidence of actual HTTPS or another-network execution.
RUNBOOK_ES.md provides the team's operating steps; DEFENSE.md includes a plain-language explanation and
four-member practice rotation. No public tunnel was opened during that preparation phase; the later authorized test is recorded below. No additional training/evaluation sweep occurred.

## Latest connectivity verification (supersedes earlier pending same-host HTTPS status)

Authorized temporary Cloudflare HTTPS with the selected model and a random test token: 50/50 contract-valid
responses, 19.252s total, .755s maximum, .385s mean. Public accuracy remains 4/5 (40/50 repeated inputs).
No technical exceptions occurred. API and tunnel were stopped; three URL probes returned HTTP 530.
Weights and calibration hashes are unchanged. Evidence: logs/continued20k/tunnel-summary.json and its linked
rehearsal/lifecycle files. This is from the origin laptop, not a mobile or separate-network test.
ngrok 3.39.11 is installed from Microsoft Store and version-checked. Configuration file is absent: user account
authtoken setup and backup tunnel testing remain pending. No instructor token was requested or used.
PHONE_TEST_ES.md explains the pending phone test. Official dry run, public sample-04, weight sharing and
submission approval remain open. No submission tag was created.

## Mobile connectivity completed — 2026-10-29

Five manual Cloudflare checks passed from a Samsung Galaxy S25 Ultra on mobile data, with the laptop on independent Wi-Fi: health, info without token, info with wrong token, authenticated info, and authenticated POST predict. The last returned Paris/start 0/score 0.8257633643007211 and server HTTP 200. Evidence: logs/continued20k/mobile_connectivity.json. Device/network and response bodies were reported by the user; server status logs corroborate them. No mobile 50-request timing is claimed. The restored API/tunnel were stopped, HTTPS returned 530, the in-memory test token is no longer usable, and model/calibration hashes are unchanged. This supersedes earlier pending phone-test status. Official dry run, ngrok backup test, sample-04 failure, separate weight delivery, presentation and final submission fields remain open.

## Ngrok backup attempt — 2026-10-29
CPU model loaded, but ngrok rejected the locally configured credential with ERR_NGROK_105 before any HTTPS functional checks. Configuration syntax validation passed; account authentication did not. User must replace the authtoken locally. API and tunnel processes stopped; two public probes returned 404 and local API was unavailable. Model/calibration unchanged. Evidence: logs/continued20k/ngrok-20261029T185458Z/evidence.json. Backup remains pending.


## Ngrok backup verified — 2026-10-29
The replacement authtoken was accepted by ngrok. Four real public HTTPS checks passed from the origin laptop: health 200, info without token 401, authenticated info 200, predict 200 (Paris, start 0, score 0.8257633643007211; 0.304 s). Selected checkpoint-1600 and calibration verified. API and tunnel stopped; two closure probes returned 404 and local API was unavailable. Weights/calibration unchanged. No ngrok mobile test or 50-request benchmark is claimed. Evidence: logs/continued20k/ngrok-20261029T190027Z/evidence.json. This supersedes the failed first attempt and earlier pending backup status. No credentials were read or included in evidence.
