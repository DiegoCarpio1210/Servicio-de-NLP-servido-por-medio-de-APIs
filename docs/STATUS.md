# Verification status — Pimientero

Snapshot: 2026-10-25. **The project is NOT ready for submission.** Full-data training is running in the
background; this document reports completed pilot evidence only. No `submission` tag has been created.

| Requirement | Status | Executed evidence / next action |
|---|---|---|
| Read all original materials | PASSED | ZIP integrity, all 12 PDF pages, YAML, template, 5 cases, master brief |
| BASE encoder, local inference, no commercial service | PASSED | DistilBERT base revision recorded; CPU Engine |
| Genuine fine-tuning | PASSED for pilot | 1,024 train examples, two epochs, 130 updates, 325.61 s |
| Full SQuAD 2.0 training | PENDING — RUNNING | 130,319 train / 11,873 validation; `logs/final/job.json` |
| Reproducible data, disjoint official splits | PASSED | `logs/dataset.json`, seed 42, exact class counts |
| Real loss curve and metadata | PASSED for pilot | `logs/pilot/loss.png`, history.json, training_metadata.json |
| Validation-only threshold | PASSED for pilot | threshold 1.94921350; 64 calibration + 64 threshold holdout |
| Competitive answer quality | FAILED for pilot | Holdout answerable F1 0%; public 2/5; requires full training |
| Exact API/schema/authentication/error behavior | PASSED | 36 automated tests, original YAML validation |
| Unicode offsets, null pairing, finite score | PASSED | Span/tokenizer/schema tests and real HTTP |
| Long contexts and final-window coverage | PASSED mechanically | Real tokenizer overflow/labels; long public response processed |
| Correct answer to long public case | FAILED for pilot | Returns null; not truncated, but undertrained |
| Public five-case evaluator | PASSED as tooling | Correctly reports three wrong answers, exits nonzero |
| Local 50-request timing | PASSED for pilot | 18.06 s total, .345 s mean, .930 s maximum |
| HTTPS tunnel, protected info and predictions | PASSED for pilot | Cloudflare round trip; 50 calls, 50 contract-valid |
| HTTPS 50-request timing | PASSED for pilot | 32.21 s total; 0.627 s mean; 1.145 s maximum |
| Second tunnel installed and tested | PENDING | ngrok requires installation/account configuration; instructions in README |
| Another-network test and instructor dry run | PENDING | Phone/mobile tethering and instructor harness |
| Team identity and members | PASSED | `configs/team.json`; production default team Pimientero |
| Repository remote/main/history | PASSED locally | origin set; authentic commits retained; remote verification recorded after push |
| Published GitHub files | PENDING verification | Check remote HEAD and fetch committed README after push |
| No weights, tokens, environment, datasets in Git | PASSED initial review | .gitignore; run audit after staging final documentation |
| Report exact headings 0–9, English, populated | PASSED as draft | REPORT.md; final full-run evidence still pending |
| At least two real issues and fix commits | PASSED | REPORT section 7; commits 8955d78 and 06b6148 |
| Decoder investigation, pitch and Q&A | PASSED as materials | docs/PITCH.md, DEFENSE.md, ARCHITECTURE.md; rehearsal pending |
| Final tag, final report, final service | PENDING | Deliberately withheld until all final evidence is checked |

## Full training operation

`python -m scripts.run_full` was launched with a hidden window. It runs full training, calibration, then a real
50-request CPU test, writing separate logs and exit codes. Check without revealing any token:

```powershell
Get-Content logs/final/job.json
Get-Content logs/final/training_execution.txt -Tail 10
Get-Content logs/final/history_live.json -Tail 20
```

The history file appears at the first logging interval. Checkpoints are saved every 500 optimizer steps and at
validation epochs; two recent/best checkpoints are retained. Do not launch a second full job simultaneously.
Keep the laptop plugged in, awake and ventilated. The pilot suggests an order-of-magnitude runtime of many hours
(roughly 15–20 hours for three full epochs, not a promise). Actual progress estimates are in the training log.
No final metrics, loss curve or quality claim exists until the corresponding stage finishes.

If interrupted, identify the latest real `checkpoints/full/checkpoint-N` and run:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs/train.json --resume checkpoints/full/checkpoint-N
.\.venv\Scripts\python.exe -m scripts.evaluate --model-dir models/final
.\.venv\Scripts\python.exe -m scripts.verify_local --model-dir models/final --repeat 10
```

To stop deliberately, read the exact `child_pid` and `pid` from job.json and stop those specific processes;
resumption starts from the last saved checkpoint. Do not delete model directories or stop unrelated Python tasks.

## Remaining manual inputs and rubric risks

- Set the instructor token locally using the secure README prompt. It has not been requested in chat or committed.
- Complete another-network test and the instructor dry run; send the current URL before each scheduled session.
- Install/test ngrok as backup; Cloudflare temporary URLs stop working when their process stops.
- Replace pilot report metrics with the completed full-run measurements; investigate remaining failures using validation.
- Rehearse all four pitch criteria and individual code questions. Every member must attend; no AI in Q&A.
- Only after all gates pass: commit final evidence, create and push `submission`, share repository and authorized weights location.

The largest rubric risk is accuracy, not latency: the pilot is a null baseline and must not be submitted as final.
The report remains a draft to avoid falsely representing training, the external rehearsal or final validation.
