# Current status — Pimientero

**Review completed; submission is NOT ready. No further training and no submission tag were created.**
The requirement-by-requirement audit is [COMPLIANCE.md](COMPLIANCE.md).

| Item | Verified result |
|---|---|
| Training | 12,000 examples, 4,005 impossible, two epochs, 1,522 steps, 26m22s |
| Selected checkpoint | checkpoint-1522; exported weights match SHA-256; pilot hash differs |
| Holdout | 1,024 independent examples; EM 52.34%, F1 53.40% |
| Answerable subgroup | 511 examples; EM 25.05%, F1 27.17% |
| Unanswerable subgroup | 513 examples; EM/F1 79.53% |
| Calibration | 512 separate examples; threshold -1.88344193 |
| Evaluation verification | Recomputed all metrics and calibration from recorded candidates; checked every reference/offset and 32 fresh forward matches |
| Public cases | 3/5; 01, 02 and 05 correct; 03/04 incorrectly produce spans instead of null |
| Local performance | 50 sequential requests in 12.51 s; max .559 s |
| HTTPS performance | 50 requests in 20.47 s; max 0.722 s; same host only |
| Technical tests | 42 pytest tests passed; 19 real-model HTTP checks passed; pip check passed |
| Model served | models/subset12k, CPU, one worker, hash and step count exposed in /info |

The public-set failures are genuine quality failures, not contract failures. This candidate is preserved,
but its answerable performance does not justify promising a high grade. An additional epoch would be roughly
13 minutes based on measured time; it was NOT started. Any further experiment requires a separate decision.

## Remaining user actions

- Obtain the instructor token and enter it locally through the secure README prompt.
- Install/test ngrok as backup and test the candidate from another network.
- Arrange the instructor dry run, start service+tunnel and send the active URL before the slot.
- Choose an authorized separate destination for weights (never normal Git commits).
- Rehearse the English pitch and individual Q&A; all members must understand and attend.
- Resolve quality/operational gates and authorize the final submission tag later.

The temporary test API and tunnel were stopped. No training process is needed to serve the saved model.
To run it: set TEAM_ID=Pimientero, MODEL_DIR=models/subset12k, QA_API_TOKEN securely, then use the README uvicorn command.
The old full-data run remains stopped, with historical evidence retained in docs/history/full_run_status.md.
