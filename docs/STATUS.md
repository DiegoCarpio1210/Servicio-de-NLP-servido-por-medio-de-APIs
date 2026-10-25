# Current verification status — Pimientero

The original full-data job was stopped at the user's request. Its logs, pilot models and test results remain
intact. No periodic full-data checkpoint existed at shutdown; no in-memory updates are claimed to be saved.
The previous status is preserved in `docs/history/full_run_status.md` as a historical snapshot.

## Authorized replacement run

- Exactly 12,000 reproducibly sampled SQuAD 2.0 training examples; 4,005 unanswerable (33.375%).
- Two epochs, seed 42, same DistilBERT BASE encoder. New directories: models/subset12k and checkpoints/subset12k.
- Batch configuration: batch 16, accumulation 1, FP32, dynamic padding grouped by length.
- Validation: 512 checkpoint selection, 512 calibration, 1,024 independent holdout; no shared IDs.
- Pre-run estimate: 56–81 minutes including evaluation; benchmark and actual window counts saved under logs/subset12k.
- State: RUNNING. Read logs/subset12k/job.json and its current_log path for the current stage.
- Checkpoints every 100 optimizer steps; evaluation every 400 and at the final step; loss logs every 25.
- No automatic extra epoch or expansion. No submission tag.

## Verified and pending gates

| Requirement | Status | Evidence |
|---|---|---|
| Previous full-data training stopped | PASSED | Process-specific stop and GPU returned to 0 MiB / 0% usage |
| Reproducible 12k representative subset | PASSED | estimate.json and data_manifest.json |
| Disjoint model-selection/calibration/holdout examples | PASSED | Partition tests and recorded IDs |
| Single concurrent training | PASSED | Exclusive cross-process lock and pre-launch process inspection |
| GPU batch/memory probes | PASSED | Batch 4/8/16; FP16/FP32; selected FP32 batch 16 reserved 2,802 MiB |
| New training implementation | PASSED preflight | Real two-step strategy smoke train; CUDA; dynamic padding; checkpoint/evaluation |
| Unit, API and partition/control tests | PASSED | logs/pytest_strategy.txt; 41 tests at initial strategy verification |
| Full two-epoch 12k run | PENDING — RUNNING | logs/subset12k/job.json |
| Independent EM/F1 and subgroup accuracy | PENDING | Generated only after training by scripts.evaluate |
| Candidate CPU latency/public five cases | PENDING | Automatic real HTTP evaluation after calibration |
| HTTP contract/security/Unicode/window handling | PASSED earlier | Tests and pilot real HTTP evidence remain unchanged |
| Prior pilot local/HTTPS latency | PASSED historical | 18.06 s local / 32.21 s HTTPS for 50 requests; not candidate measurements |
| GitHub history and team details | PASSED | Public main; Pimientero and four registered members |
| New candidate HTTPS/external-network dry run | PENDING | Requires rehearsal after candidate selection |
| Backup ngrok and instructor token | PENDING | User-local setup; never send tokens in chat |
| Final report and submission | PENDING | Report refreshes from real outputs; no final tag until review |

## Control and recovery

Read the live stage:

```powershell
Get-Content logs/subset12k/job.json
Get-Content logs/subset12k/history_live.json -Tail 20
```

Request a safe stop (saves resumable state after the current optimizer step, then evaluates and exits):

```powershell
New-Item -ItemType File -Force logs/subset12k/STOP
```

Wait until job.json reports stopped_with_checkpoint before restarting. Remove only that acknowledged STOP
file, inspect the available checkpoint directories, and resume the chosen actual checkpoint:

```powershell
Remove-Item -LiteralPath logs/subset12k/STOP
.\.venv\Scripts\python.exe -m scripts.run_subset --resume checkpoints/subset12k/checkpoint-N
```

The runner temporarily prevents idle system sleep while active and restores normal behavior on exit. Keep the
laptop plugged in and ventilated; lid closure, manual sleep, shutdown or battery loss can still interrupt it.
The pipeline writes EM/F1, subgroup EM/F1, threshold, latency and public results, then updates REPORT.md using
only those outputs. It makes a validation-based recommendation but never trains more automatically.
