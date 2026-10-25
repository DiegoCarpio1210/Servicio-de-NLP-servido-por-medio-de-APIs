# Observed issues and fixes

1. Cached-data runs still attempted Hub HEAD requests and failed with Windows socket error 10013.
   Reproduced in smoke evaluation and the first pilot launch. The smoke training had already downloaded the files.
   Fix: explicit offline mode (`scripts/offline.ps1`) and a persistent dataset snapshot (`scripts/prepare_data.py`, `qa/data.py`).
   Successful offline smoke evaluation is recorded in `logs/smoke/evaluation.json`.
   Fix commit: `06b6148`.
2. The first PDF extraction wrote the text successfully but printing it failed with UnicodeEncodeError:
   Windows cp1252 cannot encode the approximation symbol in the professor's guide.
   Fix: read the saved UTF-8 extraction with PowerShell `Get-Content -Encoding UTF8`; all 12 pages were read.
   Evidence: preserved `originals/guide_extracted.txt`, initial source commit `8955d78`.
3. Git writes were denied inside the sandbox; an elevated attempt then reported ownership mismatch.
   Fix: a command-scoped `safe.directory` for this verified workspace; no global trust wildcard.
   First real commit: `8955d78`.
4. Smoke calibration predicts null on all held-out examples (EM/F1 50%, answerable F1 0%).
   Cause: eight optimizer updates only test mechanics; the model is undertrained.
   Action: real two-epoch pilot and full-data configuration; quality remains an open gate until measured.
5. Audit found that reusing MODEL_DIR could retain an earlier checkpoint's calibration.
   Fix: invalidate calibration after saving newly trained weights; serving refuses uncalibrated artifacts.
   This was a code-path finding, not a claimed observed production outage.

## 12k strategy change

- The full-data process had no control-file listener and had not reached its first scheduled checkpoint.
  Only its verified process tree was stopped; logs were retained. The new `EvidenceCallback` handles a STOP
  sentinel at an optimizer boundary and requests checkpoint + evaluation before exiting with code 75.
- The first strategy smoke run failed while writing `data_manifest.json`: datasets 4.3 returns Column objects,
  which Python JSON does not serialize directly. Converting ID columns to lists fixed it; the real smoke run
  and separate calibration/evaluation then completed. No failed run was presented as successful training.
- A Windows source write briefly used cp1252 for an em dash, failing compileall. The report generator was
  rewritten explicitly as UTF-8 and compilation was rerun. This did not interrupt the training process.
- The original threshold holdout had participated in checkpoint selection. The replacement uses disjoint
  selection/calibration/holdout IDs, with a regression test and a recorded manifest.
