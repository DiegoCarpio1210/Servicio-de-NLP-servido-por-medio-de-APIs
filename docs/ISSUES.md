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
