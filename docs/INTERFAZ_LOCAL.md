# Local application — Pimientero AI

## Start and stop

Run `INICIAR_PIMIENTERO.bat` from the project directory. It uses `.venv`, verifies
all hashes in `configs/active_model.json`, starts the academic API on port 8000
and the UI gateway on port 8765, and opens http://127.0.0.1:8765 after readiness.
The first model load can take several minutes. Stop with `DETENER_PIMIENTERO.bat`
or Ctrl+C in the launcher window. Do not close the laptop while a benchmark runs.

The current model is quality40k checkpoint-2000, with the previously evaluated
length05 policy and threshold -0.40494251441686124. continued20k is preserved as
backup. The benchmark never replaces production or changes calibration.

## Prerequisites

Use Python 3.13 and the existing `.venv`. For a fresh setup, create the environment
and install `requirements.txt`; provide the local model artifacts separately.
GitHub intentionally excludes weights, datasets, caches and secrets. The model
folder must include weights, tokenizer, configuration, training metadata and
calibration. Expected hashes and calibration contents are recorded in
`configs/active_model.json`. Never change hashes just to bypass a mismatch.

## Interface behavior

The existing white/black/gray design is retained. Add an English context and ask
English questions. New conversation clears the current context; session history
can restore previous requests and responses. History is kept in memory, not sent
to an external AI service. Copy answer uses the browser clipboard permission.
Light/dark mode and responsive navigation retain their original behavior.

The dashboard reads verified quality40k metrics: EM 61.82%, F1 65.53%, answerable
EM 48.05%, unanswerable EM 75.59%, and weighted EM 54.93%. The active-model section
reports the actual API checkpoint. Scores are not calibrated probabilities of
correctness. A valid offset does not guarantee that the answer is semantically
correct. The model can return null and still makes unsupported-answer errors.

## Architecture and security

`frontend/index.html`, `frontend/style.css` and `frontend/app.js` require no build,
CDN or JavaScript package installation. `qa/ui.py` is the loopback-only gateway;
`qa/api.py` is the academic API. `scripts/local_app.py` owns startup and shutdown,
prevents duplicate launches, and refuses occupied ports without killing unrelated
processes. The launcher generates a temporary token in server memory. The browser
never receives the bearer token. Exact authentication, host allowlisting, origin
checks, CSP and input limits remain in place. Do not expose port 8765 publicly.

## Validation and troubleshooting

Run `.venv\Scripts\python.exe -m pytest -q`. With the application running, use
`.venv\Scripts\python.exe -m scripts.verify_ui --output tmp/ui-check.json` for real
API/gateway inference checks. It verifies the selected hash and answer/null,
Unicode and long-context behavior. This is not a browser visual test.

If disconnected, wait for readiness and retry. Local logs are in
`tmp/local-app/8000.log` and `8765.log`, excluded from Git. Input over the contract
limits is rejected rather than silently truncated. Missing model files require
restoring the approved artifacts; startup never downloads or trains a model.

Historical screenshots under `docs/screenshots/` preserve the pre-English UI test
record. They are not screenshots of this localization. Browser automation timed
out during the preceding activation check; only the recorded gateway checks are
claimed for that run. See `logs/quality20261029/activation_ui.json`.
