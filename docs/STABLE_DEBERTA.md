# Approved stable baseline

The user approved DeBERTa-v3-base-SQuAD2 as the default after reviewing the
500-question results. This is a pragmatic baseline decision, not a declaration
that every requested accuracy target passed. Overall EM 83.00%, F1 86.69%,
answerable EM 77.60%, unanswerable EM 88.40%; only overall F1 meets its target.
See [the complete study](FINAL_500_EVALUATION.md) for overlap and uncertainty.

## Identity and recovery

- Source: [deepset/deberta-v3-base-squad2](https://huggingface.co/deepset/deberta-v3-base-squad2).
- Exact revision: `eea39c60cc305c2e4a9504f5ff117294bebb42db`.
- License: CC BY 4.0; retain attribution to deepset and the model card.
- Weight SHA-256: `aff8759009b728d7d35c1163cd3578b911a3e3e1fbd2522c2b4af398e523f92a`.
- `configs/deberta_deployment.json` pins weights, config and all tokenizer files.
- Frozen threshold: `2.7772940397262573`; max answer length 30; length penalty 0.
- CPU inference: two threads, 384-token windows, stride 128.
- No local training steps are claimed for this downloaded QA checkpoint. `/info`
  reports zero local steps, its real 183,833,090 parameters and weight hash.

Install `requirements.txt` in the project environment, then run:

```powershell
.\.venv\Scripts\python.exe -m scripts.restore_deberta
```

This explicit command downloads only the pinned public files and verifies all
SHA-256 values. It does not download a moving `main` revision or train anything.
The existing cached files have been hash-verified; a redundant full download was
not required. No credentials are needed for this public checkpoint.

Start `INICIAR_PIMIENTERO.bat`; open `http://127.0.0.1:8765`.
Stop with `DETENER_PIMIENTERO.bat`. Both servers bind only to loopback.
The browser never receives the API bearer token. For direct API startup, set
`QA_API_TOKEN` locally and run `python -m uvicorn qa.api:app --host 127.0.0.1 --port 8000`.
Unset an old `MODEL_DIR` override to use the default manifest.

## Reversible activation

With services stopped:

```powershell
.\.venv\Scripts\python.exe -m scripts.select_deployment quality40k
# Restart the launcher. To restore DeBERTa later:
.\.venv\Scripts\python.exe -m scripts.select_deployment deberta
```

Selection verifies file hashes before atomically replacing the active manifest.
No checkpoint, calibration or original tokenizer is rewritten. quality40k and
continued20k files remain on disk. Dashboard metrics follow the selected manifest
and identify the balanced 500-question validation cohort, including its prior
context exposure. Switching configuration is a local Git change until committed.

## Verification evidence

- `logs/final500_summary.json`: all three frozen model evaluations and load tests.
- `logs/stable_deberta_verification.json`: production Engine, real local HTTP,
  exact contract/authentication, null/Unicode/long-context offsets and 50 requests
  per sequential and bounded-concurrent mode (two clients).
- `logs/stable_deberta_pytest.txt`: complete regression suite.
- `logs/stable_deberta_ui.json`: local gateway inference and dashboard checks.

Measured production results: 65 regression tests passed. Sequential HTTP: 50/50,
75.56 seconds total, maximum 2.828 seconds. Two-client concurrent HTTP: 50/50,
57.57 seconds total, maximum 3.283 seconds. Both meet 150-second total and
5-second per-request limits. The UI returned `Paris` normally and ` Paris` after
an emoji; both have correct offsets and normalized SQuAD EM, while the latter
does not match the expected text byte-for-byte. No extraction rule was changed.

The tokenizer and model forward pass share one lock. The earlier concurrent
`Already borrowed` tokenizer failure is covered by regression tests and actual
bounded-concurrency HTTP checks. Public samples are load inputs only, never
training or calibration data. No public tunnel or submission tag is created.

## Remaining requirements

Three requested accuracy targets remain unmet. This model has not received an
instructor-run official evaluation or a new separate-network rehearsal in this
activation task. Historical network checks on other models are not evidence for
this deployment. Final delivery, any official live URL/token and submission tag
remain separate actions requiring the appropriate user instruction.
