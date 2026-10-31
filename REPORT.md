# Project Report — Extractive QA as a Service

Measured final-development report; official submission remains pending. Active model: original DeBERTa, with quality40k retained as backup.

## 0. Team

| Field | Value |
|---|---|
| Team id | Pimientero |
| Members | Diego Roberto Carpio Flamenco; Julio Roberto Guardado Quijano; Alfredo Enmanuel Cortez Martínez; Iván Daniel Salguero Esperanza |
| Repository URL | https://github.com/DiegoCarpio1210/Servicio-de-NLP-servido-por-medio-de-APIs |
| `submission` commit hash | PENDING explicit tag authorization |
| Public service URL used at the dry run | PENDING official current-model dry run |
| AI tools used (which, and for what) | OpenAI Codex: code, tests, experiment orchestration, local semantic review and documentation; no commercial inference in the service |

## 1. Model choice and justification

Production uses `deepset/deberta-v3-base-squad2`, revision `eea39c60cc305c2e4a9504f5ff117294bebb42db`, with 183,833,090 parameters. The user reports instructor permission for pretrained QA checkpoints; this exception must be confirmed because the original brief prohibits them. We do not represent vendor training as our own.

On the same 500 SQuAD2 questions, DeBERTa achieved EM/F1 83.00/86.69, RoBERTa 78.60/81.68 and our DistilBERT quality40k 58.40/61.06. DeBERTa costs more CPU time and memory but improves measured accuracy. Only its overall F1 meets all four requested targets individually; three remain unmet. See [benchmark and uncertainty](docs/FINAL_500_EVALUATION.md).

## 2. Training

Our own fine-tuning started from `distilbert/distilbert-base-uncased`, revision `12040accade4e8a0f71eabdb258fecc2e7e948be`. These completed runs remain reproducible historical evidence, not training of the active DeBERTa.

| Run | SQuAD2 train examples | Epochs | Run / selected steps | Time |
|---|---:|---:|---:|---:|
| subset12k | 12,000 | 2 | 1,522 / 1,522 | 26m22s |
| continued20k | 20,000 | 2 | 2,534 / 1,600 | 42m59s |
| quality40k | 40,000 | 1 | 2,531 / 2,000 | 43m11s |

| quality40k setting | Value |
|---|---|
| LR / schedule / warmup | 2e-5 / linear / 5% |
| Batch / accumulation / precision | 16 / 1 / FP32 |
| Sequence / stride / seed | 384 / 128 / 42 |
| Hardware | NVIDIA Quadro P1000, 4 GiB |
| Selection / calibration / final counts | 512 / 512 / 1,024 |

[Loss curve](logs/quality40k/loss.png) and [training metadata](logs/quality40k/training_metadata.json) record genuine training. quality40k's reserved-context final EM/F1 was 61.82/65.53, answerable EM 48.05 and impossible EM 75.59. These differ from the later 500-question cohort. LoRA research did not establish an activation advantage; no adapter is active. DeBERTa reports zero local optimizer steps.

## 3. No-answer threshold

Return null when minimum-window CLS score minus best eligible span score exceeds **2.7772940397262573**. Span score is start plus end logit, maximum 30 tokens, no length penalty. The frozen weighted-75/25-EM configuration preceded the 500-question and synthetic final evaluations; neither selected a new threshold. Lower thresholds increase abstention; higher ones risk unsupported answers. Confidence reflects threshold distance, not calibrated correctness probability. Experimental thresholds remain isolated.

## 4. Serving architecture

FastAPI/uvicorn uses one worker and two CPU threads. Background loading verifies local weights/tokenizer hashes before readiness. Exact Bearer authentication reads `QA_API_TOKEN`; malformed input returns contract errors. Sliding windows handle long contexts, and tokenization plus forward inference share a lock. Original Unicode character offsets identify extractive answers. The local UI gateway on 8765 keeps the API token out of the browser.

For a clean machine: create `.venv`, install `requirements.txt`, run `python -m scripts.restore_deberta`, then `INICIAR_PIMIENTERO.bat`. Stop with `DETENER_PIMIENTERO.bat`. Direct serving uses `python -m uvicorn qa.api:app --host 127.0.0.1 --port 8000 --workers 1` after secure local token entry. Authorized HTTPS uses `tools\cloudflared.exe tunnel --url http://127.0.0.1:8000`; ngrok is the backup. [Exact secure instructions](docs/PIMIENTERO_FINAL_HANDOFF.md) separate local startup from public exposure.

## 5. Results on the public sample set

Saved production HTTP evidence was inspected, not rerun. Quotes preserve leading spaces; offsets refer to those exact spans.

| Item | Expected | DeBERTa answer | start | Correct? |
|---|---|---|---:|---|
| sample-01 | 16 | ` 16` | 382 | Normalized yes |
| sample-02 | 3.5 days | ` 3.5 days` | 436 | Normalized yes |
| sample-03 | null | null | null | Yes |
| sample-04 | null | ` 3 times` | 309 | No |
| sample-05 | 50,257 | ` 50,257` | 3156 | Normalized yes |

Sample-04 remains an unsupported non-null extraction from a distractor-rich context. Correct substring offsets do not imply factual correctness. No public-question exception was introduced; official public-sample acceptance remains unresolved. Compact evidence: [saved responses](logs/final_public_samples_audit.json).

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters / weights file | 183,833,090 / 735,360,940 bytes |
| Laptop | Lenovo P52s; Intel family 6/model 158; about 31.72 GiB RAM |
| Final synthetic 50 sequential CPU HTTP | 47.10s total, .942s mean, 2.317s maximum |
| Same 50, two concurrent clients | 46.18s total, 1.831s mean, 3.032s maximum |

All 100 requests passed contract/time checks. HTTPS round-trip limits still need a current external rehearsal. Historical 500-question inference peak RAM was 1268.3 MiB; these are observed runs, not guarantees.

## 7. Problems encountered and how you fixed them

1. Full training projected sixteen hours. Seeded subsets and measured batches reduced training time; commit `020603a`.
2. Concurrent tokenizer calls raised “Already borrowed”. Serializing tokenization with inference resolved the reproduced failure; commit `bb9cacd`, followed by real bounded-concurrency tests.
3. Exact matching penalized supported short answers. We retained extraction and distinguished lexical metrics from a clearly labeled local semantic review instead of tuning on final questions.

## 8. Decoder investigation

A decoder could express a complete answer from noncontiguous evidence or explain why sample-04 is unsupported. That does not guarantee truth: it may hallucinate and would violate verbatim extraction without grounding to a span/null. Autoregressive generation adds sequential latency and memory. No decoder was trained or benchmarked; the local encoder remains the service.

## 9. Known limitations

All 500 benchmark contexts had prior local exposure. The separate synthetic final cohort scored EM 40/50, F1 95.29%; local review supports 49/50 with one ambiguous omission. This is not Claude, independent grading or an instructor score. English questions are capped at 64 tokens; long-question quality is not established. Three historical accuracy targets, sample-04, checkpoint eligibility confirmation, current HTTPS/dry run, defense and final submission fields remain pending. No submission tag or public tunnel was created. See [final handoff](docs/PIMIENTERO_FINAL_HANDOFF.md).
