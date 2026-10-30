# Autonomous pretrained QA benchmark

The user reports that the professor now allows pretrained QA models. This is a
user-supplied clarification, not a modification of the original PDF/YAML. Current
production remains quality40k checkpoint-2000; no training, automatic promotion,
public tunnel or submission tag is part of this experiment.

## Verified sources and attribution

Both candidates are published by Deepset under **CC BY 4.0**. The exact revisions,
model-card links and published metrics are in `configs/pretrained_benchmark.json`.

| Model | Published SQuAD2 validation EM / F1 | Source |
|---|---:|---|
| deepset/deberta-v3-base-squad2 | 83.8248 / 87.4100 | [Deepset model card](https://huggingface.co/deepset/deberta-v3-base-squad2) |
| deepset/roberta-base-squad2 | 79.9309 / 82.9501 | [Deepset model card](https://huggingface.co/deepset/roberta-base-squad2) |

These are vendor-published results, not local measurements or guaranteed targets.
Availability and license metadata were checked through the Hub API before model
downloads. The primary candidate is downloaded first; only these two external
models are included. quality40k is also measured as the unchanged local reference.
Model cards/license files are retained with downloaded artifacts for attribution.

## Launch, progress and restart

Run `RUN_PRETRAINED_BENCHMARK.bat` from this repository. It uses the existing
`.venv`, launches an independent hidden Python process, and returns immediately.
The exclusive benchmark lock prevents competing benchmark workers. No Codex
message is required after launch. Keep Windows awake and connected to the internet
for downloads; inference runs locally. Logging does not require Codex credits.

Persistent outputs are under `logs/pretrained_benchmark/`:

- `pid.json`: actual orchestrator PID and UTC start time.
- `progress.json`: orchestrator stage; candidate `progress.json` files show row progress.
- `execution.log`: persistent stage log; `errors.log`: failures and tracebacks.
- `<candidate>/execution.log`: child stdout/stderr, including download failures.
- `<candidate>/download.json`: completed revision/hash verification marker.
- `<candidate>/calibration_predictions.json` and `final_predictions.json`: resumable row checkpoints.
- `<candidate>/policy.json`: frozen thresholds before final scoring.
- `<candidate>/metrics.json` and `http.json`: measured accuracy/resources/HTTP results.
- Root `metrics.json` and `comparison.json`: consolidated results after all stages.

Restart the same launcher after an interruption. Completed model results are
skipped; rows resume from atomic checkpoints every 16 examples. Downloads reuse
the pinned local cache. Each candidate has one download call per launch; library
network retries are finite. A failed candidate is recorded and the next is tried.
Do not change the frozen configuration or manifest halfway through an experiment.

The orchestrator checks dependencies and permits one CPU worker at a time, two
Torch threads, no GPU. Each worker is limited to three hours and 8 GiB process-tree
RSS, with at least 4 GiB available at startup and 2 GiB while running. At least
6 GiB free storage is required. A resource violation stops only the owned worker
and records its failure. Models and the active calibration are never modified.
If the operating system shuts down, the process cannot continue until restarted.

## Evaluation protocol and limitations

Do **not** use the previous official-training holdout to compare these candidates:
they were pretrained on SQuAD2 training examples. Instead, the pipeline freezes
512 calibration and 1,024 final examples from official SQuAD2 validation, balanced
within each partition and separated by complete paragraph context. All compared
models use exactly the same final IDs. The manifest records the fingerprint.

Every official-validation context was already consulted during earlier project
work, and vendor-reported evaluation also uses validation. quality40k's previous
selection/calibration can overlap these historically used contexts. The abandoned
full-data run still has uncertain consumed IDs. This is a transparent reused-
validation comparison, **not an absolutely unseen or official instructor test**.
No threshold or policy is selected from the new final results.

External candidates use the same 384/128 overflow tokenizer settings, maximum
30-token span and minimum-window CLS null aggregation. Their primary threshold
maximizes calibration EM weighted 75/25, with F1 tie-break. A secondary operating
point targets at least 95% null-class accuracy **on calibration only**. Both are
frozen before final scoring; the secondary result is not substituted silently if
it looks better on final data. quality40k retains its approved policy and threshold.
Parameter settings are not chosen using public examples or manual ISS questions.

Metrics include overall/group EM/F1, weighted EM/F1, abstention rate and binary
answerability accuracy. Null-class EM is explicitly identified rather than
confused with overall abstention frequency. Single-run CPU latency and sampled
RAM are descriptive; the production launcher normally uses four threads, whereas
this comparison gives every model two threads to bound resource use.

## HTTP checks and interpretation

After scoring, the same academic API factory exposes the loaded worker only on
127.0.0.1:8000 with a fresh in-memory random token. It checks authentication,
schemas, invalid requests, Unicode and long contexts, then 50 sequential and
50 bounded-concurrency requests (two concurrent clients). Each call has a five-
second timeout; results record whether all requests and the 150-second aggregate
budget passed. Five public inputs repeat ten times; this is a performance/contract
exercise, not 50 independent questions. If port 8000 is occupied, the stage fails
safely without stopping the user's application. The temporary server is closed
before the worker exits. No public exposure occurs.

The summary ranks primary-profile weighted EM and retains failed target/latency
checks. Review both answerable quality and false answers before proposing a
replacement. The requested 85–95% EM/F1 and 95% impossible accuracy are goals,
not promised results. Only the user may authorize a production replacement.

All live benchmark outputs, weights and caches are gitignored. Validated code,
tests, attribution and the protocol are published. Summaries can be reviewed and
published later after checking completion; a running job is not a completed result.
