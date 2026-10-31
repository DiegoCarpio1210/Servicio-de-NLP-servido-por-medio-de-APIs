# DeBERTa LoRA GPU feasibility: measured results

**A minimal LoRA experiment is technically feasible on this Quadro P1000.**
This establishes forward/backward/optimizer compatibility, not improved QA
accuracy or feasibility at every batch/sequence length. No model was activated,
saved, merged or calibrated. Only six optimizer steps were executed across two
independent model initializations. Stable main remains `02a9d34`.

## Environment and isolation

Windows laptop; Python 3.13.2; PyTorch 2.7.1+cu118 / CUDA 11.8;
Transformers 4.57.1; Accelerate 1.11.0; PEFT 0.17.1. GPU: Quadro P1000,
4,096 MiB, compute capability 6.1, driver 573.22. `nvidia-smi` reported 4,004 MiB
free before execution. After CUDA initialization, Torch reported 3,528,877,671
free bytes; these readings differ in timing and accounting.

PEFT was installed with `--no-deps --target tmp/lora-deps`. It is inserted into
`sys.path` only by the experiment script; stable requirements and environment
packages were not upgraded. The existing local SQuAD directories are available
at `data/squad_v2` and `data/hf`, but no dataset training was run.
The cached dataset loaded successfully: 130,319 training and 11,873 validation
examples. These counts describe availability, not examples consumed by this test.

Precision was FP32. The default Torch BF16 capability check returned true because
it includes emulation; the explicit `including_emulation=False` check returned
false. No claim of native BF16 support or FP16 acceleration is made.

## Configuration and verified mechanics

- Original pinned `deepset/deberta-v3-base-squad2` files; no downloads or writes
  to its checkpoint directory during the test.
- Rank 4, alpha 8, dropout 0.05, bias frozen; 24 actual `query_proj` / `value_proj`
  linear modules across 12 layers, inspected before injection.
- `TaskType.QUESTION_ANS`; `qa_outputs` handled with `modules_to_save`.
- Non-reentrant gradient checkpointing, microbatch 1, seed 20261030.
- AdamW LR 1e-4, weight decay 0, for mechanics only; not an optimized training LR.
- Synthetic, explicitly labeled QA mechanics fixture (answer `cobalt`). It is
  not a public example or any 500/1,024 benchmark question. The shared project
  `training_features` utility constructs valid token-level start/end labels.
- Base parameters: 183,833,090. PEFT-wrapped total including saved-head copy:
  **183,982,084**. Trainable: **148,994 (0.080983%)**, comprising 147,456 LoRA
  parameters plus 1,538 QA-head parameters.

Both runs produced finite start/end logits of the expected shape and finite
losses. All trainable gradients existed and were finite; the QA head and LoRA
had nonzero gradients. On the first step, 24 LoRA tensors had nonzero gradients;
on later steps, all 48 did. Initial zero gradients in A are expected because
PEFT initializes B to zero. See [the pinned PEFT documentation](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora).

Optimizer updates changed adapters and the saved QA head. All frozen base
parameters had no gradients, and their full in-memory digest matched before
and after. The original DeBERTa and quality40k files, tokenizer files, calibration
and active manifest also retained their hashes.

## Measured memory and time

| Setting | 128 tokens, accumulation 1 | 256 tokens, accumulation 8 |
|---|---:|---:|
| Optimizer steps | 3 | 3 |
| Forward/backward microsteps | 3 | 24 |
| Peak allocated bytes | 787,685,376 | 813,071,360 |
| Peak allocated MiB | 751.1953 | 775.4053 |
| Peak reserved bytes | 830,472,192 | 866,123,776 |
| Peak reserved MiB | 792 | 826 |
| Step times, seconds | 0.8394 / 1.0082 / 0.1884 | 2.8450 / 2.3044 / 2.3073 |
| Mean of steps 2–3, seconds | 0.5983 | 2.3059 |
| Mean losses per optimizer step | .041971 / .004191 / .003942 | .049813 / .074070 / .043017 |
| OOM / captured warnings | None / none | None / none |

Timing includes forward/backward, gradient checks, optimizer and CUDA
synchronization, excluding model loading, tokenization and frozen-parameter
hashing. The 128-token timings vary substantially; three steps are not a robust
throughput benchmark. The fixture is short and padded to the specified length;
real questions, nonpadding tokens and overflow windows may change costs.
These losses are not validation quality measurements.

Allocated/reserved figures are PyTorch allocator peaks, not total device usage.
After Python cleanup, 17,039,360 allocated bytes remained in-process; process exit
released the CUDA context, and `nvidia-smi` then reported 0 MiB device usage.
No persistent training process or adapter checkpoint was left behind.

## Failures and limitations

Two setup failures occurred before GPU execution: the sandbox could not read
PEFT files installed by the host process, and the host Git invocation initially
rejected ownership of the working directory. Execution used the authorized host
context and a command-scoped `safe.directory`; no global Git setting or GPU
protection was changed. Neither was an OOM. Raw logs preserve the errors.
An initial unrestricted pytest discovery also traversed the host-installed
dependency directory and hit sandbox permission errors. The complete project
suite is run explicitly with `python -m pytest tests -q`, avoiding discovery in
third-party package directories; the failed collection log is preserved locally.

No 384-token trial, FP16 trial, generalization gain, adapter reload test or long
training stability is claimed. No full-model fine-tuning occurred. The six steps
use one synthetic fixture and cannot establish any accuracy improvement.

## Reproduction

On `experiment/deberta-lora`, with the existing project environment and pinned
local weights, install the isolated dependency once:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps --target tmp/lora-deps -r experiments/requirements-lora.txt
PROBAR_LORA.bat --length 128 --steps 3 --accumulation 1
PROBAR_LORA.bat --length 256 --steps 3 --accumulation 8
```

The script refuses other branches and allows only 1–3 optimizer steps, sequence
lengths 64/128/256 and accumulation 1/2/4/8. It never saves model weights or
activates an adapter. Timestamped raw results, environment, gradient norms,
execution and error logs are ignored under `logs/deberta_lora/`. The compact
published evidence is `logs/deberta_lora_smoke_summary.json`. The launcher runs
in the foreground; no background process is necessary.

## Proposed next experiment — not executed

1. Freeze context-grouped manifests before any fitting. Prefer an independently
   annotated domain set with original source-grounded answers and deliberate
   unanswerable questions. If only SQuAD2 is available, describe the experiment
   as continued adaptation on vendor-seen data, not new generalization evidence.
   Exclude the old 500- and 1,024-question benchmarks and their contexts from
   training/calibration, plus the public samples. Audit all recorded partitions.
2. A bounded SQuAD2 pilot could use at most 1,600 training windows, retaining the
   source answerability mix, with separate context-disjoint development (512)
   and calibration (512) examples if the overlap audit permits. Preserve a
   separately sealed final set; if genuine independence is unavailable, say so
   and obtain new private/instructor-held examples. Do not silently reuse an
   observed benchmark as a fresh final test.
3. Start with the measured 256-token FP32 configuration: microbatch 1,
   accumulation 8, rank 4/alpha 8/dropout .05, query/value and QA head. Use shared
   overflow preprocessing with stride 96 and original SQuAD labels. Proposed LR
   **2e-5**, 10 warmup steps, maximum **200 optimizer steps**; no performance
   guarantee. Compare against the unchanged base on development data.
4. Validate every 50 steps. Stop on nonfinite loss/gradients, OOM, or two checks
   without improvement in predeclared development 75/25 weighted EM. Preserve
   group metrics to expose answerable/null tradeoffs. Never select a configuration
   using the old 500/1,024 final results. Experimental targets: EM >=85%, F1 >=88%,
   answerable EM >=80%, unanswerable EM >=95%; all must be reported independently.
5. Save adapter + QA head every 50 steps, optimizer/scheduler, RNG states,
   sampler position, global step and manifest hashes in a separate ignored
   experiment directory. Resume only with identical base revision, configuration,
   data hashes and dependencies. Do not merge into original weights. A substantive
   training/resume driver still needs implementation and verification; the smoke
   launcher is intentionally not a long-training launcher.
6. After selecting the adapter on development, calibrate once on its separate
   partition, freeze the policy, then evaluate the sealed final set and API/CPU
   latency. No activation without separate approval.

Measured-step extrapolation for 200 steps at accumulation 8 is approximately
**461 seconds (7.7 minutes) of optimizer-loop time**. Loading, data processing,
validation, checkpoint I/O and real-context effects are additional and unmeasured;
this is not a promised total run duration. Measure a short real-data pilot before
committing to a longer schedule. A 16 GB GPU can use the same pinned base and
manifests; first repeat the smoke test there and only then increase sequence
length/batch size. Native supported mixed precision must be checked on that GPU.

## Stable regression checks

The complete existing suite is recorded in `logs/deberta_lora_regression.txt`.
All **65 tests passed in 13.21 seconds** with explicit project-test discovery.
`logs/deberta_lora_api.json` records real CPU loading and authenticated API
inference after GPU cleanup. Production API/UI/default configuration and stable
requirements have no diff against main. No public tunnel, submission tag, full
evaluation rerun or merge into main is part of this experiment.
