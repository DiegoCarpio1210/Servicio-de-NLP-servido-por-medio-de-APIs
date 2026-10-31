# DeBERTa LoRA feasibility — preparation only

Stable base: `02a9d34` on `main`. Experimental branch:
`experiment/deberta-lora`. No training, dependency installation, adapter creation
or calibration change has been performed. This branch currently adds only this
report; stable weights and deployment remain unchanged.

## Recommendation

Try a bounded memory/gradient smoke test before considering any longer LoRA run.
A 4 GB P1000 may fit a minimal configuration, but this has **not been measured**.
Prefer an available 16 GB or larger GPU for substantive comparisons. More SQuAD
training does not guarantee improvement: this checkpoint is already SQuAD2
fine-tuned, and additional updates can worsen answerability or forget useful
behavior. The missing accuracy targets must not be described as solved.

## Memory and method comparison

The actual checkpoint contains 183,833,090 parameters. Observed local hardware:
Quadro P1000, 4,096 MiB VRAM, 4,004 MiB free at inspection, driver 573.22.
These are device readings, not a training-memory measurement.

| Method | Static memory estimate, excluding activations/workspaces | Assessment |
|---|---|---|
| Full FP32 AdamW | About 2.74 GiB for weights, gradients and two optimizer states (16 bytes/parameter) | Little headroom on 4 GB; not recommended as the first laptop experiment |
| Typical mixed-precision AdamW | About 3.08 GiB at 18 bytes/parameter | Mixed precision does not automatically solve the memory limit |
| FP32 frozen base + rank-4 query/value LoRA | About 701 MiB base weights; about 0.15M trainable adapter/head parameters, a few MiB of optimizer/gradient storage | Activations, CUDA context and temporary tensors still matter; possible minimal experiment, not a verified fit |

These are arithmetic estimates, not measured total peaks. Actual consumption
depends on sequence length, attention implementation and checkpointing. See
[Transformers memory anatomy](https://huggingface.co/docs/transformers/v4.32.0/model_memory_anatomy)
and [PEFT LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora).
The [NVIDIA P1000 specification](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/productspage/quadro/quadro-desktop/quadro-pascal-p1000-data-sheet-us-nv-704475-r1.pdf)
identifies its Pascal architecture and 4 GB capacity.

## Proposed minimal smoke test, not executed

- Use a separate virtual environment; pin and record compatible PyTorch,
  Transformers, PEFT, CUDA and driver versions before changing dependencies.
- Exact base revision and file hashes: `configs/deberta_deployment.json`.
- Rank 4, alpha 8, dropout 0.05; target `query_proj` and `value_proj`;
  preserve/train the QA head through PEFT `modules_to_save=["qa_outputs"]`.
  Projection names were inspected directly in the local safetensors keys.
- FP32 initially, microbatch 1, maximum sequence length 128, accumulation 8,
  gradient checkpointing. Verify nonzero adapter/head gradients with a frozen
  base; an embeddings/checkpointing interaction must not silently disable them.
- At most 20 optimizer steps for the initial diagnostic; measure peak allocated
  and reserved VRAM and warmed step time. Stop on OOM/nonfinite loss, preserve
  logs, and do not automatically retry a long run. Close the UI/API first.
- Only after that result, estimate duration as warmed seconds per optimizer
  step multiplied by the proposed step count, plus measured evaluation overhead.
  No hour estimate is justified by the old DistilBERT timing.

A 128-token smoke test only checks feasibility; it is not the final long-context
quality protocol. Try 256/384 tokens only after memory evidence supports it.
Use a more capable GPU if realistic contexts leave insufficient headroom.
Do not assume BF16, quantized optimizer support or FP16 speed benefits on P1000.

## Data separation and reproducibility

Before any substantive experiment, freeze context-grouped training, development,
calibration and final-evaluation manifests with IDs, seeds and hashes. Deduplicate
normalized questions and context hashes across partitions. Training uses original
SQuAD annotations; development selects adapters; a separate calibration partition
fits the predeclared operating point; final evaluation is consulted only after
locking both. Keep the existing 500 questions and public examples out of all
fitting and hyperparameter selection. Their previous results are descriptive
baselines, not a fresh blind test.

Both official SQuAD splits have exposure limitations: the vendor trained on train
and selected/evaluated on validation, and local work has reused validation
contexts. A new private, independently annotated or instructor-held set is needed
for a genuinely new external estimate. Do not claim unseen SQuAD data merely
because a new local ID split was made. If only reused SQuAD data is available,
label results accordingly and retain the original frozen baseline comparison.

Save adapters separately under ignored experiment directories; log base revision,
trainable parameter names/count, seed, batch/accumulation, optimizer, learning
rate, precision, versions, manifests and every checkpoint selection decision.
Do not merge adapters into the original pretrained files. Run API/CPU latency
checks on any future candidate before proposing promotion.

## Rollback and branch isolation

No experimental change goes directly to `main`. Compare against stable commit
`02a9d34`, record results on this branch, and request activation separately.
With services stopped, `git switch main` restores stable source. The existing
`scripts.select_deployment` command can select verified DeBERTa or quality40k.
Original weights/calibration are preserved; no submission tag is created.
