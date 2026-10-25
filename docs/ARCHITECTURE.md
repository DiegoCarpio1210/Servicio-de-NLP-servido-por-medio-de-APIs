# Architecture and engineering decisions

The service receives an English paragraph and a question. It must select characters already in the paragraph;
it cannot synthesize an explanation. The implementation has two separate lifecycles.

## Offline training

`qa.data.load_splits` loads official SQuAD 2.0 splits. The seeded subset separately samples answerable and
unanswerable rows, preserving their proportions. `qa.features.training_features` expands each example into
overlapping windows. Sequence IDs distinguish question tokens from context tokens. Character offsets map the
first reference answer to start/end token labels. Missing answers and windows not containing the entire answer
are supervised at CLS. `scripts.train` initializes a fresh QA head on the BASE encoder and trains all parameters.
The Trainer writes resumable optimizer checkpoints, then saves the best validation-loss model and tokenizer.
Training metadata records the exact sample counts, seed, hardware, elapsed time and optimizer-step count.

`evaluate` uses model logits, not a generic question-answering pipeline. It splits validation into threshold
calibration and threshold holdout sets. Checkpoint selection already used validation loss, so the holdout is
not a completely untouched research test set. Public samples are excluded from every learning decision.

## Online prediction

```mermaid
flowchart LR
  Client[Professor HTTPS client] --> Tunnel[Tunnel to laptop port 8000]
  Tunnel --> Auth[Exact Bearer comparison]
  Auth --> Validate[Strict request schema]
  Validate --> Windows[Question cap and all context windows]
  Windows --> Encoder[One CPU encoder in eval mode]
  Encoder --> Span[Best valid span and minimum CLS score]
  Span --> Threshold[Validation threshold]
  Threshold --> JSON[Original substring or paired null]
```

`create_app` starts loading in a background thread. Readiness stays 503 until the calibrated local model has
loaded and warmed up. `Engine` is constructed once and forces CPU. A lock serializes model execution;
`torch.inference_mode` suppresses gradients. One uvicorn worker prevents duplicate model memory.

Questions are limited internally to 64 tokens to leave room for context and the 128-token overlap. This can
lose information from unusually long questions; the API still accepts the contracted 500 characters. Every
context window is processed, including the final window. The batch size of eight windows bounds per-forward
memory; unusually fragmented 5000-character text may still be slower than ordinary English.

For each window, CLS start+end logits represent no-answer. The global null score is the minimum across windows:
an irrelevant window must not veto a supported answer elsewhere. The best valid context span is selected across
windows. Only start<=end spans of at most 30 tokens are considered. The difference `null_score-best_span_score`
is compared with a validation-selected threshold; larger differences favor abstention. This standard aggregation
is still susceptible to window-count effects, so calibration must use the same window settings as serving.
The score is a sigmoid of distance to the threshold, a ranking confidence rather than a calibrated probability.

The answer is sliced from the original Unicode string. It is never reconstructed from WordPiece tokens.
The final invariant is `context[start:start+len(answer)] == answer`. This preserves repeated-answer locations,
newlines and non-ASCII characters. Error handlers omit input values and exceptions from public responses.

## Model trade-off

DistilBERT has six Transformer layers versus twelve in BERT-base; both use 768-dimensional hidden states.
Fewer layers make DistilBERT a sensible CPU candidate. Actual QA parameters are counted from the loaded model,
not copied from a model card. BERT-base is a documented architectural alternative, not an empirically trained
competitor in this run. Train it with a separate config, its own base revision and output directory if resources
permit; compare the same validation split and CPU workload. Do not claim BERT's accuracy or latency without that run.
