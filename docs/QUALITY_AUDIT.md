# Quality audit and controlled experiment

## Stable system and diagnostic evidence

Production remains `models/continued20k`, checkpoint-1600. Neither its weights nor
its threshold is changed by the experimental scripts. The GUI work was published
first in commit `35fa732`; the experiment is on `codex/qa-quality-audit`.

The user supplied the complete 1,295-character ISS context. CPU reproduction is
recorded in `logs/quality20261029/iss_diagnosis.json`. All six diagnostic questions
use one window, finite logits and valid original-text offsets. Therefore these
specific failures are not caused by lost overflow context or Unicode indexing.

For the speed question, the model selects the correct start but extends the answer
through “every 90 minutes”. Its independent end logits favor the later boundary.
For “What is my name?”, the Zarya span scores 6.0361 versus CLS 3.1096: the margin
is -2.9265, below the stable -0.9831 null threshold. The model therefore answers an
unsupported question. This is a semantic/model decision, not a fabricated answer
or an offset bug. Simply forcing abstention on that wording is not acceptable.

The API score is a sigmoid of absolute distance from the decision threshold. It
is not an empirically calibrated probability of correctness; the wrong Zarya
answer has score 0.8747. The GUI now labels it `Score (not a probability)` and avoids
a percentage. Historical screenshots predate this small label correction.

## Implementation review

The shared tokenizer bounds questions to 64 tokens and retains context through
384-token windows with stride 128. Training uses CLS for impossible answers or
answers outside a window. Spans require context tokens, ordered start/end and at
most 30 tokens; offsets slice the original string. Null aggregation takes the
minimum CLS score across windows. These are deliberate general choices, not proof
that every semantic answer will be correct. Independent boundary logits can still
favor an overly long span, and a plausible entity can outrank CLS incorrectly.

The experimental vectorized standard decoder matches production on randomized
fixtures. General length penalties and a shorter maximum are compared only on
calibration/development data. They are hypotheses, not established corrections.
The five public questions and ISS diagnostics never choose a policy or threshold.

## Data provenance and independence limits

The frozen manifest records 3,266 historically consulted official-validation IDs.
They cover every validation context, so there is no unused context-disjoint final
subset left in that split. Historical final scores are now development knowledge;
they cannot be described as a new blind test.

The new manifest prospectively reserves 1,024 development and 1,024 final examples
(512 answerable and 512 impossible each) from official **training** contexts not
used by any retained fine-tuned model or by official validation. Entire selected
contexts are excluded from candidate gradient training. The final examples remain
unscored until model and policy choices are locked. This is a new local holdout,
not the professor's test, and its source differs from historical validation.
Official training examples also have fewer annotated alternative answers than
official validation; this can change EM/F1. Only same-example comparisons below
are interpretable as measured differences between these candidates.

An abandoned full-data run has no retained checkpoint or complete consumed-ID
record. It is not an ancestor of the compared models, but universal historical
non-exposure cannot be certified. This limitation must accompany final results.
Calibration and checkpoint selection reuse their original 512-example partitions;
those are intentionally development resources, not independent evaluation.
Their IDs are disjoint but 132 contexts overlap, so they must not be represented
as independent paragraphs. `partition_audit.json` verifies that the new final and
development contexts overlap neither those partitions nor candidate training.

## Frozen comparison

`logs/quality20261029/plan.json` defines five postprocessors, the 75/25 weighted EM
objective, a 70% impossible-EM constraint, one optional 40k/one-epoch continuation,
and a 3,600-second training stop guard. Thresholds use calibration only; policies
use development only. Final paired comparisons include a context-cluster bootstrap
95% interval. No retuning is permitted after final scoring.

The candidate keeps the existing encoder and uses 20,000 prior plus 20,000 new
examples. Learning rate 2e-5 restarts a short schedule; this may improve adaptation
but can also cause forgetting, so the original checkpoint is preserved and the
selection split chooses the exported checkpoint. Real GPU benchmarking precedes
any run. No candidate is installed in production without user approval.

## Alternative encoders

An academically permitted alternative is a team-fine-tuned BERT-base encoder,
with greater CPU and GPU cost than six-layer DistilBERT. DeBERTa-v3-base has about
86M backbone plus 98M embedding parameters; it requires a separate memory and
latency feasibility check on this 4 GB GPU. No local quality or timing result is
claimed for either alternative. A ready-made SQuAD-fine-tuned checkpoint would not
satisfy the team's BASE fine-tuning requirement.

LoRA reduces trainable parameters through low-rank updates; it does not guarantee
better accuracy or eliminate inference cost. Continuing the already measured
DistilBERT is the bounded first experiment here.

References: [Transformers QA guide](https://huggingface.co/docs/transformers/en/tasks/question_answering),
[BERT-base model card](https://huggingface.co/google-bert/bert-base-uncased),
[DeBERTa-v3-base model card](https://huggingface.co/microsoft/deberta-v3-base),
[LoRA paper](https://arxiv.org/abs/2106.09685).

## Completed comparison and post-selection ISS check

The authorized run completed in 43m11s, selecting checkpoint-2000. See
[QUALITY_RESULTS.md](QUALITY_RESULTS.md) for the same-example comparison, bootstrap,
CPU HTTP checks and the recommendation pending approval. No further training is
authorized by the latest recovery request.

After final scoring, the candidate answered the speed question with exactly
“28,000 kilometers per hour” at offset 638. It still answered “Zarya” to “What is
my name?” at offset 284, score 0.9805. The other four diagnostic answers retained
their expected behavior. `iss_candidate.json` records raw candidates and logits.
These checks did not change any setting; high scores still do not establish
semantic correctness. Sample-04 likewise remains incorrect.
