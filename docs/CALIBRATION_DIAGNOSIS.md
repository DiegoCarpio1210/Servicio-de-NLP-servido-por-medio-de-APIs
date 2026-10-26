# Calibration diagnosis and fixed-policy evaluation

No additional training was run. The encoder, QA head, span algorithm and checkpoint-1522 weights are unchanged.
The active threshold is 1.7854061126708984, replacing -1.8834419250488281.
This improves the stated 75% answerable target mix, but substantially reduces correct abstentions.
It is not an across-the-board accuracy improvement or a prediction of the instructor's grade.

## Experiment boundaries

All diagnostic and policy-selection work used the 512 calibration examples only (256 per class).
The plan is recorded in `logs/recalibration/plan.json`. The policy was committed in `759bcb3` before
the post-adjustment holdout evaluation. Five-fold stratified cross-validation also stayed within calibration.
The 1,024 holdout examples and five public questions did not select thresholds or span settings.
This is a fixed post-adjustment check on an already reported holdout, not a newly collected unseen test set.
Further experiments should avoid repeatedly selecting models on these results.

## Causes established on calibration

1. Excessive abstention is a major contributor: the old threshold suppressed 182/256 answerable cases,
   including 73 exact raw spans. The new threshold suppresses only 11 exact raw spans.
2. Extraction remains a bottleneck. Before any null decision, only 131/256 answerable cases have an exact
   best span; 61 have partial overlap and 64 have no overlap. Raw span EM is 51.17%, F1 is 62.58%.
   A threshold cannot repair incorrect raw spans. Some semantic errors select plausible numbers without
   the relationship requested by the question.
3. All logits were finite. Independent context start/end argmax positions matched a gold boundary in
   144/256 and 158/256 answerable examples respectively. In 82 windows, independently chosen start/end
   positions were reversed or exceeded the length cap; the existing constrained joint search handles these.
   There is no evidence of swapped logits or broken indexing.
4. Every answerable gold span was represented by a window and fit the 30-token cap. No question in this
   calibration sample hit the question cap. All 512 fresh candidates matched stored candidates.
   Ten cases use multiple windows, seven answerable: their raw-span F1 is 34.94 versus 63.36 for single-window
   answerable cases. This tiny subgroup suggests difficulty, not a demonstrated overflow defect.
   No window, offset or span-selection change was justified.

Detailed evidence: `logs/recalibration/calibration_logit_audit.json`, `calibration_logit_details.json`,
and `error_taxonomy.json`. The minimum CLS aggregation across windows agrees with the
[Transformers QA postprocessor](https://github.com/huggingface/transformers/blob/v4.57.1/examples/pytorch/question-answering/utils_qa.py).

## Metric correction

The [official SQuAD v2 evaluator](https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py)
filters gold references that normalize to empty, unless every reference is empty. Our previous metric did not.
One mixed-reference calibration case exposed this bug; it was fixed before policy locking and regression-tested.
The final holdout comparison found one affected case there too. Consequently, the historical baseline EM/F1
52.34/53.40 becomes 52.25/53.30, and answerable EM/F1 25.05/27.17 becomes 24.85/26.97.
This is a scoring correction, not a weight or prediction change. Original evidence is retained in
`logs/recalibration/baseline_evaluation.json`; preliminary diagnostic output is explicitly superseded.

## Objective and calibration evidence

Rather than discard calibration rows to create a 75/25 sample, each class is weighted:
`target metric = 0.75 * answerable metric + 0.25 * unanswerable metric`.
This uses all 512 examples and avoids extra sampling noise. The threshold sweep maximizes weighted normalized
EM, then weighted F1, then proximity to the old threshold to resolve ties. The professor gives exact/correct-null
credit and may award partial credit through an LLM judge. Weighted EM is a conservative proxy for exact credit;
neither normalized EM nor token F1 exactly reproduces that judge. No claim of grade-optimal calibration is made.

On calibration, weighted EM increases 37.21 to 43.36 and weighted F1 38.77 to 49.53.
Five-fold calibration-only out-of-fold weighted EM increases 34.38 to 41.70, F1 35.94 to 47.87.
Fitted fold thresholds vary from 0.813 to 1.853, showing calibration uncertainty.
All sweep points and decisions are preserved in `logs/recalibration/locked_policy.json`.

## Fixed independent comparison

All values are percentages, using the corrected official metric in both columns.

| Metric | Previous policy | Adjusted policy |
|---|---:|---:|
| Overall EM, actual 511/513 class mix | 52.25 | 41.11 |
| Overall F1, actual class mix | 53.30 | 44.68 |
| Answerable EM | 24.85 | 48.73 |
| Answerable F1 | 26.97 | 55.88 |
| Unanswerable correct-null rate / EM | 79.53 | 33.53 |
| Target 75/25 weighted EM | 38.52 | 44.93 |
| Target 75/25 weighted F1 | 40.11 | 50.29 |
| Answerable cases incorrectly suppressed | 341/511 | 112/511 |
| Unanswerable cases incorrectly given a span | 105/513 | 341/513 |
| Public examples correct | 3/5 | 3/5 |

The weighted EM gain is 6.41 percentage points; a 2,000-resample paired class-stratified bootstrap gives
a 95% interval of 3.27 to 9.39 points, conditional on this dataset and the assumed prior. Weighted results
are estimates under a target mix, not scores on a separate 75/25 test. Domain shift and the unknown judge remain.
The new policy sacrifices substantial impossible-question accuracy. It should remain a candidate, not a claim
that model quality is sufficient for final delivery. The original threshold is preserved for reproducibility.

Evaluation used all 1,024 cached candidates with the fixed policy, plus 32 seeded fresh forwards; candidates,
offsets and margins matched. `logs/recalibration/holdout_comparison.json` records weight identity and timings.
The active API reports its threshold/objective and the unchanged weight hash through optional `/info` fields.
All 46 pytest tests and 19 real-model HTTP checks passed. Fifty sequential local requests took 15.32 seconds,
maximum 0.694 seconds. Public predictions remain 16, 3.5 days, 50,000 (wrong),
10,000 times and the GPU memory requirement by 3 times (wrong), and 50,257.
The earlier 20.47-second HTTPS measurement belongs to the previous threshold; no new tunnel test was run.

## Additional training options: proposals only

Measured training was 26m22s for two 12k epochs on the Quadro P1000, with batch 16, FP32 and dynamic padding.
These estimates scale that observation; workload, window lengths and thermal throttling may change them.
Allow additional time for checkpoint selection, CPU calibration and evaluation.

| Option | Estimated training | Benefit | Main risk |
|---|---|---|---|
| One more epoch on existing 12k | 13–18 minutes | Cheapest test of under-training; reuses learned head | Repetition can overfit and may not improve semantic coverage |
| One epoch on nested 20k (12k existing + 8k new) | 22–30 minutes | Broader coverage with modest time | Only one exposure to new cases; may alter abstention balance |
| Two epochs on nested 20k | 44–60 minutes | More learning on added examples | Larger commitment and no guaranteed gain |

My preferred next experiment for broader coverage is the nested 20k one-epoch option; the 12k extra epoch is
the lower-cost alternative. Keep stratified sampling on the official training split, seed 42, and all validation
partitions separate. Warm-start both encoder and QA head from checkpoint-1522. Consider learning rate 1e-5
(or 5e-6 for a conservative extra epoch), with a fresh short schedule and validation-loss checkpoint selection.
The original schedule has reached its endpoint: resuming the unchanged two-epoch configuration will not add
training. A changed dataset needs a fresh optimizer/scheduler workflow rather than old step-skipping semantics.
These workflow changes and hyperparameters are proposed, not implemented or validated experiments.
Keep the measured stable batch/precision configuration; do not expand automatically to 130,319 examples.
After any authorized weight change, recalibrate only on calibration, then perform a fixed evaluation.

No new training, final tag, weight upload or credential publication occurred. Instructor dry run, another-network
test, backup tunnel and authorized weight distribution remain open operational gates.
