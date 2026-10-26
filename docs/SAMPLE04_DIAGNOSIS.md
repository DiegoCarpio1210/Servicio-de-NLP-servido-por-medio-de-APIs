# Sample-04: diagnostic inspection, no tuning

One CPU forward inspection of the unchanged candidate was executed. Evidence:
`logs/continued20k/sample04_diagnosis.json`; reproducer: `scripts/diagnose_public_failure.py`.
No counterfactual prompts, threshold sweeps, public-specific rules or training were used.

The context attributes LoRA reductions to GPT-3 175B. The question asks about BERT-base, for which no
reduction is supplied. The correct answer is null, even though the context contains plausible numerical spans.

| Check | Observation | Meaning |
|---|---|---|
| Artifact | SHA-256 7d66ffcff40157ed8f9f728f51ffe3c1b5b3f249ac8c100ece9666d1f7007848 | Same selected checkpoint-1600 export |
| Device | CPU | Actual diagnostic forward used CPU |
| Coverage | One window, 124 active tokens; context ends at character 464 of 464 | No truncation or multi-window aggregation issue in this case |
| Logits | All finite | No NaN/infinity corruption |
| Best span | Characters 263–317, 13 tokens, score 11.98944 | Valid exact extraction, but wrong semantic attribution and overlong boundary |
| Second span | `3 times`, characters 310–317, score 11.96195 | Near-tie; shortening the span would still answer about the wrong model |
| Null score | CLS start 4.68248 + end 5.17728 = 9.85976 | Correctly computed from model outputs |
| Decision | Margin -2.12968; threshold -0.98309; return null only if margin > threshold | The implementation correctly follows its locked decision rule |

There are two prediction weaknesses: imperfect span boundaries and failure to reject a plausible span whose
entity does not match the question. The second is decisive for the expected null. This is consistent with a
learned semantic generalization limitation; raw logits do not reveal a definitive internal causal mechanism.
The `score` is sigmoid distance from the threshold, not a calibrated probability of factual correctness.

Changing the threshold could change the decision, but choosing a value from this public example would be
public-set tuning. A smaller maximum span or a shorter-span preference would not establish that BERT-base
has the claimed reduction. Likewise, a rule rejecting a question whenever a named string is absent would
break legitimate paraphrases and coreference and has not been validated as a general correction.

No inference bug has been demonstrated here, so no production change is justified from this case alone.
A future authorized quality experiment could use independent non-public calibration examples with entity
mismatches and numerical distractors to test abstention improvements. It would need a fixed protocol and a
new final test; none was performed in this task. The current public result remains honestly 4/5.
