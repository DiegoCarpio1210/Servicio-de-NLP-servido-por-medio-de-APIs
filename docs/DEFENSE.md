# Individual Q&A preparation

Each member should open the named function, trace an example, and answer without an AI assistant.

| Question | Repository-specific answer |
|---|---|
| Where is authentication checked? | `qa.api.create_app`, middleware `guard`, before request-body validation; exact UTF-8 byte comparison with hmac.compare_digest. |
| Why does a missing token fail closed? | Empty startup token causes every protected request to return 401; default loader also rejects missing setup. |
| Why is health public and sometimes 503? | `health` checks whether the background loader has installed the warmed Engine. |
| What is trained? | `scripts.train.main` creates AutoModelForQuestionAnswering from BASE and updates all parameters, including the fresh QA head. |
| How are impossible questions labeled? | `training_features` uses CLS start and end positions. |
| What if the answer is outside a window? | That window is labeled CLS even if another window contains the answer. |
| How is sample 05 handled? | `encode` uses overflowing tokens; Engine loops over every window, including the final one. |
| Why only_second truncation? | It applies overflow to context; the question has an explicit 64-token cap. |
| How are offsets preserved? | Fast tokenizer offset_mapping, context sequence IDs, original string slicing; no decoding of answer tokens. |
| What prevents backwards spans? | `aggregate` enumerates end indices from start up to 30 tokens. |
| Why minimum CLS across windows? | Unrelated windows should not force null when another window supports a span; standard SQuAD v2 aggregation. |
| What is threshold direction? | Null when null-minus-span exceeds threshold; lowering it increases abstention. |
| Is score a probability? | Sigmoid of threshold-distance; not a calibrated probability or an accuracy guarantee. |
| Why is validation not training? | No gradients use validation examples; it selects checkpoint and threshold. In the current 12k run, checkpoint selection, threshold calibration and final holdout have disjoint example IDs. Historical pilot evaluation used a weaker split. |
| How do subsets stay representative? | `subset` samples both answerability classes with a seeded NumPy RNG and logs exact counts. |
| How is evaluation normalized? | Lowercase, remove ASCII punctuation and English articles, normalize whitespace; maximum score over references. |
| Why a single worker? | Multiple workers duplicate weights and CPU contention; no measurement currently justifies them. |
| How does resuming work? | Trainer restores model, optimizer, scheduler and RNG from checkpoint-N via --resume. |
| What happens if training overwrites a calibrated model? | Training removes stale calibration; evaluate again before serving. |
| What failed in the smoke experiment? | Correct HTTP and offsets did not imply quality; answerable F1 was zero after only eight updates. |
| Does a decoder fix a null-heavy model? | Not automatically; supervision and calibration are still needed, and generated answers require grounding. |
| What is pending before submission? | Read docs/STATUS.md; never substitute expected results for actual measurements. |

Practice: locate each function, predict its output on a tiny example, then explain why a malicious or malformed
request cannot bypass the authentication checks. Each member must understand training, API and inference,
not only their preferred module. Bring the repository to the defense; no AI assistance is permitted during Q&A.
