# Individual Q&A preparation

Each member should open the named function, trace an example, and answer without an AI assistant.

| Question | Repository-specific answer |
|---|---|
| Where is authentication checked? | `qa.api.create_app`, middleware `guard`, before request-body validation; exact UTF-8 byte comparison with hmac.compare_digest. |
| Why does a missing token fail closed? | Empty startup token causes every protected request to return 401; default loader also rejects missing setup. |
| Why is health public and sometimes 503? | `health` checks whether the background loader has installed the warmed Engine. |
| What is trained? | The first run starts from BASE with a fresh QA head; continuation restores our own encoder and head. Both runs update all parameters. |
| How are impossible questions labeled? | `training_features` uses CLS start and end positions. |
| What if the answer is outside a window? | That window is labeled CLS even if another window contains the answer. |
| How is sample 05 handled? | `encode` uses overflowing tokens; Engine loops over every window, including the final one. |
| Why only_second truncation? | It applies overflow to context; the question has an explicit 64-token cap. |
| How are offsets preserved? | Fast tokenizer offset_mapping, context sequence IDs, original string slicing; no decoding of answer tokens. |
| What prevents backwards spans? | `aggregate` enumerates end indices from start up to 30 tokens. |
| Why minimum CLS across windows? | Unrelated windows should not force null when another window supports a span; standard SQuAD v2 aggregation. |
| What is threshold direction? | Null when null-minus-span exceeds threshold; lowering it increases abstention. |
| Is score a probability? | Sigmoid of threshold-distance; not a calibrated probability or an accuracy guarantee. |
| Why is validation not training? | No gradients use validation examples. The current run keeps selection, calibration and both holdouts separate; the fresh holdout also excludes historical contexts. Historical pilot evaluation used a weaker split. |
| How do subsets stay representative? | `subset` samples both answerability classes with a seeded NumPy RNG and logs exact counts. |
| How is evaluation normalized? | Lowercase, remove ASCII punctuation and English articles, normalize whitespace; maximum score over references. |
| Why a single worker? | Multiple workers duplicate weights and CPU contention; no measurement currently justifies them. |
| How does resuming work? | Trainer restores model, optimizer, scheduler and RNG from checkpoint-N via --resume. |
| What happens if training overwrites a calibrated model? | Training removes stale calibration; evaluate again before serving. |
| What failed in the smoke experiment? | Correct HTTP and offsets did not imply quality; answerable F1 was zero after only eight updates. |
| Does a decoder fix a null-heavy model? | Not automatically; supervision and calibration are still needed, and generated answers require grounding. |
| What is pending before submission? | Read docs/STATUS.md; never substitute expected results for actual measurements. |
| Why does the exported model have fewer steps than the run? | Training completed 2,534 continuation steps, but selection loss was lowest at step 1,600. We export that checkpoint; /info distinguishes selected and run steps. |
| Why a new holdout? | The old 1,024 scores had already been consulted. A fresh 1,024 excludes historical validation IDs and contexts; the candidate was fixed before its predictions were read. |
| What does 75/25 weighting mean? | It is .75 times the answerable metric plus .25 times the impossible metric, assuming class-conditional performance transfers. It is not the professor's actual grade. |
| Why constrain calibration? | Maximizing the target average alone severely reduced correct nulls. The continued model's threshold must first reach 70% impossible EM on calibration; this does not guarantee 70% on test. |
| Why no LoRA? | Full fine-tuning already fits the 4 GiB GPU and time budget. Adapter savings do not establish better QA accuracy, and we avoided an unmeasured integration change. |

Practice: locate each function, predict its output on a tiny example, then explain why a malicious or malformed
request cannot bypass the authentication checks. Each member must understand training, API and inference,
not only their preferred module. Bring the repository to the defense; no AI assistance is permitted during Q&A.

## Explain the system in plain language

Think of the model as two highlighters: one scores where an answer starts, the other where it ends.
The tokenizer first turns question and context into numbered pieces and remembers their character positions.
DistilBERT compares the pieces using attention. The QA head produces start and end scores for every piece.
Our code searches valid spans of up to 30 tokens and compares the best one with the CLS no-answer score.
If no-answer wins by the calibrated rule, return null; otherwise copy the original characters exactly.
Long paragraphs are split into overlapping windows, so the end is not silently lost.

Training adjusts weights from labeled examples. Calibration adjusts only the decision threshold on separate
examples. Final evaluation measures a fixed model and threshold on data not used for either choice.
Correct offsets prove where text came from, not whether it answers the question: sample-04 demonstrates this.

## Four-member rehearsal rotation

Each member must answer every topic; these are only first-round assignments. Rotate roles in round two.

| First speaker | Pitch responsibility | Individual code question and expected evidence |
|---|---|---|
| Diego Roberto Carpio Flamenco | Model choice and value | Trace tokenization to exact character offsets in qa/features.py and qa/spans.py. Why can a valid substring still be wrong? |
| Julio Roberto Guardado Quijano | Training and evidence | Explain why 2,534 run steps can export checkpoint-1600; show selection loss and explain which data never trains/calibrates. |
| Alfredo Enmanuel Cortez Martínez | Serving and security | Trace an invalid Bearer request through qa/api.py; show why health is public and why the secret stays outside Git. |
| Iván Daniel Salguero Esperanza | Decoder investigation and limitations | Explain sample-04's wrong entity attribution and why choosing a shorter span or a decoder is not a proven fix. |

Practice the whole pitch in three minutes using PITCH.md's four timed sections. Record time, not fabricated
performance claims. Each member then answers a randomly selected question in about 45 seconds with a code
reference, a concrete example and a limitation. A teammate may give feedback after practice, never during the
official Q&A. The rubric assigns 60 live-evaluation, 20 pitch and 20 individual Q&A points. Missing attendance
or report has separate consequences described in the professor's instructions.
