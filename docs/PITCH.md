# Three-minute English pitch

Rehearsal target: roughly 360 words at 120 words/minute. Replace bracketed factual fields only after verification.
This is a rehearsal draft; report only the completed candidate metrics; do not present pending results as final.

**0:00-0:40 — Model choice.**
Our project exposes extractive question answering as an authenticated API. Given a paragraph and a question,
we return an exact substring or null when the paragraph does not support an answer. We chose the base
DistilBERT encoder and trained its question-answering head and encoder ourselves. Its six layers make it a
practical candidate for CPU inference on our laptop. BERT-base has twelve layers and might improve quality,
but we have not measured that trade-off empirically. Our loaded QA model contains 66,364,418 parameters.

**0:40-1:30 — Training and evidence.**
We use the official SQuAD 2.0 training split, with a fixed seed and representative sampling when resources
are limited. The evaluated run uses 12,000 examples and two epochs. Our loss curve and validation metrics
are saved in the repository. Unanswerable questions and windows without the answer are trained at the CLS
position. Long contexts become overlapping windows rather than being cut off. We tune the abstention
threshold using validation only, and report a separate threshold holdout. Our measured answerable F1 is
27.17 percent, and our unanswerable F1 is 79.53 percent. The five public cases never enter training.

**1:30-2:20 — Service and access.**
FastAPI validates the exact request schema. The model loads once, warms up, and runs on CPU with one server
worker. Health reports loading until initialization succeeds. Protected routes require an exact, case-sensitive
Bearer header, with the token read from the environment. For each request, we select a valid span across all
windows, compare it with the null score, and slice the original context. That guarantees character offsets
remain meaningful. Our HTTPS tunnel forwards requests to the laptop. Locally, the maximum was 0.559 seconds and fifty requests took 12.51 seconds. The same candidate took 20.47 seconds through HTTPS from this host. The instructor dry run is still pending.

**2:20-3:00 — Decoder investigation and limitations.**
Our evaluated model answered three public cases correctly but failed both impossible questions, selecting
numbers that did not support the requested facts. A working API does not establish model accuracy. A decoder
could generate explanations or combine non-contiguous facts, but generated text may violate the exact-span
contract. It would also require sequential generation and additional grounding checks. We therefore keep
our encoder, document limitations honestly, and distinguish tested software behavior from measured model quality.
