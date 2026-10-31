# Final local semantic review — 31 October 2026

This is a **local Codex review**, not Claude grading, an independent human review or the instructor's private evaluation. All 50 saved contexts, questions, references and actual API responses in `logs/semantic50/final_grading_export.json` were inspected. Predictions were not rerun; weights, threshold and extraction were not changed. The same assistant workflow authored these synthetic scenarios, so reviewer/author independence is not established.

The criterion is whether the response answers the precise question, is supported by the context, introduces no contradiction and omits no essential requested information. A noun or unit supplied by the question need not be repeated. High F1 alone is not sufficient. Uncertainty about required specificity is excluded from the conservative count.

## Ten non-exact answers

Short excerpts below explain decisions without publishing the full evaluation dataset. Leading whitespace is omitted for readability; original responses and offsets remain intact locally.

| ID | Decision | Evidence and reasoning |
|---|---|---|
| final-01-1 | Semantically correct | The question asks what **kind of bottle**; “amber glass” supplies the color/material stated in the context. The word bottle is already supplied by the question, so its omission loses no requested property. |
| final-01-2 | Semantically correct | The question explicitly asks how many **hours**; “four” gives the exact number in the reference. The unit is not missing from the question-answer pair. |
| final-01-3 | Semantically correct | “Technician Omar Vale” identifies the exact person who labels the bottles. The added occupation is explicitly supported and does not change identity. |
| final-05-3 | Semantically correct | “Teacher Nell Imani” identifies the named coordinator. The supported title adds no incorrect assertion. |
| final-06-1 | Semantically correct | The question explicitly asks how many **meters**; “3.2” supplies the correct quantity. |
| final-06-2 | Semantically correct | “After the water levels have equalized” retains the prerequisite governing when the gate may open. Omitting “only” does not license opening before equalization in this direct answer to the when-question. |
| final-07-4 | Semantically correct | The answer states the complete causal proposition: direct sunlight can fade the colors. “Because” is unnecessary in a direct response to “why”. |
| final-08-4 | Semantically correct | “On the low rear shelves” states the exact location of the picture books; the added preposition is appropriate and supported. |
| final-09-2 | Semantically correct | The question asks which **team**; “Amber” uniquely identifies Team Amber, distinguished from Team Violet in the context. |
| final-10-4 | Ambiguous / requires independent judgment | “Behind the kitchen” gives a supported location but omits the stainless steel tanks. A geographical reading of “where” accepts it; an exact storage-location reading requires the tanks. Excluded from supported full credit because the required specificity is uncertain. |

The other 40 normalized exact matches were also reviewed: 28 answerable responses supply the requested fact, and 12 null responses correctly decline information absent from their contexts. Normalized exact matching is not byte equality; original-context substring offsets were verified separately. No impossible question was answered non-null and no answerable question was incorrectly rejected.

## Result and interpretation

| Category | Count |
|---|---:|
| Reviewed normalized exact matches | 40/50 |
| Additional semantically correct non-exact answers | 9 |
| Confirmed semantically incorrect answers | 0 |
| Ambiguous answers, not awarded conservative credit | 1 |
| Conservative locally supported result | **49/50 (98%)** |
| Possible result if the ambiguous answer is accepted | **50/50 (100%)** |

Answerable local supported accuracy is 37/38 (97.37%); impossible-question accuracy is 12/12. The range 49–50 is an adjudication range, **not a statistical confidence interval**. The local review supports the 45/50 target on this synthetic simulation only. It does not establish 45/50 under Claude or the professor's unknown private rubric. The factual lexical metrics remain EM 40/50 and F1 95.29%; neither was replaced or recalculated as semantic accuracy.

Common authorship, only ten contexts with five questions each, fictional templated prose and deliberately repeated long-context padding limit generalization. Exact/near overlap checks against available historical records passed, but universal pretraining independence cannot be established. The historical 500-question SQuAD results remain the production dashboard metrics.

Independent review of all 50 examples is still advisable before treating this as an externally validated score. Preserve the original ungraded export; a separate local review record is kept at `logs/semantic50/final_local_review.json`. No external API credits were used, no data was transmitted to Claude and no production improvement was selected from these final answers.
