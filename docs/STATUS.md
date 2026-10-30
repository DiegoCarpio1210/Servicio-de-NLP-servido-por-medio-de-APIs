# Current status — Pimientero — 2026-10-30

**quality40k is recommended for approval; continued20k remains active. Submission
is NOT complete.** No new training is authorized or started during this recovery.

quality40k completed 40,000 examples / one epoch / 2,531 steps in 43m11s on Quadro
P1000. Export matches checkpoint-2000 (minimum selection loss 1.441129).

Same final examples, with methodological limitations described in QUALITY_RESULTS:

| Profile | EM | F1 | Answerable EM | Answerable F1 | Impossible EM | 75/25 EM |
|---|---:|---:|---:|---:|---:|---:|
| continued20k stable | 57.71 | 60.81 | 36.33 | 42.52 | 79.10 | 47.02 |
| continued20k short15 | 57.81 | 60.83 | 36.33 | 42.36 | 79.30 | 47.07 |
| quality40k + length05 | 61.82 | 65.53 | 48.05 | 55.47 | 75.59 | 54.93 |

Weighted EM gain +7.91 points, paired context-cluster 95% CI [+5.48,+10.43].
Both models score 4/5 public examples; sample-04 remains incorrect. All three
profiles pass CPU HTTP authentication/contract checks and 50-request time budgets.
quality40k: 28.54s total, 1.42s maximum; no public tunnel was opened.

The GUI design and production hashes remain unchanged. The dashboard now displays
verified quality40k candidate metrics, explicitly separated from the active model.
Recovery tests: 60 passed; dashboard/gateway tests: 7 passed after this update.
Prior mobile Cloudflare and ngrok backup checks are completed for stable; they
are not a new external test of quality40k. See QUALITY_RESULTS.md, QUALITY_AUDIT.md,
CANDIDATE_REVIEW_ES.md and COMPLIANCE.md.

Pending: candidate approval/integration, sample-04 public conformance, official
instructor dry run and live URL/token operation, authorized separate weights,
team pitch/Q&A rehearsal, final review and explicitly authorized submission tag.
Keep REPORT.md; any later REPORT.pdf is supplementary unless the instructor says otherwise.

Historical status chronology is preserved in history/status_before_quality40k.md.
