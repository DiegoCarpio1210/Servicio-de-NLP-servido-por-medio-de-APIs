# Requirements and acceptance inventory

Historical planning inventory. The current exhaustive review is [COMPLIANCE.md](COMPLIANCE.md).

Authority: original OpenAPI > PDF summary; original report headings are immutable.
All 12 PDF pages (I-XV), four ZIP entries and master prompt were read. ZIP CRC integrity passed.
Original files remain unchanged. No public case may enter training or calibration.

| Source | Acceptance criterion | Evidence / current gate |
|---|---|---|
| I, III | Extractive QA and abstention | qa/spans.py; real trained accuracy pending |
| II, IV, V.1 | Own fine-tuning from BASE encoder; English project | train.py base allowlist; training evidence pending |
| II | PBL team of 4-5; own laptop CPU serving | Team roster pending; Engine forces CPU |
| V.2 | No commercial inference | Local-only Engine; no inference provider |
| V.3, VII, VIII | Exact contract and authentication | tests/test_contract.py |
| V.4 | Environment token; never logged or committed | Middleware + generated test tokens + .gitignore |
| V.5, XI | Real Git history, no squash; final submission tag | Progressive commits; final tag pending final gates |
| VI | SQuAD 2 train only; validation separated, reproducible subset | qa/data.py; split metadata |
| VI | Five public cases, two impossible, long final answer | HTTP public evaluator; accuracy pending |
| VII | Public health 503 during load, 200 only ready | Loading tests; checkpoint and calibration required |
| VII | Info real base ID, parameter count, team, window metadata | Engine metadata + schema tests; real readiness pending |
| VII | Strict types, lengths, extras, JSON errors, exact id | Parameterized contract tests |
| VII | Exact Unicode character offsets, null pairing, finite score | Span and contract tests |
| VII, XII | Every context window; 5 s/item, 150 s/50 sequential | Overflow code; real latency pending |
| IX | Laptop tunnel HTTPS; second installed/tested tool | ngrok + cloudflared instructions; external tests pending |
| IX, X, XI | Dry run via tunnel; send URL; AC power; no sleep | Manual operational gate |
| XI | Repository shared; invite professor if private; weights external | Repository URL and authorized destination pending |
| XIII | Code, logs, loss curve, running service, report | Evidence paths in README; final evidence pending |
| XIV | 3-minute pitch, 4 criteria, individual Q&A without AI | docs/PITCH.md and docs/DEFENSE.md |
| XIV | Report complete; attendance; live accuracy | Draft is not final submission; rehearsal and attendance pending |
| XV | Readings inform technical explanation | docs/SOURCES.md |
| Template 0 | Team, members, repo, commit, public URL, AI disclosure | Missing factual fields explicitly PENDING |
| Template 1-3 | Model/alternative, real training, threshold | Model choice implemented; measurements pending |
| Template 4-6 | Architecture, five results, size/latency | Populate exclusively from execution logs |
| Template 7 | Two real issues with causes/fixes/commits | Actual issue log; no invented defects |
| Template 8-9 | Specific decoder discussion and limits | REPORT.md and defense |
| Master 3 | Resume, loss, real optimizer steps; tune validation only | TrainingArguments; checkpoints; evaluation split |
| Master 4 | Load once, inference_mode, CPU, warmup | qa/inference.py and qa/api.py |
| Master 6 | Unit, integration, schema, HTTP and security tests | tests/ and scripts/evaluate_public.py |
| Master 7 | Reproducible venv; PowerShell; five curl tests | requirements.txt and README |
| Master 8-9 | Honest status, English files, no automatic publishing | No submission tag until all final gates pass |

Stages: A inventory; B API and unit tests; C genuine smoke training; D calibration and CPU serving;
E HTTP/latency/tunnel checks; F final audit and team handoff. A smoke model is not an academic-quality final model.
