# Approved quality40k deployment and candidate review

quality40k checkpoint-2000 is active. Run `INICIAR_PIMIENTERO.bat` and open
http://127.0.0.1:8765. continued20k checkpoint-1600 remains intact as backup.
The dashboard displays the active quality40k evaluation, not pretrained results.

## Frozen approved configuration

- Weights: `models/quality40k/model.safetensors`.
- SHA-256: `d42cc12e1ba17bc04e1ff652bd7e3bc7292d6d9ce39032501820c470b844d60f`.
- Maximum span length: 30 tokens.
- Span score: start logit + end logit - 0.5 * log(token length).
- Null threshold: -0.40494251441686124, applied to minimum-window CLS minus best span.
- Sequence length 384, stride 128, CPU inference with four production threads.
- Tokenizer/config/calibration hashes: `configs/active_model.json`.
- Original decision record: `logs/quality20261029/final_lock.json`, profile trained.

Selection and calibration were committed before final scoring in `37fc668`.
The policy contains no question-specific rules. Subsequent production HTTP outputs
matched the evaluated candidate exactly. See `logs/quality20261029/activation.json`.
The architecture and white/black/gray design are preserved. A browser automation
timeout means the activation run did not include a new visual-click verification;
real UI gateway inference, offsets, Unicode, long contexts and authentication passed.

## New pretrained candidates

The user now reports that the professor permits pretrained QA models. The
benchmark evaluates the named Deepset models without training, using the same
context-disjoint local validation partitions for all compared models. It does not
replace quality40k or update its calibration. Published vendor results are not
local measurements, and the historical validation-exposure limitations remain.

Any future replacement requires separate user approval. A promotion must activate
weights, tokenizer, policy and threshold together, update expected hashes and the
dashboard's evaluation source, and repeat critical API/UI checks. Preserve this
approved model as a rollback option. Candidate approval never authorizes public
exposure, credential publication or the submission tag automatically.
