# Remaining human-operated rehearsal

Concrete PowerShell steps for the team: [RUNBOOK_ES.md](RUNBOOK_ES.md).
Use the new `scripts.rehearse` runner for HTTPS deadlines, schema/authentication checks and separate technical
versus accuracy outcomes. The older evaluator below remains a historical command, not proof of a completed external test.

The current candidate is models/continued20k, checkpoint-1600, threshold -0.98309326171875.
Its 47 automated tests, 19 real HTTP checks and local 50-request timing passed. Public sample 04 is still wrong:
the model extracts GPT-3 reduction quantities for a question about BERT-base. The public set is 4/5, not 5/5.
The professor requires the public sample set to pass at the dry run. Review this unresolved quality issue;
do not hard-code an answer or claim that the service is ready for submission.

1. Obtain the instructor's development token. Enter it locally with the secure README Read-Host prompt;
   never paste it in chat, source files or Git. Use the different presentation-day token when provided.
2. From the project directory, set TEAM_ID=Pimientero and MODEL_DIR=models/continued20k. Start the README
   single-worker uvicorn command. Check /health locally before making protected calls.
3. A temporary authorized current-model HTTPS test from the origin laptop passed and was closed; five manual another-network checks also passed (logs/continued20k/mobile_connectivity.json). Both temporary sessions are now inactive. When you decide to authorize/open the rehearsal,
   run `.\tools\cloudflared.exe tunnel --url http://localhost:8000` in a separate terminal. Keep its HTTPS URL.
   The manual mobile checks are separate from the 50-request same-host HTTPS timing.
4. COMPLETED: ngrok 3.39.11 backup passed four real HTTPS checks from the laptop; see logs/ngrok_setup.json. For future setup, create/sign in to your account and configure
   its authtoken locally. Do not share that authtoken. Test `ngrok http 8000` as the separate backup tunnel.
5. COMPLETED for Cloudflare: on a second device/network (for example mobile data), enter the QA token securely and run the five curl
   checks from README against the HTTPS URL: health 200, missing-token info 401, wrong-token info 401,
   valid-token info 200, valid predict 200 with exact offsets. A second terminal on the same laptop/network
   does not establish independent-network connectivity.
6. Additional full external-network timing remains unmeasured. From that external network, run `python -m scripts.evaluate_public --url <HTTPS-URL> --repeat 10
   --output logs/external_network_verification.json` with QA_API_TOKEN set locally. Record the actual results.
   Require <5 seconds per request and <150 seconds for all 50; these are repeated public inputs, not hidden data.
   An exit code of 1 can mean wrong predictions even when connectivity and schemas pass; inspect the JSON.
7. Arrange the instructor's official dry run. Send the active HTTPS URL before the slot and again if it changes.
   Keep the laptop on AC with sleep disabled, and keep the API/tunnel running throughout. The instructor's
   harness and token checks have not been executed here. Stop the temporary tunnel after your rehearsal.
8. Choose an authorized separate release/shared-folder destination for model weights and tokenizer/metadata.
   They remain local and ignored by Git. Preserve the original model and all resumable checkpoints.
9. Rehearse the three-minute English pitch and individual Q&A. All four members must understand training,
   calibration, security and offsets, attend, and answer without AI assistance or teammate help.
10. After all quality and operational gates are resolved, review REPORT.md, fill the real dry-run URL and
    final commit reference, then explicitly authorize creation/push of the submission tag. No tag exists yet.

The rubric is 60 points live evaluation, 20 team pitch and 20 individual Q&A. Partial credit uses the
instructor's judge and provisional anchors; SQuAD weighted EM is not a grade prediction. A missing/empty report
halves the final grade. The current report has every required heading, but final delivery fields remain pending.
