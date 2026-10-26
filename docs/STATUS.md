# Current status — Pimientero

One controlled continuation was authorized on 2026-10-26 and is running. No submission tag exists.
See [CONTINUATION_EXPERIMENT.md](CONTINUATION_EXPERIMENT.md) for its frozen plan and recovery details.

The completed recalibration experiment is documented in [CALIBRATION_DIAGNOSIS.md](CALIBRATION_DIAGNOSIS.md).
Original 12k/two-epoch weights are preserved. With adjusted threshold 1.78540611, old holdout EM/F1 is
41.11/44.68, answerable 48.73/55.88, impossible 33.53, and 75/25 weighted EM/F1 44.93/50.29.
This improves the target-weighted objective but reduces correct abstentions. Public cases remain 3/5.
46 tests and 19 real HTTP checks passed; 50 local requests took 15.32 seconds (maximum .694).
Historical HTTPS timing 20.47 seconds used the previous threshold; independent-network verification is pending.

The continuation starts from the saved trained weights with a nested 20k sample, two additional epochs and
1e-5 learning rate. Estimate 44–60 minutes; time guard at 55 minutes. Results are not yet available.
Selection/calibration remain separate; a fresh 1,024-example test excludes historical validation contexts.
Do not launch another run. Inspect logs/continued20k/job.json and history_live.json for actual progress.

Remaining delivery gates: instructor token entered locally, backup tunnel and another-network rehearsal,
official dry run and active URL, authorized separate weight distribution, English pitch/Q&A preparation,
quality review and later explicit submission-tag authorization. See COMPLIANCE.md for the full matrix.
