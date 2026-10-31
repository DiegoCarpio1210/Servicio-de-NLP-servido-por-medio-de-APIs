"""Report the locked pilot's actual artifacts; never rerun inference or fitting."""
import json,statistics
from pathlib import Path
from experiments.deberta_lora_smoke import ROOT,digest,protected_hashes

OUT=ROOT/'logs/deberta_lora/pilot200'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    progress=read(OUT/'progress.json');history=read(OUT/'training_history.json');validation=read(OUT/'validation_metrics.json')
    assert progress['state']=='completed' and progress['step']==200
    assert [r['step'] for r in history]==list(range(1,201))
    assert [r['step'] for r in validation]==[0,100,200]
    assert protected_hashes()==read(OUT/'protected_before.json')
    manifest=read(OUT/'manifest.json');cfg=read(OUT/'config.json');recovery=read(OUT/'resume_verification.json')
    blocks=[{'updates':f'{start+1}-{start+50}','mean_loss':statistics.mean(r['loss'] for r in history[start:start+50])} for start in range(0,200,50)]
    summary={'completed':True,'updates':200,'config':cfg,'dataset_audit':manifest['audit'],
      'dataset_revision':manifest['adversarial_revision'],'feature_stats':read(OUT/'feature_stats.json'),
      'validation':validation,'loss_blocks':blocks,'training_seconds':sum(r['seconds'] for r in history),
      'training_phase_wall_seconds':progress['elapsed_seconds'],'mean_update_seconds':statistics.mean(r['seconds'] for r in history),
      'recorded_peak_allocated_mib':max(r['peak_allocated_bytes'] for r in history)/2**20,
      'recorded_peak_reserved_mib':max(r['peak_reserved_bytes'] for r in history)/2**20,
      'recorded_peak_process_ram_mib':max(r['peak_rss_bytes'] for r in history)/2**20,
      'best_step':progress['best_step'],'recovery':recovery,'protected_files_unchanged':True,
      'latest_checkpoint':progress['latest_checkpoint'],'calibration_used':False,'previous_benchmarks_re_evaluated':False,
      'manifest_sha256':digest(OUT/'manifest.json'),'max_gradient_norm_before_clip':max(r['gradient_norm'] for r in history)}
    latest=OUT/progress['latest_checkpoint']['directory'];meta=read(latest/'metadata.json')
    assert digest(latest/'adapter/adapter_model.safetensors')==meta['adapter_sha256']
    summary['latest_adapter_sha256']=meta['adapter_sha256']
    (ROOT/'logs/deberta_lora_pilot200_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    old,new=validation[0],validation[-1]
    lines=['# Controlled DeBERTa LoRA pilot: actual results','',
      '**Completed exactly 200 optimizer updates. No adapter was activated or merged.**',
      'Stable main remains `02a9d34`; work is isolated on `experiment/deberta-lora`. This is development evidence, not an official final evaluation.','',
      '## Provenance and overlap audit','',
      'Training: **1,024 questions**, 512 answerable AdversarialQA training examples plus 512 genuine SQuAD2 training unanswerables. Development: **128 questions**, 64 AdversarialQA validation answers plus 64 SQuAD2 training unanswerables. A further **128 calibration examples** remain unused. Every selected question has a distinct normalized context and normalized question across all three partitions.',
      '', 'AdversarialQA: [UCLNLP dataset](https://huggingface.co/datasets/UCLNLP/adversarial_qa), revision `c2d5f738db1ad21a4126a144dfbb00cb51e0a4a9`, crowdsourced adversarial extractive questions. SQuAD2: [Stanford source](https://rajpurkar.github.io/SQuAD-explorer/). Both are distributed under CC BY-SA 4.0; source file SHA-256 values and local SQuAD fingerprints are frozen in the committed manifest. Original answers and offsets are retained; no artificial null labels were manufactured.',
      '',f"The audit scanned {manifest['audit']['scanned_records']} local records, excluding {manifest['audit']['historical_ids_excluded']} historical question IDs and {manifest['audit']['historical_contexts_excluded']} normalized contexts. AdversarialQA train contained 30,000 rows, 51 duplicate context/question pairs and 1,142 historical-overlap exclusions; validation contained 3,000 rows, 3 duplicates and 1,072 historical-overlap exclusions. No invalid or empty answer spans were found. The eligible SQuAD negative pool had 42,470 examples. Selection was frozen before fitting in commit `d03c05b`.",
      '', 'The audit includes the previous 500- and 1,024-question benchmark records and professor public examples. None was used for training, calibration or this development evaluation. The vendor already trained on SQuAD2, and AdversarialQA shares Wikipedia/SQuAD source contexts: this is complementary-question adaptation, not certified unseen data. Exact normalized context/question checks do not exclude every paraphrase or near-duplicate.',
      '', '**Important design limitation:** answerability is confounded with dataset source in this pilot: all selected AdversarialQA examples have answers and all selected SQuAD examples are impossible. Question-style cues may help classification. Any future quality experiment should include SQuAD positives too and use a separately annotated balanced external set. Do not infer generalization from this mixed-source development cohort.',
      '', '## Frozen training and inference configuration','',
      'Pinned original DeBERTa; rank 4, alpha 8, dropout .05, FP32, microbatch 1, accumulation 8, non-reentrant gradient checkpointing. Only query/value LoRA and the saved QA head train: **148,994 parameters**. LR 2e-5, 10 warmup updates, linear decay through update 200, AdamW weight decay .01, gradient clipping 1.0; seed 20261031. Training windows: 256 tokens, stride 96; questions bounded by the existing shared preprocessing utility. Genuine impossible questions and answer-outside-window cases receive CLS start/end supervision.',
      '',f"Tokenization produced {summary['feature_stats']['features']} windows: {summary['feature_stats']['positive_features']} positive and {summary['feature_stats']['null_features']} CLS-supervised windows. The run consumed exactly 1,600 microbatches, using deterministic epoch shuffles over windows; 200 updates is not a claim of 200 epochs or 1,600 distinct source questions.",
      '', 'Development uses the production 384-token/128-stride inference policy, max span 30, length penalty 0, threshold **2.7772940397262573**. No threshold sweep or calibration was performed. Model selection was declared as maximum development 75/25 weighted EM, with ties retaining the earlier state including the original base at step 0. No final test set was consulted.',
      '', '## Development metrics','',
      '| Update | Overall EM | Overall F1 | Answerable EM | Unanswerable EM | Weighted 75/25 EM | Validation loss |',
      '|---:|---:|---:|---:|---:|---:|---:|']
    for row in validation:
        g=row['groups'];lines.append(f"| {row['step']} | {g['all']['EM']:.2f}% | {g['all']['F1']:.2f}% | {g['answerable']['EM']:.2f}% | {g['unanswerable']['EM']:.2f}% | {row['weighted_75_25']['EM']:.2f}% | {row['validation_loss']:.4f} |")
    lines += ['',f"Selected development state: **step {summary['best_step']}**. Overall EM change from the same-cohort base to update 200: {new['groups']['all']['EM']-old['groups']['all']['EM']:+.2f} points; F1 change: {new['groups']['all']['F1']-old['groups']['all']['F1']:+.2f} points.",
      '', 'EM/F1 use the existing official SQuAD normalization and best original reference. Validation loss is the mean of overflow-window losses; metrics are per question. Each class has only 64 questions, so a single class error moves EM by 1.5625 points. This dataset differs from the previous 500-question benchmark; do not subtract its 83.00/86.69 baseline scores from these development results.',
      '', '## Loss, resources and duration','', '| Update block | Mean training loss |','|---|---:|']
    for block in blocks:lines.append(f"| {block['updates']} | {block['mean_loss']:.4f} |")
    lines += ['',f"Optimizer-loop time: **{summary['training_seconds']:.2f} seconds**; mean update **{summary['mean_update_seconds']:.3f}s**. Training-phase wall clock including development evaluation and checkpoints: **{summary['training_phase_wall_seconds']:.2f}s**. Model loading, initial tokenization and the deliberate recovery pause between processes are excluded from this clock; these are not fabricated end-to-end timings.",
      '',f"Maximum recorded PyTorch allocated/reserved peaks across training logs: **{summary['recorded_peak_allocated_mib']:.2f} / {summary['recorded_peak_reserved_mib']:.2f} MiB**. Peak recorded process RAM: **{summary['recorded_peak_process_ram_mib']:.2f} MiB**. These are allocator/process counters, not total computer usage. Finite losses and gradients were checked on every update; frozen parameters had no gradients. Gradient clipping was applied after checking and before each optimizer step.",
      '', '## Checkpoints and actual recovery test','',
      f"The pilot stopped cleanly at update {recovery['restored_step']} and resumed from `checkpoint-{recovery['restored_step']:04d}`. Verified restoration: cursor {recovery['restored_cursor']}, {recovery['optimizer_state_entries']} optimizer-state entries, scheduler epoch {recovery['scheduler_last_epoch']}, complete history, Python/NumPy/Torch/CUDA RNG state. Updates 1–200 occur exactly once in the final history. No silent restart or extra training budget was used.",
      '', 'Local checkpoints are under `logs/deberta_lora/pilot200/checkpoint-NNNN/`: adapter and QA-head safetensors, trusted local `trainer.pt` (optimizer, scheduler, RNG, cursor, history) and hash metadata. Checkpoints are written to unique partial directories and atomically renamed before updating `latest.json`. Partial directories are never treated as resumable checkpoints. Latest complete checkpoint: `checkpoint-0200`.',
      '', 'From this branch, `PILOTO_LORA.bat` starts only a new frozen run; `PILOTO_LORA.bat --resume` explicitly restores the latest checkpoint. It refuses a silent restart. `DETENER_PILOTO_LORA.bat` creates STOP; the current update finishes and is saved. Remove only `logs/deberta_lora/pilot200/STOP` before explicit resume. A completed 200-update run exits without further updates. Load trainer.pt only from trusted local checkpoints.',
      '', 'For a fresh checkout, recover the pinned public base as documented in STABLE_DEBERTA.md, install isolated PEFT as in DEBERTA_LORA_SMOKE_TEST.md, and restore the exact pinned AdversarialQA files and original cached SQuAD dataset. `python -m experiments.restore_lora_pilot_data` reconstructs the committed ID selection and checks its byte hash; it never resamples after results. Adapter/checkpoint files are deliberately local and must be backed up separately to resume actual training.',
      '', '## Interpretation and preserved baseline','',
      'Decreasing training loss alone does not demonstrate success. Use the table above and its source-confounding/small-sample limitations. This pilot does not establish the experimental targets (overall EM >=85, F1 >=88, answerable EM >=80, unanswerable EM >=95) on an independent final set. No new production metrics replace the stable 83.00/86.69 benchmark.',
      '', 'The stable DeBERTa files, tokenizer, frozen policy and quality40k backup hashes were verified unchanged. API, dashboard, stable requirements and main were not modified. All adapters/data/raw predictions remain ignored; only source code, manifests and compact results are published on experiment/deberta-lora. No automatic activation, public tunnel, submission tag or merge into main occurred.',
      '', 'Evidence: logs/deberta_lora_pilot200_summary.json; full local logs/deberta_lora/pilot200/{training_history,validation_metrics,progress,resume_verification}.json. Calibration remains unused, and no prior final benchmark was rerun.']
    (ROOT/'docs/DEBERTA_LORA_PILOT200.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('updates','training_seconds','training_phase_wall_seconds','best_step','recorded_peak_allocated_mib','recorded_peak_reserved_mib','recorded_peak_process_ram_mib')}))

if __name__=='__main__':main()
