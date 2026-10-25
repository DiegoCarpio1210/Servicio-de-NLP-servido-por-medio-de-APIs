"""Run from repository root: python -m scripts.train --config configs/train.json."""
import argparse
import json
import platform
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainerCallback, DataCollatorWithPadding, TrainingArguments, set_seed
from qa.data import load_splits, stats, validation_partitions
from qa.training_control import training_lock
from qa.features import training_features, trim_training_padding


class EvidenceCallback(TrainerCallback):
    def __init__(self, logdir, save_steps=500):
        self.logdir=logdir
        self.save_steps=save_steps
        self.stop_requested=False
        self.logdir.mkdir(parents=True,exist_ok=True)

    def on_log(self, args, state, control, **kwargs):
        (self.logdir/'history_live.json').write_text(json.dumps(state.log_history,indent=2))

    def on_step_end(self, args, state, control, **kwargs):
        # Preserve resumable state even during long epochs; recent checkpoints and the best validation checkpoint are retained.
        if state.global_step % self.save_steps == 0:
            control.should_save=True
        if state.max_steps and state.global_step >= state.max_steps:
            control.should_save=True
            control.should_evaluate=True
        if (self.logdir/'STOP').exists():
            self.stop_requested=True
            control.should_save=True
            control.should_evaluate=True
            control.should_training_stop=True
        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/subset12k.json')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    if c['base_model'] not in ('distilbert/distilbert-base-uncased', 'google-bert/bert-base-uncased'):
        raise ValueError('Use one of the reviewed BASE encoders, never a SQuAD checkpoint')
    set_seed(c['seed'])
    torch.set_num_threads(4)
    raw = load_splits(c)
    logdir=Path('logs')/Path(c['model_dir']).name
    logdir.mkdir(parents=True,exist_ok=True)
    if (logdir/'STOP').exists():
        raise RuntimeError('Remove the acknowledged STOP file before resuming')
    partitions=None
    if 'validation_partition' in c:
        partitions=validation_partitions(raw['validation'],c)
        raw['validation']=partitions['selection']
        manifest={'train_ids':list(raw['train']['id']),**{k+'_ids':list(v['id']) for k,v in partitions.items()},
                  'train_stats':stats(raw['train']), 'validation_stats':{k:stats(v) for k,v in partitions.items()}}
        (logdir/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    tokenizer = AutoTokenizer.from_pretrained(c['base_model'], use_fast=True, cache_dir='cache', revision=c.get('base_revision','main'))
    model = AutoModelForQuestionAnswering.from_pretrained(c['base_model'], cache_dir='cache', revision=c.get('base_revision','main'))
    encoded = {k: ds.map(lambda x: training_features(x, tokenizer, c['max_seq_length'], c['doc_stride']),
                        batched=True, remove_columns=ds.column_names) for k, ds in raw.items()}
    if c.get('dynamic_padding'):
        encoded={k:ds.map(lambda x:trim_training_padding(x,tokenizer.model_input_names),batched=True) for k,ds in encoded.items()}
    gpu = torch.cuda.is_available() and not args.cpu
    if c.get('require_cuda') and not gpu:
        raise RuntimeError('CUDA required by this configuration; refusing silent CPU fallback')
    if gpu: torch.cuda.reset_peak_memory_stats()
    callback=EvidenceCallback(logdir,c.get('save_steps',500))
    training_args = TrainingArguments(output_dir=c['output_dir'], num_train_epochs=c['epochs'],
        learning_rate=c['learning_rate'], per_device_train_batch_size=c['batch_size'],
        per_device_eval_batch_size=c['batch_size'], gradient_accumulation_steps=c['gradient_accumulation_steps'],
        warmup_ratio=c['warmup_ratio'], weight_decay=c['weight_decay'], lr_scheduler_type='linear',
        eval_strategy='steps' if c.get('eval_steps') else 'epoch',
        save_strategy='steps' if c.get('eval_steps') else 'epoch',
        eval_steps=c.get('eval_steps'), save_steps=c.get('eval_steps',500),
        logging_steps=1 if c['max_steps']>0 else c.get('logging_steps',50),
        save_total_limit=3, load_best_model_at_end=True, metric_for_best_model='eval_loss',
        seed=c['seed'], data_seed=c['seed'], fp16=gpu and c.get('fp16',True), use_cpu=not gpu,
        group_by_length=c.get('dynamic_padding',False),
        report_to='none', dataloader_num_workers=0, max_steps=c['max_steps'])
    trainer = Trainer(model=model, args=training_args, train_dataset=encoded['train'],
                      eval_dataset=encoded['validation'], processing_class=tokenizer,
                      callbacks=[callback],
                      data_collator=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8) if c.get('dynamic_padding') else None)
    runtime={'device':'cuda' if gpu else 'cpu','gpu':torch.cuda.get_device_name(0) if gpu else None,
             'training_windows':len(encoded['train']),'selection_windows':len(encoded['validation']),
             'config':c,'resume_from':args.resume}
    (logdir/'runtime.json').write_text(json.dumps(runtime,indent=2))
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=args.resume)
    elapsed = time.perf_counter()-started
    root = Path(c['model_dir']); root.mkdir(parents=True,exist_ok=True)
    trainer.save_model(root); tokenizer.save_pretrained(root)
    # A previous model's threshold must never survive retraining in the same directory.
    (root/'calibration.json').unlink(missing_ok=True)
    metadata = {**c, 'optimizer_steps': trainer.state.global_step, 'elapsed_seconds': elapsed,
                'completed':not callback.stop_requested, 'actual_epochs':trainer.state.epoch,
                'best_checkpoint':trainer.state.best_model_checkpoint,
                'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if gpu else None,
                'peak_gpu_reserved_mib':torch.cuda.max_memory_reserved()/2**20 if gpu else None,
                'dataset': 'rajpurkar/squad_v2', 'splits': {k:stats(ds) for k,ds in raw.items()},
                'hardware': torch.cuda.get_device_name(0) if gpu else platform.processor(),
                'torch_version': torch.__version__, 'base_revision': getattr(model.config, '_commit_hash', None),
                'params': sum(p.numel() for p in model.parameters()), 'training_device': 'cuda' if gpu else 'cpu'}
    (root/'training_metadata.json').write_text(json.dumps(metadata,indent=2))
    logdir=Path('logs')/root.name; logdir.mkdir(parents=True,exist_ok=True)
    (logdir/'training_metadata.json').write_text(json.dumps(metadata,indent=2))
    (logdir/'history.json').write_text(json.dumps(trainer.state.log_history,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key in ('loss','eval_loss'):
        entries=[r for r in trainer.state.log_history if key in r]
        plt.plot([r['step'] for r in entries],[r[key] for r in entries], marker='.',label=key)
    plt.xlabel('Optimizer step'); plt.ylabel('QA loss'); plt.legend(); plt.tight_layout()
    plt.savefig(logdir/'loss.png'); plt.close()
    print(json.dumps(metadata,indent=2))
    if callback.stop_requested: raise SystemExit(75)


if __name__ == '__main__':
    with training_lock(): main()
