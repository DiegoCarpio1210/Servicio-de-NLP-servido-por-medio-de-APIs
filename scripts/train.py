"""Run from repository root: python -m scripts.train --config configs/train.json."""
import argparse
import json
import platform
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainerCallback, TrainingArguments, set_seed
from qa.data import load_splits, stats
from qa.features import training_features


class EvidenceCallback(TrainerCallback):
    def __init__(self, logdir):
        self.logdir=logdir
        self.logdir.mkdir(parents=True,exist_ok=True)

    def on_log(self, args, state, control, **kwargs):
        (self.logdir/'history_live.json').write_text(json.dumps(state.log_history,indent=2))

    def on_step_end(self, args, state, control, **kwargs):
        # Preserve resumable state even during long epochs; latest two checkpoints are retained.
        if state.global_step % 500 == 0:
            control.should_save=True
        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train.json')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    if c['base_model'] not in ('distilbert/distilbert-base-uncased', 'google-bert/bert-base-uncased'):
        raise ValueError('Use one of the reviewed BASE encoders, never a SQuAD checkpoint')
    set_seed(c['seed'])
    torch.set_num_threads(4)
    raw = load_splits(c)
    tokenizer = AutoTokenizer.from_pretrained(c['base_model'], use_fast=True, cache_dir='cache', revision=c.get('base_revision','main'))
    model = AutoModelForQuestionAnswering.from_pretrained(c['base_model'], cache_dir='cache', revision=c.get('base_revision','main'))
    encoded = {k: ds.map(lambda x: training_features(x, tokenizer, c['max_seq_length'], c['doc_stride']),
                        batched=True, remove_columns=ds.column_names) for k, ds in raw.items()}
    gpu = torch.cuda.is_available() and not args.cpu
    training_args = TrainingArguments(output_dir=c['output_dir'], num_train_epochs=c['epochs'],
        learning_rate=c['learning_rate'], per_device_train_batch_size=c['batch_size'],
        per_device_eval_batch_size=c['batch_size'], gradient_accumulation_steps=c['gradient_accumulation_steps'],
        warmup_ratio=c['warmup_ratio'], weight_decay=c['weight_decay'], lr_scheduler_type='linear',
        eval_strategy='epoch', save_strategy='epoch', logging_steps=1 if c['max_steps']>0 else 50,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss',
        seed=c['seed'], data_seed=c['seed'], fp16=gpu, use_cpu=not gpu,
        report_to='none', dataloader_num_workers=0, max_steps=c['max_steps'])
    trainer = Trainer(model=model, args=training_args, train_dataset=encoded['train'],
                      eval_dataset=encoded['validation'], processing_class=tokenizer,
                      callbacks=[EvidenceCallback(Path('logs')/Path(c['model_dir']).name)])
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=args.resume)
    elapsed = time.perf_counter()-started
    root = Path(c['model_dir']); root.mkdir(parents=True,exist_ok=True)
    trainer.save_model(root); tokenizer.save_pretrained(root)
    # A previous model's threshold must never survive retraining in the same directory.
    (root/'calibration.json').unlink(missing_ok=True)
    metadata = {**c, 'optimizer_steps': trainer.state.global_step, 'elapsed_seconds': elapsed,
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


if __name__ == '__main__': main()
