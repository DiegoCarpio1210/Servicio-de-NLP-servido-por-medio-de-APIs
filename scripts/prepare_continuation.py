"""Prepare one nested training experiment; never inspect final evaluation predictions."""
import json
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets
from qa.data import subset, stats


def main():
    out=Path('logs/continued20k'); out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists(): raise RuntimeError('Experiment already prepared; do not replace its plan')
    old=json.loads(Path('logs/subset12k/data_manifest.json').read_text())
    raw=load_from_disk('data/squad_v2'); train=raw['train']; val=raw['validation']
    known=set(old['train_ids'])
    previous=train.select([i for i,item in enumerate(train['id']) if item in known])
    remaining=train.select([i for i,item in enumerate(train['id']) if item not in known])
    added=subset(remaining,8000,20261026)
    combined=concatenate_datasets([previous,added]).shuffle(seed=42)
    # Exclude all recorded historical validation uses, including pilots, plus their contexts.
    used=set(old['selection_ids']+old['calibration_ids']+old['holdout_ids'])
    for path in Path('logs').rglob('validation_predictions.json'):
        for row in json.loads(path.read_text()): used.add(row['id'])
    for path in Path('logs').rglob('evaluation.json'):
        obj=json.loads(path.read_text())
        used.update(obj.get('calibration_ids',[])); used.update(obj.get('held_out_ids',[]))
    used_contexts={row['context'] for row in val if row['id'] in used}
    unused=val.select([i for i,row in enumerate(val) if row['id'] not in used and row['context'] not in used_contexts])
    assert len(unused)>=1024
    fresh=subset(unused,1024,20261026)
    manifest={**old,'train_ids':list(combined['id']),'added_train_ids':list(added['id']),
              'fresh_holdout_ids':list(fresh['id']),'train_stats':stats(combined),
              'historically_used_validation_ids':sorted(used),'fresh_holdout_stats':stats(fresh)}
    assert len(set(manifest['train_ids']))==20000 and not known&set(added['id'])
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    c=json.loads(Path('configs/subset12k.json').read_text()); c.pop('validation_partition')
    c.update(train_size=20000,validation_size=512,epochs=2,learning_rate=1e-5,warmup_ratio=.05,
             output_dir='checkpoints/continued20k',model_dir='models/continued20k',
             data_manifest=str(out/'data_manifest.json'),warm_start='models/subset12k',
             warm_start_sha256='60ff15f44cc8bf2bd0a98fceae9a594f536874ae957d76279b86b308604ab1d4',
             max_training_seconds=3300)
    Path('configs/continued20k.json').write_text(json.dumps(c,indent=2))
    plan={'authorization':'One controlled continuation authorized in user attached request',
          'config':c,'estimated_training_minutes':[44,60],'measured_baseline_seconds':1582.5165039002895,
          'time_guard_seconds':3300,'new_examples':8000,'new_holdout_count':1024,
          'decision_rule':'On 512 calibration examples maximize 75/25 weighted EM subject to impossible EM >=70%; weighted F1 tie-break. Compare original, recalibrated, and continued weights on calibration only; select highest weighted EM among eligible profiles before any final evaluation.',
          'null_floor_reason':'Predeclared safeguard against the previously observed 33.53% correct-null collapse; not fitted to the new holdout.',
          'checkpoint_rule':'Minimum loss on original 512 selection examples',
          'evaluation_rule':'Lock candidate and thresholds before evaluating old 1024 and fresh 1024; public examples last; no post-evaluation tuning',
          'fresh_holdout_excludes_previously_used_contexts':True,'training_runs_authorized':1}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({'train':stats(combined),'added':stats(added),'fresh':stats(fresh),'estimate_minutes':[44,60]},indent=2))


if __name__=='__main__': main()
