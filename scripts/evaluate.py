"""Calibrate on half of validation, report held-out half separately."""
import argparse
import json
from pathlib import Path
from qa.data import load_splits, subset, validation_partitions
from qa.inference import Engine
from qa.metrics import summarize, threshold_sweep, weighted_threshold_sweep, prior_weighted_metrics


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-dir',default='models/subset12k')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--objective',choices=['auto','balanced_f1','teacher_em'],default='auto')
    p.add_argument('--answerable-prior',type=float,default=None)
    args=p.parse_args()
    current_path=Path(args.model_dir)/'calibration.json'
    previous=json.loads(current_path.read_text()) if current_path.exists() else {}
    objective=previous.get('objective','balanced_f1') if args.objective=='auto' else args.objective
    prior=args.answerable_prior if args.answerable_prior is not None else previous.get('answerable_prior',.75)
    engine=Engine(args.model_dir, require_calibration=False)
    config=engine.metadata.copy()
    if config.get('data_manifest'):
        raise ValueError('Manifest-based continuation uses scripts.evaluate_continuation; never split its checkpoint-selection data for calibration')
    config['validation_size']=args.limit or config['validation_size']
    ds=load_splits(config)['validation']
    if 'validation_partition' in config:
        from datasets import concatenate_datasets
        parts=validation_partitions(ds,config)
        calibration=parts['calibration']
        ds=concatenate_datasets([parts['calibration'],parts['holdout']])
    else:
        calibration=subset(ds,len(ds)//2,config['seed']+1)
    ids=set(calibration['id'])
    rows=[]
    for i,ex in enumerate(ds):
        rows.append({**engine.candidate(ex['context'],ex['question']), 'id':ex['id'],
                     'references':ex['answers']['text'], 'calibration':ex['id'] in ids})
        if (i+1)%50==0: print(f'Evaluated {i+1}/{len(ds)}',flush=True)
    cal=[r for r in rows if r['calibration']]; held=[r for r in rows if not r['calibration']]
    if not cal or not held: raise ValueError('Need at least two validation examples')
    if objective=='teacher_em':
        results=weighted_threshold_sweep(cal,prior)
        old=previous.get('old_threshold',previous.get('threshold',0.))
        best=max(results,key=lambda r:(round(r['EM'],10),round(r['F1'],10),-abs(r['threshold']-old)))
    else:
        results=threshold_sweep(cal)
        old=previous.get('threshold',0.)
        best=max(results,key=lambda r:(r['F1'],r['EM'],-abs(r['threshold'])))
    result={'threshold':best['threshold'],'criterion':'prior-weighted calibration EM, F1 tie-break' if objective=='teacher_em' else 'calibration F1, EM tie-break',
            'objective':objective,'answerable_prior':prior,'old_threshold':old,
            'calibration':summarize(cal,best['threshold']), 'held_out':summarize(held,best['threshold']),
            'sensitivity':results, 'dataset':'rajpurkar/squad_v2 validation',
            'independent_of_checkpoint_selection':'validation_partition' in config,
            'calibration_ids':[r['id'] for r in cal], 'held_out_ids':[r['id'] for r in held]}
    result['weighted_75_25']=prior_weighted_metrics(result['held_out'],.75)
    root=Path(args.model_dir); (root/'calibration.json').write_text(json.dumps(result,indent=2))
    out=Path('logs')/root.name; out.mkdir(parents=True,exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    (out/'validation_predictions.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('sensitivity','calibration_ids','held_out_ids')},indent=2))


if __name__=='__main__': main()
