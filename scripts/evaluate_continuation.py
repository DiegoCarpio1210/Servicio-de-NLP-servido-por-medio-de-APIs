"""Fit only on calibration, lock a candidate, then report fixed final comparisons."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from datasets import load_from_disk
from qa.inference import Engine
from qa.metrics import prior_weighted_metrics, summarize, weighted_threshold_sweep, score

ROOT=Path('logs/continued20k')


def dump(path, obj):
    path.write_text(json.dumps(obj,indent=2),encoding='utf-8')


def candidates(engine, dataset, ids, label):
    lookup={row['id']:row for row in dataset if row['id'] in set(ids)}
    rows=[]
    for i,item in enumerate(ids):
        ex=lookup[item]
        rows.append({**engine.candidate(ex['context'],ex['question']),'id':item,
                     'references':ex['answers']['text'],'calibration':label=='calibration'})
        if (i+1)%100==0: print(f'{label}: {i+1}/{len(ids)}',flush=True)
    return rows


def metrics(rows,threshold):
    result=summarize(rows,threshold)
    return {'groups':result,'weighted_75_25':prior_weighted_metrics(result)}


def calibrate():
    if (ROOT/'locked_candidate.json').exists(): raise RuntimeError('Candidate already locked')
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    old=json.loads(Path('logs/recalibration/locked_policy.json').read_text())
    baseline=[r for r in json.loads(Path('logs/subset12k/validation_predictions.json').read_text()) if r['calibration']]
    assert {r['id'] for r in baseline}==set(manifest['calibration_ids'])
    engine=Engine('models/continued20k',require_calibration=False)
    ds=load_from_disk('data/squad_v2')['validation']
    rows=candidates(engine,ds,manifest['calibration_ids'],'calibration')
    dump(ROOT/'calibration_predictions.json',rows)
    sweep=weighted_threshold_sweep(rows,.75)
    feasible=[p for p in sweep if summarize(rows,p['threshold'])['unanswerable']['EM']>=70.]
    best=max(feasible,key=lambda p:(round(p['EM'],10),round(p['F1'],10),-abs(p['threshold'])))
    threshold=best['threshold']
    profiles={
        'original':{'model_dir':'models/subset12k','threshold':old['old_threshold'],**metrics(baseline,old['old_threshold'])},
        'recalibrated':{'model_dir':'models/subset12k','threshold':old['threshold'],**metrics(baseline,old['threshold'])},
        'continued':{'model_dir':'models/continued20k','threshold':threshold,**metrics(rows,threshold)}}
    eligible={k:v for k,v in profiles.items() if v['groups']['unanswerable']['EM']>=70.}
    selected=max(eligible,key=lambda k:(round(eligible[k]['weighted_75_25']['EM'],10),round(eligible[k]['weighted_75_25']['F1'],10)))
    lock={'locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selected':selected,
          'selection_partition':'512 calibration examples only','profiles':profiles,
          'decision_rule':json.loads((ROOT/'plan.json').read_text())['decision_rule'],
          'continued_weights_sha256':engine.weights_sha256,'baseline_weights_sha256':old['weights_sha256'],
          'heldout_consulted_for_selection':False,'public_consulted_for_selection':False}
    dump(ROOT/'calibration_sweep.json',sweep)
    dump(Path('models/continued20k/calibration.json'),{'threshold':threshold,'objective':'teacher_em_null_floor70',
         'answerable_prior':.75,'unanswerable_calibration_floor':70.,'calibration_ids':manifest['calibration_ids'],
         'calibration':profiles['continued']['groups']})
    dump(ROOT/'locked_candidate.json',lock)
    print(json.dumps(lock,indent=2),flush=True)


def evaluate():
    lockpath=ROOT/'locked_candidate.json'; locked_bytes=lockpath.read_bytes(); lock=json.loads(locked_bytes)
    if (ROOT/'comparison.json').exists(): raise RuntimeError('Final comparison already exists; do not rerun to select settings')
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    ds=load_from_disk('data/squad_v2')['validation']; results={}
    started=time.time()
    # Baseline old holdout needs no repeated forward pass; fresh evaluation does.
    baseline_rows=[r for r in json.loads(Path('logs/subset12k/validation_predictions.json').read_text()) if not r['calibration']]
    base=Engine('models/subset12k'); assert base.weights_sha256==lock['baseline_weights_sha256']
    fresh_base=candidates(base,ds,manifest['fresh_holdout_ids'],'fresh baseline')
    dump(ROOT/'baseline_fresh_predictions.json',fresh_base)
    del base
    improved=Engine('models/continued20k'); assert improved.weights_sha256==lock['continued_weights_sha256']
    old_new=candidates(improved,ds,manifest['holdout_ids'],'old holdout continued')
    fresh_new=candidates(improved,ds,manifest['fresh_holdout_ids'],'fresh continued')
    dump(ROOT/'validation_predictions.json',old_new)
    dump(ROOT/'fresh_predictions.json',fresh_new)
    del improved
    for name,profile in lock['profiles'].items():
        rows=old_new if name=='continued' else baseline_rows
        fresh=fresh_new if name=='continued' else fresh_base
        results[name]={'previously_consulted_holdout':metrics(rows,profile['threshold']),
                       'fresh_holdout':metrics(fresh,profile['threshold'])}
    # Fixed paired uncertainty estimate, never used to change the frozen selection.
    rng=np.random.default_rng(42); intervals={}
    assert [r['id'] for r in fresh_base]==[r['id'] for r in fresh_new]
    def em(row,threshold):
        return score('' if row['margin']>threshold else row['text'],row['references'])[0]
    for name in ('original','recalibrated'):
        delta={flag:np.array([em(new,lock['profiles']['continued']['threshold'])-em(old,lock['profiles'][name]['threshold'])
                             for old,new in zip(fresh_base,fresh_new) if bool(new['references'])==flag]) for flag in (True,False)}
        samples=[100*(.75*rng.choice(delta[True],len(delta[True]),replace=True).mean()+
                      .25*rng.choice(delta[False],len(delta[False]),replace=True).mean()) for _ in range(2000)]
        intervals['continued_minus_'+name]=np.percentile(samples,[2.5,97.5]).tolist()
    assert lockpath.read_bytes()==locked_bytes
    result={'selected_before_final_evaluation':lock['selected'],'profiles':results,
            'lock_sha256':hashlib.sha256(locked_bytes).hexdigest(),'elapsed_seconds':time.time()-started,
            'old_holdout_previously_consulted':True,'fresh_holdout_previously_unused':True,
            'fresh_weighted_EM_gain_95pct_bootstrap_intervals':intervals,
            'candidate_selection_not_revised_after_evaluation':True}
    dump(ROOT/'comparison.json',result)
    # Compatibility evidence for the existing HTTP verifier; no calibration is refitted here.
    dump(ROOT/'evaluation.json',{'threshold':lock['profiles']['continued']['threshold'],
         'objective':'teacher_em_null_floor70','held_out':results['continued']['previously_consulted_holdout']['groups'],
         'fresh_holdout':results['continued']['fresh_holdout']['groups'],
         'weighted_75_25':results['continued']['fresh_holdout']['weighted_75_25']})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['calibrate','evaluate']); a=p.parse_args()
    calibrate() if a.stage=='calibrate' else evaluate()
