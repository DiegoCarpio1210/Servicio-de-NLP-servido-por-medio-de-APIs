"""One preregistered comparison: calibration/development first, locked final evaluation last."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from datasets import load_from_disk
from qa.metrics import score
from experiments.inference import ResearchEngine,candidate,metrics,fit

ROOT=Path('logs/quality20261029')
def dump(name,obj):(ROOT/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def load(name):return json.loads((ROOT/name).read_text())


def collect(engine,ds,ids,policies,label):
    lookup={id:i for i,id in enumerate(ds['id'])};rows={p['name']:[] for p in policies};started=time.perf_counter()
    for i,id in enumerate(ids):
        ex=ds[lookup[id]];windows=engine.windows(ex['context'],ex['question'])
        for p in policies:
            item=candidate(ex['context'],windows,p)
            if item['start'] is not None:assert ex['context'][item['start']:item['start']+len(item['text'])]==item['text']
            rows[p['name']].append({**item,'id':id,'references':ex['answers']['text'],'context_hash':hashlib.sha256(ex['context'].encode()).hexdigest()})
        if (i+1)%128==0:print(f'{label}: {i+1}/{len(ids)} ({time.perf_counter()-started:.1f}s)',flush=True)
    return rows


def select(model_dir,label):
    if (ROOT/f'{label}_selection.json').exists():raise RuntimeError('Selection already completed; do not overwrite')
    manifest=load('manifest.json');plan=load('plan.json');raw=load_from_disk('data/squad_v2')
    engine=ResearchEngine(model_dir,'cuda' if torch.cuda.is_available() else 'cpu')
    policies=plan['postprocessors']
    cal=collect(engine,raw['validation'],manifest['calibration_ids'],policies,label+' calibration')
    dev=collect(engine,raw['train'],manifest['development_ids'],policies,label+' development')
    profiles={}
    for p in policies:
        name=p['name'];threshold=fit(cal[name])
        profiles[name]={'model_dir':model_dir,'weights_sha256':engine.sha,'policy':p,'threshold':threshold,'calibration':metrics(cal[name],threshold),'development':metrics(dev[name],threshold)}
    if label=='stable':
        threshold=json.loads(Path('models/continued20k/calibration.json').read_text())['threshold']
        profiles['production']={**profiles['standard'],'threshold':threshold,'calibration':metrics(cal['standard'],threshold),'development':metrics(dev['standard'],threshold)}
    feasible={k:v for k,v in profiles.items() if v['development']['groups']['unanswerable']['EM']>=70.}
    selected=max(feasible,key=lambda k:(round(feasible[k]['development']['weighted_75_25']['EM'],10),round(feasible[k]['development']['weighted_75_25']['F1'],10))) if feasible else 'standard'
    dump(f'{label}_calibration_predictions.json',cal);dump(f'{label}_development_predictions.json',dev)
    result={'selected':selected,'profiles':profiles,'device':engine.device,'final_not_consulted':True}
    dump(f'{label}_selection.json',result)
    print(json.dumps({'label':label,'selected':selected,'profiles':{k:v['development'] for k,v in profiles.items()}},indent=2),flush=True)


def lock():
    if (ROOT/'final_lock.json').exists():raise RuntimeError('Final already locked')
    stable=load('stable_selection.json');trained=load('trained_selection.json')
    profiles={'production':stable['profiles']['production'],'inference_only':stable['profiles'][stable['selected']], 'trained':trained['profiles'][trained['selected']]}
    dump('final_lock.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'profiles':profiles,'manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),'no_final_predictions_seen':True,'no_public_or_manual_selection':True})


def bootstrap(a,b,ta,tb):
    # Paired resampling of context clusters respects correlation among questions in a paragraph.
    assert [r['id'] for r in a]==[r['id'] for r in b]
    groups={}
    for x,y in zip(a,b):
        flag=bool(x['references']);v=score('' if x['margin']>ta else x['text'],x['references'])[0]-score('' if y['margin']>tb else y['text'],y['references'])[0]
        g=groups.setdefault(x['context_hash'],np.zeros(4));g[0 if flag else 2]+=v;g[1 if flag else 3]+=1
    blocks=np.array(list(groups.values()));rng=np.random.default_rng(42);values=[]
    for _ in range(3000):
        totals=blocks[rng.integers(0,len(blocks),len(blocks))].sum(axis=0)
        if totals[1] and totals[3]:values.append(100*(.75*totals[0]/totals[1]+.25*totals[2]/totals[3]))
    return np.percentile(values,[2.5,97.5]).tolist()


def final():
    if (ROOT/'final_comparison.json').exists():raise RuntimeError('Final already scored; no retuning')
    frozen=load('final_lock.json');manifest=load('manifest.json')
    assert hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()==frozen['manifest_sha256']
    ds=load_from_disk('data/squad_v2')['train'];rows={};results={}
    for model in sorted({p['model_dir'] for p in frozen['profiles'].values()}):
        profiles={k:p for k,p in frozen['profiles'].items() if p['model_dir']==model}
        engine=ResearchEngine(model,'cuda' if torch.cuda.is_available() else 'cpu')
        assert all(p['weights_sha256']==engine.sha for p in profiles.values())
        policies=list({p['policy']['name']:p['policy'] for p in profiles.values()}.values())
        candidates=collect(engine,ds,manifest['final_ids'],policies,model+' final')
        for name,p in profiles.items():
            rows[name]=candidates[p['policy']['name']];results[name]=metrics(rows[name],p['threshold'])
        del engine
        if torch.cuda.is_available():torch.cuda.empty_cache()
    intervals={name:bootstrap(rows[name],rows['production'],p['threshold'],frozen['profiles']['production']['threshold']) for name,p in frozen['profiles'].items() if name!='production'}
    result={'profiles':results,'weighted_EM_gain_vs_production_95pct_context_bootstrap':intervals,'final_count':len(manifest['final_ids']),
      'provenance':'Prospectively frozen context-disjoint official train holdout; no retained model was trained on its contexts. See manifest limitations for abandoned full-data run.',
      'historical_metrics_not_directly_comparable':True,'no_post_final_tuning':True,'production_unchanged':True}
    dump('final_predictions.json',rows);dump('final_comparison.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['stable','trained','lock','final']);a=p.parse_args()
    if a.stage in ('stable','trained'):select('models/continued20k' if a.stage=='stable' else 'models/quality40k',a.stage)
    elif a.stage=='lock':lock()
    else:final()
