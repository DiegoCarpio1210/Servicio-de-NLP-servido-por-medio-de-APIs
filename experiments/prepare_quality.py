"""Freeze partitions and experiment budget before inspecting candidate results."""
import hashlib
import json
from pathlib import Path
import numpy as np
from datasets import load_from_disk, concatenate_datasets
from qa.data import subset, stats

OUT=Path('logs/quality20261029')
def sha(text): return hashlib.sha256(text.encode()).hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():raise RuntimeError('Frozen manifest already exists')
    raw=load_from_disk('data/squad_v2'); train=raw['train']; val=raw['validation']
    prior=json.loads(Path('logs/continued20k/data_manifest.json').read_text())
    trained=set(prior['train_ids']); lineage={}
    for model in Path('models').iterdir():
        path=model/'training_metadata.json'
        if not path.exists():continue
        meta=json.loads(path.read_text()); manifest=Path(meta.get('data_manifest',str(Path('logs')/model.name/'data_manifest.json')))
        ids=json.loads(manifest.read_text())['train_ids'] if manifest.exists() else list(subset(train,meta['train_size'],meta['seed'])['id'])
        trained.update(ids);lineage[model.name]={'train_count':len(ids),'completed':meta.get('completed',True)}
    known_val=set()
    all_val=set(val['id'])
    def visit(x):
        if isinstance(x,str) and x in all_val:known_val.add(x)
        elif isinstance(x,dict):
            for v in x.values():visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    for folder in ('logs','models'):
        for path in Path(folder).rglob('*.json'):
            try:visit(json.loads(path.read_text(encoding='utf-8-sig')))
            except (ValueError,UnicodeError):pass
    used_val_contexts={sha(r['context']) for r in val if r['id'] in known_val}
    used_train_contexts={sha(r['context']) for r in train if r['id'] in trained}
    forbidden=used_train_contexts|{sha(c) for c in val['context']}
    groups={}
    for i,row in enumerate(train):
        key=sha(row['context'])
        if key not in forbidden:groups.setdefault(key,[]).append(i)
    rng=np.random.default_rng(20261029);keys=list(groups);rng.shuffle(keys)
    reserved=set();partitions={}
    for name in ('final','development'):
        chosen={False:[],True:[]};contexts=[]
        while min(map(len,chosen.values()))<512:
            if not keys:raise RuntimeError('Insufficient context-disjoint unused training data')
            key=keys.pop();contexts.append(key);reserved.add(key)
            for idx in groups[key]:chosen[bool(train[idx]['answers']['text'])].append(idx)
        selected=[]
        for flag in (False,True):selected.extend(rng.choice(chosen[flag],512,replace=False).tolist())
        rng.shuffle(selected);ds=train.select(selected)
        partitions[name]={'ids':list(ds['id']),'context_hashes':contexts,'stats':stats(ds)}
    previous_ids=set(prior['train_ids'])
    eligible=train.select([i for i,row in enumerate(train) if sha(row['context']) not in reserved and row['id'] not in previous_ids])
    added=subset(eligible,20000,20261030)
    lookup={id:i for i,id in enumerate(train['id'])}
    old=train.select([lookup[id] for id in prior['train_ids']])
    combined=concatenate_datasets([old,added]).shuffle(seed=42)
    assert not {sha(c) for c in combined['context']} & reserved
    manifest={'seed':20261029,'train_ids':list(combined['id']),'selection_ids':prior['selection_ids'],
      'calibration_ids':prior['calibration_ids'],'development_ids':partitions['development']['ids'],
      'final_ids':partitions['final']['ids'],'reserved_context_hashes':sorted(reserved),
      'partitions':partitions,'train_stats':stats(combined),'dataset_fingerprints':{k:v._fingerprint for k,v in raw.items()},
      'historical_validation_ids':sorted(known_val),'historical_validation_contexts':len(used_val_contexts),
      'remaining_context_disjoint_official_validation':sum(sha(c) not in used_val_contexts for c in val['context']),
      'previous_completed_model_training':lineage,
      'independence_limitations':['Prospectively reserved official TRAIN contexts, not the official validation split or instructor distribution.',
       'Excluded all contexts used by retained fine-tuned models and all official validation contexts.',
       'An aborted full-data run existed with no retained checkpoint; exact consumed IDs are unavailable. It is not an ancestor of the compared models. Thus universal historical non-exposure cannot be certified.'],
      'role_rule':'Never gradient-train on reserved development/final contexts. Fit thresholds on original 512 calibration rows only. Select on development; final read once after lock.'}
    dump(OUT/'manifest.json',manifest)
    c=json.loads(Path('configs/continued20k.json').read_text())
    c.update(train_size=40000,epochs=1,learning_rate=2e-5,warmup_ratio=.05,output_dir='checkpoints/quality40k',model_dir='models/quality40k',data_manifest=str(OUT/'manifest.json'),warm_start='models/continued20k',warm_start_sha256='7d66ffcff40157ed8f9f728f51ffe3c1b5b3f249ac8c100ece9666d1f7007848',max_training_seconds=3600,eval_steps=400,save_steps=200)
    dump(Path('configs/quality40k.json'),c)
    plan={'stable':'models/continued20k','stable_sha256':c['warm_start_sha256'],
      'postprocessors':[{'name':'standard','max_length':30,'length_penalty':0.0},{'name':'short15','max_length':15,'length_penalty':0.0},{'name':'length01','max_length':30,'length_penalty':.1},{'name':'length025','max_length':30,'length_penalty':.25},{'name':'length05','max_length':30,'length_penalty':.5}],
      'selection_rule':'Choose each model postprocessor by development 75/25 EM among profiles with development impossible EM >=70%, F1 tie-break. Threshold is fitted only on calibration with >=70% impossible EM. Final compare stable, stable experimental and one trained candidate; never retune after final.',
      'training_authorized_candidates':1,'training_budget_seconds':3600,'estimated_training_minutes':[42,60],
      'estimate_basis':'2579 seconds / 2534 optimizer steps in previous real GPU run; ~2500 new steps; previous peak reserved 2798 MiB of 4096.',
      'training_trigger':'If inference-only development metrics do not meet 65 EM / 70 F1 / 50 answerable EM / 70 impossible EM, run one 40k/1epoch continuation.',
      'promotion_rule':'No automatic replacement. Recommend only if paired context-cluster bootstrap 95% CI for weighted EM gain excludes zero and impossible EM >=70%; otherwise keep stable.',
      'public_and_manual':'Diagnostics only after policy selection; never calibrate or train on these examples.'}
    dump(OUT/'plan.json',plan)
    print(json.dumps({'train':manifest['train_stats'],'reserved':{k:v['stats'] for k,v in partitions.items()},'unused_official_validation_contexts':manifest['remaining_context_disjoint_official_validation'],'lineage':lineage},indent=2))

if __name__=='__main__':main()
