"""Freeze genuine SQuAD train partitions, excluding historical evaluation records."""
import random,re,unicodedata
from collections import Counter
from datasets import load_from_disk
from experiments.baseqa_common import ROOT,OUT,BASE,REVISION,read,sha,atomic
def norm(s):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).lower()))
def main():
    assert not (OUT/'data.json').exists(),'Prepared data already exists; do not resample'
    source={'repo_id':'microsoft/deberta-v3-base','revision':REVISION,'license':'MIT',
      'sha256':{name:sha(BASE/name) for name in ('config.json','tokenizer_config.json','spm.model','pytorch_model.bin','README.md')}}
    atomic(ROOT/'configs/deberta_base_source.json',source)
    oldids=set();contexts=set();questions=set();records=[]
    def visit(x):
        if isinstance(x,str) and re.fullmatch('[0-9a-f]{24}',x):oldids.add(x)
        elif isinstance(x,list):
            for y in x:visit(y)
        elif isinstance(x,dict):
            if isinstance(x.get('context'),str):contexts.add(norm(x['context']))
            if isinstance(x.get('question'),str):questions.add(norm(x['question']))
            for k,v in x.items():
                if k not in ('train_ids','added_train_ids','previous_completed_model_training'):visit(v)
    for name in ('configs/final500_manifest.json','logs/pretrained_benchmark/manifest.json','logs/quality20261029/manifest.json',
                 'logs/continued20k/data_manifest.json','logs/subset12k/data_manifest.json','logs/deberta_lora/pilot200/data.json','originals/public_sample.json'):
        visit(read(ROOT/name));records.append({'path':name,'sha256':sha(ROOT/name)})
    ds=load_from_disk(str(ROOT/'data/squad_v2'));integrity=Counter();seen_ids=set()
    for split in ds.values():
        for r in split:
            assert r['id'] not in seen_ids;seen_ids.add(r['id'])
            assert len(r['answers']['text'])==len(r['answers']['answer_start'])
            for text,start in zip(r['answers']['text'],r['answers']['answer_start']):assert text and r['context'][start:start+len(text)]==text
            integrity['records']+=1
            if r['id'] in oldids:contexts.add(norm(r['context']));questions.add(norm(r['question']))
    pool=[];seen_q=set();excluded=Counter()
    for r in ds['train']:
        c,q=norm(r['context']),norm(r['question'])
        if r['id'] in oldids or c in contexts or q in questions:excluded['historical_overlap']+=1;continue
        if q in seen_q:excluded['duplicate_question']+=1;continue
        if not (len(r['context'])<=5000 and len(r['question'])<=500):excluded['API_length']+=1;continue
        seen_q.add(q);pool.append(r)
    rng=random.Random(20261031);rng.shuffle(pool);used_contexts=set();data={}
    # Holdout is official train because all validation contexts had prior local exposure.
    for name,n in (('development',256),('calibration',256),('final',500)):
        rows=[];counts=Counter()
        for r in pool:
            c=norm(r['context']);cls=bool(r['answers']['text'])
            if c in used_contexts or counts[cls]>=n//2:continue
            rows.append(r);counts[cls]+=1;used_contexts.add(c)
            if len(rows)==n:break
        assert len(rows)==n;data[name]=rows
    available=[r for r in pool if norm(r['context']) not in used_contexts]
    ratio=sum(bool(r['answers']['text']) for r in ds['train'])/len(ds['train'])
    n=20000;yes=round(n*ratio)
    data['train']=[r for r in available if r['answers']['text']][:yes]+[r for r in available if not r['answers']['text']][:n-yes]
    rng.shuffle(data['train']);assert len(data['train'])==n
    for a in data:
        for b in data:
            if a!=b:
                assert not ({norm(r['context']) for r in data[a]}&{norm(r['context']) for r in data[b]})
                assert not ({norm(r['question']) for r in data[a]}&{norm(r['question']) for r in data[b]})
    atomic(OUT/'data.json',data)
    manifest={'seed':20261031,'dataset':'SQuAD 2.0 cached official train/validation','dataset_fingerprints':{k:v._fingerprint for k,v in ds.items()},
      'source_split':'train','data_sha256':sha(OUT/'data.json'),'integrity':dict(integrity),'excluded':dict(excluded),'historical_records':records,
      'splits':{k:{'count':len(v),'answerable':sum(bool(r['answers']['text']) for r in v),'ids':[r['id'] for r in v]} for k,v in data.items()},
      'limitations':'Context/question exact-normalized exclusions; not universal near-duplicate or pretraining independence. Stable SQuAD-specialized comparator was vendor-trained on official train; comparison may favor it.'}
    atomic(OUT/'manifest.json',manifest)
    compact={**manifest,'splits':{k:{x:y for x,y in v.items() if x!='ids'} for k,v in manifest['splits'].items()}}
    atomic(ROOT/'configs/deberta_base_data.json',compact)
    atomic(OUT/'protected_before.json',read(ROOT/'logs/deberta_lora/calibration_analysis/integrity_before.json'))
    print(compact)
if __name__=='__main__':main()
