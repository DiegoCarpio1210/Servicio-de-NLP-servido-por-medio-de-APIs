"""Freeze local authored cohorts and audit recorded exact/near overlap; no inference."""
import hashlib,json,re,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
from datasets import load_from_disk

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'logs/semantic50'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def norm(s):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).lower()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def grams(s,n):
    words=norm(s).split();return set(zip(*(words[i:] for i in range(n))))

def main():
    if (ROOT/'configs/semantic50_protocol.json').exists():raise RuntimeError('Cohorts already frozen')
    ids=set();contexts=set();questions=set();sources=[]
    def visit(x):
        if isinstance(x,str):
            if re.fullmatch(r'[0-9a-f]{24}',x):ids.add(x)
        elif isinstance(x,list):
            for v in x:visit(v)
        elif isinstance(x,dict):
            if isinstance(x.get('context'),str):contexts.add(x['context'])
            if isinstance(x.get('question'),str):questions.add(x['question'])
            for k,v in x.items():
                if k not in ('train_ids','added_train_ids','previous_completed_model_training'):visit(v)
    paths=['configs/final500_manifest.json','logs/pretrained_benchmark/manifest.json','logs/quality20261029/manifest.json',
      'logs/continued20k/data_manifest.json','logs/subset12k/data_manifest.json','logs/deberta_lora/pilot200/data.json','originals/public_sample.json']
    for name in paths:
        p=ROOT/name;visit(read(p));sources.append({'path':name,'sha256':sha(p)})
    ds=load_from_disk(str(ROOT/'data/squad_v2'))
    for split in ds.values():
        for row in split:
            if row['id'] in ids:contexts.add(row['context']);questions.add(row['question'])
    audits={};cohorts={name:read(OUT/(name+'.json')) for name in ('development','final')}
    assert not ({norm(r['context']) for r in cohorts['development']}&{norm(r['context']) for r in cohorts['final']})
    assert not ({norm(r['question']) for r in cohorts['development']}&{norm(r['question']) for r in cohorts['final']})
    def overlap(values,old,n):
        old_norm={norm(x) for x in old};old_grams=[grams(x,n) for x in old];index=defaultdict(set)
        for i,g in enumerate(old_grams):
            for token in g:index[token].add(i)
        matches=[]
        for value in set(values):
            g=grams(value,n);candidates=set().union(*(index.get(t,set()) for t in g)) if g else set()
            for i in candidates:
                union=g|old_grams[i];similarity=len(g&old_grams[i])/len(union) if union else 0.
                if similarity>=.8:matches.append({'text_sha256':hashlib.sha256(value.encode()).hexdigest(),'jaccard':similarity})
        return {'exact_normalized_overlap':sum(norm(x) in old_norm for x in set(values)),
          'near_duplicate_matches':matches,'word_shingle_size':n,'jaccard_threshold':.8}
    for name,rows in cohorts.items():
        assert len(rows)==50 and len({r['id'] for r in rows})==50
        for r in rows:
            assert 1<=len(r['context'])<=5000 and 1<=len(r['question'])<=500
            assert all(r['context'][a:a+len(t)]==t for t,a in zip(r['answers']['text'],r['answers']['answer_start']))
        audit={'questions':overlap([r['question'] for r in rows],questions,3),'contexts':overlap([r['context'] for r in rows],contexts,5)}
        assert all(x['exact_normalized_overlap']==0 and not x['near_duplicate_matches'] for x in audit.values()),audit
        audits[name]={'count':50,'answerable':sum(bool(r['answers']['text']) for r in rows),'unanswerable':sum(not r['answers']['text'] for r in rows),
          'contexts':len({r['context_id'] for r in rows}),'max_context_characters':max(len(r['context']) for r in rows),
          'categories':dict(Counter(r['category'] for r in rows)),'sha256':sha(OUT/(name+'.json')),'overlap':audit}
    protocol={'seed':20261031,'model':'deepset/deberta-v3-base-squad2','production_threshold':2.7772940397262573,
      'cohorts':audits,'historical_audit':{'records':sources,'contexts':len(contexts),'questions':len(questions)},
      'modes':{'sequential':1,'bounded_concurrent':2},'requests_per_cohort_per_mode':50,'request_limit_seconds':5,'total_limit_seconds':150,
      'provenance':'Assistant-authored fictional contexts/questions, frozen before inference; not an independently authored or vendor-audited benchmark',
      'semantic_grading':'Claude unavailable; independent grading export, no invented semantic labels',
      'final_rule':'Freeze selected policy after development analysis; run final accuracy once. Concurrent replay tests reliability, not model selection.'}
    (ROOT/'configs/semantic50_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audits,indent=2))

if __name__=='__main__':main()
