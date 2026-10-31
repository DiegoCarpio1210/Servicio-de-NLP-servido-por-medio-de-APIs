"""Frozen 500-question comparison. No fitting, downloads, training or activation."""
import argparse,csv,hashlib,json,random,re,socket,threading,time,unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from experiments.run_pretrained_benchmark import atomic,read,sha,measure,BenchmarkEngine

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/final500'
MODELS=['deberta-v3-base-squad2','roberta-base-squad2','quality40k']

def qnorm(s):return re.sub(r'[^\w\s]','',unicodedata.normalize('NFKC',s).lower()).strip()
def h(s):return hashlib.sha256(s.encode()).hexdigest()

def prepare():
    from datasets import load_from_disk
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():raise RuntimeError('Manifest already frozen; use the existing one')
    ds=load_from_disk(str(ROOT/'data/squad_v2'))['validation'];lookup={r['id']:r for r in ds}
    known=set();extra_questions=set();extra_contexts=set();sources=[]
    def visit(x):
        if isinstance(x,str):
            if x in lookup:known.add(x)
        elif isinstance(x,dict):
            if isinstance(x.get('question'),str):extra_questions.add(qnorm(x['question']))
            if isinstance(x.get('context'),str):extra_contexts.add(h(x['context']))
            for v in x.values():visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    paths=[p for folder in ('logs','models') for p in (ROOT/folder).rglob('*.json') if OUT not in p.parents]
    paths+=[ROOT/'originals/public_sample.json',ROOT/'experiments/iss_diagnostic.json']
    for path in paths:
        try:
            raw=path.read_bytes();visit(json.loads(raw.decode('utf-8-sig')))
            sources.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest()})
        except (UnicodeError,ValueError):continue
    questions={qnorm(lookup[id]['question']) for id in known}|extra_questions
    contexts={h(lookup[id]['context']) for id in known}|extra_contexts
    eligible=[r for r in ds if r['id'] not in known and qnorm(r['question']) not in questions]
    rng=random.Random(50020261030);rng.shuffle(eligible)
    eligible.sort(key=lambda r:h(r['context']) in contexts) # prefer novel paragraphs, if any
    selected=[];counts={False:0,True:0};selected_contexts=set();selected_questions=set()
    for row in eligible:
        flag=bool(row['answers']['text']);context=h(row['context']);question=qnorm(row['question'])
        if counts[flag]==250 or context in selected_contexts or question in selected_questions:continue
        selected.append(row);counts[flag]+=1;selected_contexts.add(context);selected_questions.add(question)
    if counts!={False:250,True:250}:raise RuntimeError(f'Insufficient question/context-diverse pool: {counts}')
    rng.shuffle(selected);profiles={}
    for key in MODELS:
        prior=ROOT/'logs/pretrained_benchmark'/key;policy=read(prior/'policy.json');profile=read(prior/'metrics.json')['primary_profile']
        folder='models/quality40k' if key=='quality40k' else 'models/pretrained/'+key
        assert sha(ROOT/folder/'model.safetensors')==policy['weights_sha256']
        profiles[key]={'folder':folder,'profile':profile,'threshold':policy['thresholds'][profile],'policy':policy['policy'],'weights_sha256':policy['weights_sha256'],'source_policy_sha256':sha(prior/'policy.json')}
    audit={'historical_validation_ids':len(known),'scanned_json_files':len(sources),'eligible_unseen_id_and_normalized_question_count':len(eligible),
      'selected_id_overlap':sum(r['id'] in known for r in selected),'selected_normalized_question_overlap':sum(qnorm(r['question']) in questions for r in selected),
      'selected_context_overlap':sum(h(r['context']) in contexts for r in selected),'distinct_selected_contexts':len(selected_contexts),
      'unused_context_count_in_eligible_pool':len({h(r['context']) for r in eligible if h(r['context']) not in contexts})}
    manifest={'seed':50020261030,'split':'validation','fingerprint':ds._fingerprint,'ids':[r['id'] for r in selected],
      'answerable':250,'unanswerable':250,'audit':audit,'profiles':profiles,
      'protocol':'Existing primary profiles only. Secondary null95 profiles are NOT selected using this test. No recalibration.',
      'limitations':['New IDs/questions relative to available local JSON records, not guaranteed universal non-exposure.',
        'Contexts have historical local exposure; vendor model selection used SQuAD validation. Not an independent official test.',
        'Abandoned full-data run consumed IDs are unknown; SQuAD pretrained models saw the training split, so only validation is used.',
        'Prior recorded metrics use different examples and are descriptive comparisons, not paired gains.']}
    atomic(OUT/'manifest.json',manifest);atomic(OUT/'audit_sources.json',sources)
    atomic(OUT/'progress.json',{'stage':'prepared','count':500});print(json.dumps(audit,indent=2))

def api_checks(engine,out,null_example):
    import httpx,secrets,uvicorn,yaml
    from jsonschema import validate
    from qa.api import create_app
    from scripts.http_contract_checks import check
    import os
    with socket.socket() as probe:probe.bind(('127.0.0.1',8000))
    token=secrets.token_urlsafe(32);os.environ['QA_API_TOKEN']=token
    server=uvicorn.Server(uvicorn.Config(create_app(loader=lambda:engine),host='127.0.0.1',port=8000,log_level='error'))
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    schema=yaml.safe_load((ROOT/'originals/qa_api_contract.yaml').read_text(encoding='utf-8'))['components']['schemas']
    headers={'Authorization':'Bearer '+token};result={}
    try:
        with httpx.Client(base_url='http://127.0.0.1:8000',timeout=5,trust_env=False) as client:
            for _ in range(100):
                try:
                    health=client.get('/health')
                    if health.status_code==200:break
                except httpx.HTTPError:pass
                time.sleep(.1)
            else:raise TimeoutError('API not ready')
            validate(health.json(),schema['Health']);info=client.get('/info',headers=headers);info.raise_for_status();validate(info.json(),schema['Info'])
            assert info.json()['checkpoint_sha256']==engine.weights_sha256
            assert info.json()['no_answer_threshold']==engine.threshold
            result['info']=info.json();check(client,token,out)
            null=client.post('/predict',headers=headers,json={k:null_example[k] for k in ('id','context','question')});null.raise_for_status()
            validate(null.json(),schema['PredictResponse']);assert null.json()['answer'] is None and null.json()['start'] is None
            result['null_check']=null.json();result['contract_pass']=True
        items=read(ROOT/'originals/public_sample.json')['items']
        def request(i,deadline):
            item=items[i%5];start=time.perf_counter();row={'index':i,'id':item['id']}
            try:
                if start>=deadline:raise TimeoutError('Global deadline')
                r=httpx.post('http://127.0.0.1:8000/predict',headers=headers,json={k:item[k] for k in ('id','context','question')},timeout=min(5,deadline-start),trust_env=False)
                row['status']=r.status_code;row['body']=r.json();r.raise_for_status();body=row['body'];validate(body,schema['PredictResponse'])
                assert body['id']==item['id'] and set(body)=={'id','answer','start','score'}
                assert (body['answer'] is None)==(body['start'] is None)
                if body['answer'] is not None:assert item['context'][body['start']:body['start']+len(body['answer'])]==body['answer']
                row['valid']=True
            except Exception as exc:row.update(valid=False,error=type(exc).__name__)
            row['seconds']=time.perf_counter()-start;return row
        for name,n in [('sequential',1),('concurrent',2)]:
            started=time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as pool:rows=list(pool.map(lambda i:request(i,started+150),range(50)))
            total=time.perf_counter()-started
            result[name]={'requests':50,'concurrency':n,'seconds':total,'maximum_seconds':max(r['seconds'] for r in rows),
              'valid':sum(r['valid'] for r in rows),'pass':total<=150 and all(r['valid'] and r['seconds']<=5 for r in rows),'rows':rows}
    finally:
        server.should_exit=True;thread.join(timeout=15);os.environ.pop('QA_API_TOKEN',None)
        result['server_stopped']=not thread.is_alive()
    return result

def evaluate(key):
    import psutil,numpy as np
    from datasets import load_from_disk
    from qa.spans import decide
    from qa.metrics import score
    manifest=read(OUT/'manifest.json');p=manifest['profiles'][key];out=OUT/key;out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():print('Completed; skipping '+key);return
    engine=BenchmarkEngine(ROOT/p['folder'],p['policy'],2);assert engine.weights_sha256==p['weights_sha256'];engine.threshold=p['threshold']
    engine.metadata=read(ROOT/'models/quality40k/training_metadata.json') if key=='quality40k' else {'base_model':'deepset/'+key,'optimizer_steps':0}
    ds=load_from_disk(str(ROOT/'data/squad_v2'))['validation'];lookup={r['id']:r for r in ds}
    engine.predict('Paris is in France.','Where is Paris?') # untimed warmup
    target=out/'predictions.json';rows=read(target) if target.exists() else []
    assert [r['id'] for r in rows]==manifest['ids'][:len(rows)]
    for i in range(len(rows),500):
        ex=lookup[manifest['ids'][i]];begin=time.perf_counter();candidate=engine.candidate(ex['context'],ex['question']);response=decide(candidate,engine.threshold);elapsed=time.perf_counter()-begin
        if response['answer'] is not None:assert ex['context'][response['start']:response['start']+len(response['answer'])]==response['answer']
        em,f1=score(response['answer'] or '',ex['answers']['text'])
        rows.append({'id':ex['id'],'question':ex['question'],'context_hash':h(ex['context']),'references':ex['answers']['text'],
          'expected_answer_starts':ex['answers']['answer_start'],**candidate,'response':response,'EM':100*em,'F1':100*f1,'seconds':elapsed,'offset_valid':True})
        if (i+1)%25==0:
            atomic(target,rows);atomic(OUT/'progress.json',{'stage':'inference','model':key,'completed':i+1,'total':500})
            print(f'{key}: {i+1}/500',flush=True)
    null=next(lookup[r['id']] for r in rows if not r['references'] and r['response']['answer'] is None)
    http=api_checks(engine,out,null);atomic(out/'http.json',http)
    memory=psutil.Process().memory_info();metrics=measure(rows,engine.threshold)
    targets={'overall_EM_85':metrics['groups']['all']['EM']>=85,'overall_F1_85':metrics['groups']['all']['F1']>=85,
      'answerable_EM_80':metrics['groups']['answerable']['EM']>=80,'unanswerable_EM_95':metrics['groups']['unanswerable']['EM']>=95}
    atomic(out/'summary.json',{'metrics':metrics,'targets':targets,'all_targets':all(targets.values()),'profile':p,
      'latency':{'mean_seconds':float(np.mean([r['seconds'] for r in rows])),'p95_seconds':float(np.percentile([r['seconds'] for r in rows],95)),'max_seconds':max(r['seconds'] for r in rows)},
      'peak_rss_mib':getattr(memory,'peak_wset',memory.rss)/2**20,'http':{k:({a:b for a,b in v.items() if a!='rows'} if isinstance(v,dict) else v) for k,v in http.items()},
      'all_offsets_valid':True,'manifest_sha256':sha(OUT/'manifest.json')})
    print('Completed '+key,flush=True)

def main():
    import os
    os.chdir(ROOT)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare',*MODELS]);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:evaluate(a.stage)

if __name__=='__main__':main()
