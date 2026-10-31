"""Exactly 50 evaluated HTTP requests per cohort/mode; all model calls go through FastAPI."""
import argparse,csv,hashlib,json,math,os,secrets,socket,threading,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx,yaml,uvicorn
from jsonschema import validate
from qa.api import create_app
from qa.deployment import load_engine
from qa.metrics import score

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'logs/semantic50'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(context,question):return hashlib.sha256((context+'\0'+question).encode()).hexdigest()
def check_response(body,item,schema):
    validate(body,schema);assert body['id']==item['id']
    assert math.isfinite(body['score'])
    assert (body['answer'] is None)==(body['start'] is None)
    if body['answer'] is not None:assert item['context'][body['start']:body['start']+len(body['answer'])]==body['answer']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('cohort',choices=['development','final']);args=parser.parse_args()
    os.chdir(ROOT);protocol=read(ROOT/'configs/semantic50_protocol.json');path=OUT/(args.cohort+'.json')
    assert sha(path)==protocol['cohorts'][args.cohort]['sha256']
    target=OUT/(args.cohort+'_results.json')
    if target.exists():raise RuntimeError('Completed cohort cannot be rerun or overwritten')
    if args.cohort=='final':
        lock=read(OUT/'final_lock.json');assert lock['active_manifest_sha256']==sha(ROOT/'configs/active_model.json')
        assert lock['selected_configuration']=='unchanged original production policy'
    items=read(path);assert len(items)==50
    for port in (8000,):
        with socket.socket() as probe:probe.bind(('127.0.0.1',port))
    token=secrets.token_urlsafe(32);previous_token=os.environ.get('QA_API_TOKEN');os.environ['QA_API_TOKEN']=token
    previous_model=os.environ.pop('MODEL_DIR',None)
    traces={};trace_lock=threading.Lock();wire=[]
    def loader():
        engine=load_engine();original=engine.candidate
        def instrument(context,question):
            result=original(context,question)
            with trace_lock:traces[key(context,question)]=result
            return result
        engine.candidate=instrument
        return engine
    app=create_app(loader=loader)
    @app.middleware('http')
    async def record(request,call_next):
        response=await call_next(request);wire.append({'method':request.method,'path':request.url.path,'status':response.status_code});return response
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=8000,log_level='error'))
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    schemas=yaml.safe_load((ROOT/'originals/qa_api_contract.yaml').read_text(encoding='utf-8'))['components']['schemas']
    result={'cohort':args.cohort,'dataset_sha256':sha(path),'semantic_judge':None,'semantic_correct_out_of_50':None,'modes':{}}
    headers={'Authorization':'Bearer '+token}
    try:
        with httpx.Client(base_url='http://127.0.0.1:8000',timeout=5,trust_env=False) as client:
            for _ in range(120):
                try:
                    health=client.get('/health')
                    if health.status_code==200:break
                except httpx.HTTPError:pass
                time.sleep(.5)
            else:raise RuntimeError('Model never became ready')
            validate(health.json(),schemas['Health']);assert health.json()=={'status':'ok'}
            info=client.get('/info',headers=headers);info.raise_for_status();validate(info.json(),schemas['Info'])
            result['info']=info.json();active=read(ROOT/'configs/active_model.json')
            assert info.json()['checkpoint_sha256']==active['hashes']['model.safetensors']
            assert info.json()['no_answer_threshold']==protocol['production_threshold']
            checks=[]
            for route in ('/info','/predict'):
                for auth in ('','Bearer incorrect','bearer '+token):
                    r=client.request('GET' if route=='/info' else 'POST',route,headers={'Authorization':auth},json={})
                    assert r.status_code==401 and r.json()=={'error':'unauthorized'} and r.headers['www-authenticate']=='Bearer'
                    checks.append(route+' authentication rejection')
            for payload in ({},{'id':'invalid','context':'x'*5001,'question':'q'}):
                r=client.post('/predict',headers=headers,json=payload);assert r.status_code in (400,422)
                validate(r.json(),schemas['Error']);assert r.json()['error']=='bad_request';checks.append('invalid request body')
            r=client.post('/predict',headers=headers,content='{');assert r.status_code in (400,422) and r.json()['error']=='bad_request'
            checks.append('malformed JSON');result['contract_checks']=checks
            for mode,concurrency in protocol['modes'].items():
                start=time.perf_counter();deadline=start+150
                def request(item):
                    tick=time.perf_counter();row={'id':item['id'],'http_valid':False,'sent':False}
                    try:
                        if tick>=deadline:raise TimeoutError('Global deadline reached before dispatch')
                        row['sent']=True
                        response=client.post('/predict',headers=headers,json={k:item[k] for k in ('id','context','question')},timeout=min(5,deadline-tick))
                        row['status']=response.status_code;row['response']=response.json();response.raise_for_status()
                        assert response.headers['content-type']=='application/json'
                        check_response(row['response'],item,schemas['PredictResponse']);row['http_valid']=True
                    except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)
                    row['seconds']=time.perf_counter()-tick
                    row['within_limits']=row['http_valid'] and row['seconds']<=5 and time.perf_counter()<=deadline
                    em,f1=score((row.get('response') or {}).get('answer') or '',item['answers']['text']) if row['within_limits'] else (0.,0.)
                    row.update(EM=100*em,F1=100*f1)
                    return row
                with ThreadPoolExecutor(max_workers=concurrency) as pool:rows=list(pool.map(request,items))
                elapsed=time.perf_counter()-start
                yes=[r for r,item in zip(rows,items) if item['answers']['text']];no=[r for r,item in zip(rows,items) if not item['answers']['text']]
                result['modes'][mode]={'rows':rows,'concurrency':concurrency,'sent':sum(r['sent'] for r in rows),
                  'http_valid':sum(r['http_valid'] for r in rows),'within_limits':sum(r['within_limits'] for r in rows),
                  'seconds':elapsed,'mean_seconds':sum(r['seconds'] for r in rows)/50,'max_seconds':max(r['seconds'] for r in rows),
                  'exact_correct_out_of_50':sum(r['EM']==100 for r in rows),'EM':sum(r['EM'] for r in rows)/50,'F1':sum(r['F1'] for r in rows)/50,
                  'answerable_EM':sum(r['EM'] for r in yes)/len(yes),'unanswerable_EM':sum(r['EM'] for r in no)/len(no),
                  'correct_nulls':sum(r['within_limits'] and r['response']['answer'] is None for r in no),
                  'incorrect_nulls':sum(r['within_limits'] and r['response']['answer'] is None for r in yes)}
                print(mode,json.dumps({k:v for k,v in result['modes'][mode].items() if k!='rows'}),flush=True)
            result['sequential_concurrent_outputs_equal']=all(a.get('response')==b.get('response') for a,b in zip(result['modes']['sequential']['rows'],result['modes']['bounded_concurrent']['rows']))
            from fastapi.testclient import TestClient
            from qa.ui import create_app as create_ui
            with TestClient(create_ui(),base_url='http://127.0.0.1:8765') as ui:
                assert ui.get('/ui/info').json()['checkpoint_sha256']==info.json()['checkpoint_sha256']
                assert ui.get('/ui/evaluation').json()['metrics']['EM']==83.
                assert all(ui.get(p).status_code==200 for p in ('/','/assets/app.js','/assets/style.css'))
                result['ui_assets_gateway_and_unchanged_dashboard']=True
    finally:
        server.should_exit=True;thread.join(timeout=20);result['server_stopped']=not thread.is_alive()
        if previous_token is None:os.environ.pop('QA_API_TOKEN',None)
        else:os.environ['QA_API_TOKEN']=previous_token
        if previous_model is not None:os.environ['MODEL_DIR']=previous_model
    result['all_wire_requests']=len(wire);result['wire_request_counts']=dict(Counter(f"{r['method']} {r['path']} {r['status']}" for r in wire))
    (OUT/(args.cohort+'_wire.json')).write_text(json.dumps(wire,indent=2))
    (OUT/(args.cohort+'_traces.json')).write_text(json.dumps(traces,indent=2))
    target.write_text(json.dumps(result,indent=2)+'\n')
    export=[]
    for item,row in zip(items,result['modes']['sequential']['rows']):
        answer=(row.get('response') or {}).get('answer');correct=row['EM']==100
        reason='API failure' if not row['within_limits'] else ('incorrect abstention' if item['answers']['text'] and answer is None else
          ('unsupported non-null answer' if not item['answers']['text'] and answer is not None else ('grading uncertainty: non-exact span' if not correct else 'exact/null automatic match; semantic review pending')))
        export.append({**item,'response':row.get('response'),'EM':row['EM'],'F1':row['F1'],'provisional_error_category':reason,
          'semantic_score':None,'semantic_justification':'','supporting_evidence':'','grader_id':'','grading_method':None})
    (OUT/(args.cohort+'_grading_export.json')).write_text(json.dumps(export,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (OUT/(args.cohort+'_grading_export.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        fields=['id','context','question','expected_answers','actual_answer','semantic_score','semantic_justification','supporting_evidence','grader_id']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in export:writer.writerow({**{k:row[k] for k in ('id','context','question','semantic_score','semantic_justification','supporting_evidence','grader_id')},
          'expected_answers':json.dumps(row['answers']['text'],ensure_ascii=False),'actual_answer':json.dumps((row.get('response') or {}).get('answer'),ensure_ascii=False)})

if __name__=='__main__':main()
