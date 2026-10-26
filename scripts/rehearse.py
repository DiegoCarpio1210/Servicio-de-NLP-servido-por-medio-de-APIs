"""Explicitly invoked HTTPS rehearsal; no tunnel launch, training or calibration."""
import argparse
import asyncio
import datetime
import getpass
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import httpx
from jsonschema import validate
import yaml


def check_prediction(response,item,schema):
    assert response.status_code==200
    assert response.headers.get('content-type','').split(';')[0]=='application/json'
    body=response.json(); validate(body,schema)
    assert body['id']==item['id']
    assert (body['answer'] is None)==(body['start'] is None)
    if body['answer'] is not None:
        assert body['answer'] and item['context'][body['start']:body['start']+len(body['answer'])]==body['answer']
    return body


async def batch(client,items,schema,per_request=5.,global_cap=150.):
    started=time.monotonic(); rows=[]
    for i,item in enumerate(items):
        remaining=global_cap-(time.monotonic()-started)
        if remaining<=0: break
        begin=time.monotonic(); row={'index':i,'id':item['id']}
        try:
            response=await asyncio.wait_for(client.post('/predict',json={k:item[k] for k in ('id','context','question')}),
                                            timeout=min(per_request,remaining))
            elapsed=time.monotonic()-begin
            assert elapsed<=per_request and time.monotonic()-started<=global_cap
            body=check_prediction(response,item,schema)
            row.update(contract_valid=True,response=body,
                       correct=body['answer'] is None if item['is_impossible'] else body['answer'] in [a['text'] for a in item['answers']])
        except Exception as exc:
            # Never serialize exception messages: they can contain credentials or request details.
            row.update(contract_valid=False,correct=False,error=type(exc).__name__)
        row['seconds']=time.monotonic()-begin; rows.append(row)
    elapsed=time.monotonic()-started
    return {'rows':rows,'planned_requests':len(items),'completed_requests':len(rows),'elapsed_seconds':elapsed,
            'technical_pass':len(rows)==len(items) and elapsed<=global_cap and all(r['contract_valid'] and r['seconds']<=per_request for r in rows),
            'accuracy_pass':len(rows)==len(items) and all(r['correct'] for r in rows)}


async def run(args,token):
    schemas=yaml.safe_load(Path('originals/qa_api_contract.yaml').read_text(encoding='utf-8'))['components']['schemas']
    expected=json.loads(Path('logs/continued20k/info.json').read_text())
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'url':args.url,
            'network_scope':'operator declares another network' if args.external_network else 'network independence NOT established',
            'network_note':args.network_note,'https':args.url.startswith('https://'),'preflight_pass':False,
            'dataset':'Five unchanged public examples repeated ten times; not 50 distinct hidden questions',
            'per_request_limit_seconds':5,'global_limit_seconds':150}
    try:
        async with httpx.AsyncClient(base_url=args.url,headers={'Authorization':'Bearer '+token},timeout=5,
                                     follow_redirects=False,trust_env=False) as client:
            async def request(method,path,**kwargs):
                return await asyncio.wait_for(client.request(method,path,**kwargs),5)
            health=await request('GET','/health',headers={'Authorization':''})
            assert health.status_code==200 and health.json()=={'status':'ok'}
            for route,method in (('/info','GET'),('/predict','POST')):
                for auth in ('','Bearer wrong'):
                    r=await request(method,route,headers={'Authorization':auth})
                    assert r.status_code==401 and r.json().get('error')=='unauthorized'
            info=await request('GET','/info'); assert info.status_code==200
            validate(info.json(),schemas['Info'])
            for key in ('team','model','params','checkpoint_sha256','training_steps','no_answer_threshold'):
                assert info.json()[key]==expected[key]
            result.update(preflight_pass=True,info=info.json())
            items=json.loads(Path('originals/public_sample.json').read_text())['items']*10
            result.update(await batch(client,items,schemas['PredictResponse']))
    except Exception as exc:
        result.update(technical_pass=False,error=type(exc).__name__)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','info')},indent=2))
    return 0 if result.get('technical_pass') and result.get('accuracy_pass') else 1


def main():
    p=argparse.ArgumentParser(); p.add_argument('--url',required=True); p.add_argument('--output',required=True)
    p.add_argument('--external-network',action='store_true'); p.add_argument('--network-note',default='not supplied')
    p.add_argument('--allow-local-http',action='store_true'); args=p.parse_args()
    url=urlsplit(args.url)
    if url.username or url.password or url.query or url.fragment: p.error('URL must not contain credentials, query or fragment')
    if not url.hostname or (url.scheme!='https' and not (args.allow_local_http and url.scheme=='http' and url.hostname in ('localhost','127.0.0.1'))):
        p.error('HTTPS required, except explicit localhost-only HTTP rehearsal')
    if Path(args.output).exists(): p.error('Output exists; choose a new filename to preserve earlier evidence')
    token=os.environ.get('QA_API_TOKEN') or getpass.getpass('QA_API_TOKEN (hidden, not saved): ')
    if not token: p.error('Token required')
    raise SystemExit(asyncio.run(run(args,token)))


if __name__=='__main__': main()
