"""Strict contract checks against the actual running model, not a fixture."""
import json
from pathlib import Path
import yaml
from jsonschema import validate


def check(client,token,out):
    schemas=yaml.safe_load(Path('originals/qa_api_contract.yaml').read_text(encoding='utf-8'))['components']['schemas']
    base='http://127.0.0.1:8000'; headers={'Authorization':'Bearer '+token}
    cases=[]
    for route in ('/info','/predict'):
        for auth in ('','Bearer wrong','bearer '+token,'BEARER '+token,'Bearer  '+token):
            r=client.request('GET' if route=='/info' else 'POST',base+route,headers={'Authorization':auth},json={})
            assert r.status_code==401 and r.json()=={'error':'unauthorized'}
            assert r.headers['www-authenticate']=='Bearer'; cases.append({'check':'exact_auth'+route,'passed':True})
    for body in ({},{'id':'x','context':'x','question':'q','extra':1},{'id':1,'context':'x','question':'q'},
                 {'id':'','context':'x','question':'q'},{'id':'x','context':'x'*5001,'question':'q'}):
        r=client.post(base+'/predict',headers=headers,json=body)
        assert r.status_code in (400,422) and r.json()['error']=='bad_request'
        validate(r.json(),schemas['Error']); cases.append({'check':'invalid_body','passed':True})
    r=client.post(base+'/predict',headers=headers,content='{')
    assert r.status_code in (400,422) and r.json()['error']=='bad_request'
    cases.append({'check':'malformed_json','passed':True})
    examples=[{'id':'unicode','context':'The café in Paris opened in 2001. 🌎 Paris is mentioned twice.\nParis.', 'question':'Where is the café?'},
              {'id':'m'*64,'context':('This is ordinary English context. '*160)[:5000], 'question':('Where is the city? '*28)[:500]}]
    # Verify paired null on a previously evaluated held-out example, without changing the model.
    if (out/'validation_predictions.json').exists():
        from datasets import load_from_disk
        threshold=json.loads((out/'evaluation.json').read_text())['threshold']
        row=next(r for r in json.loads((out/'validation_predictions.json').read_text()) if not r['calibration'] and not r['references'] and r['margin']>threshold)
        ds=load_from_disk('data/squad_v2')['validation']
        ex=next(x for x in ds if x['id']==row['id'])
        examples.append({'id':'heldout-null','context':ex['context'],'question':ex['question']})
    for body in examples:
        r=client.post(base+'/predict',headers=headers,json=body,timeout=5)
        assert r.status_code==200 and r.headers['content-type']=='application/json'
        result=r.json(); validate(result,schemas['PredictResponse']); assert result['id']==body['id']
        assert (result['answer'] is None)==(result['start'] is None)
        if result['answer'] is not None:
            assert body['context'][result['start']:result['start']+len(result['answer'])]==result['answer']
        if body['id']=='heldout-null': assert result['answer'] is None
        cases.append({'check':body['id'],'passed':True,'response':result})
    (out/'real_http_contract.json').write_text(json.dumps({'checks':cases,'all_passed':True},indent=2))
