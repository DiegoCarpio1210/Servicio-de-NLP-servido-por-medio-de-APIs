"""Small real-model UI gateway smoke test, against an already running local launcher."""
import json
import time
from pathlib import Path
import httpx


def main():
    checks=[]
    with httpx.Client(base_url='http://127.0.0.1:8765',timeout=30,trust_env=False) as client:
        info=client.get('/ui/info').json()
        assert info['checkpoint_sha256']=='7d66ffcff40157ed8f9f728f51ffe3c1b5b3f249ac8c100ece9666d1f7007848'
        assert info['training_steps']==1600
        headers={'Origin':'http://127.0.0.1:8765','X-Pimientero-UI':'1'}
        for case,context,question in [
            ('answer','Paris is the capital of France.','What is the capital of France?'),
            ('null','Paris is the capital of France.','Who discovered penicillin?'),
            ('unicode','🌶 Paris is the capital of France.','What is the capital of France?'),
            ('long_context','Background information. '*150+'Paris is the capital of France.','What is the capital of France?'),
        ]:
            start=time.perf_counter()
            r=client.post('/ui/predict',headers=headers,json={'id':case,'context':context,'question':question})
            data=r.json(); assert r.status_code==200
            assert set(data)=={'id','answer','start','score'} and data['id']==case
            if data['answer'] is None: assert data['start'] is None
            else: assert context[data['start']:data['start']+len(data['answer'])]==data['answer']
            if case=='answer':assert data['answer']=='Paris' and data['start']==0
            if case=='null':assert data['answer'] is None
            checks.append({'case':case,'status':r.status_code,'response':data,'seconds':time.perf_counter()-start,'context_chars':len(context),'offset_valid':True})
        invalid=client.post('/ui/predict',headers=headers,json={'id':'over','context':'x'*5001,'question':'What?'})
        assert invalid.status_code==422
        rebinding=client.get('/ui/info',headers={'Host':'malicious.example:8765'})
        assert rebinding.status_code==403
        crosssite=client.post('/ui/predict',headers={'Origin':'https://malicious.example','X-Pimientero-UI':'1'},json={'id':'cross','context':'text','question':'What?'})
        assert crosssite.status_code==403
        with httpx.Client(base_url='http://127.0.0.1:8000',timeout=3,trust_env=False) as api:
            assert api.get('/health').status_code==200
            assert api.get('/info').status_code==401
            assert api.post('/predict',json={'id':'direct','context':'Paris','question':'Where?'}).status_code==401
    evidence={'checks':checks,'info':info,'oversized_input':422,'host_rebinding':403,'cross_site_post':403,'academic_api_without_token':401,'scope':'Real local CPU model through UI gateway; four smoke inputs, not a new accuracy evaluation'}
    Path('logs/ui_http_verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
