"""Exercise the running loopback gateway; no browser automation is claimed."""
import json
import time
from pathlib import Path
import httpx
from qa.metrics import score

def main():
    root=Path(__file__).resolve().parents[1]
    active=json.loads((root/'configs/active_model.json').read_text())
    with httpx.Client(base_url='http://127.0.0.1:8765',trust_env=False,timeout=15) as client:
        for _ in range(120):
            try:
                info=client.get('/ui/info')
                if info.status_code==200:break
            except httpx.HTTPError:pass
            time.sleep(1)
        else:raise RuntimeError('UI model did not become ready')
        info=info.json()
        assert info['checkpoint_sha256']==active['hashes']['model.safetensors']
        evaluation=client.get('/ui/evaluation').json()
        assert evaluation['metrics']['count']==500 and evaluation['metrics']['EM']==83.0
        assert evaluation['metrics']['F1']==86.68685393984626
        assets={p:client.get(p).status_code for p in ('/','/assets/app.js','/assets/style.css')}
        assert all(s==200 for s in assets.values())
        predictions=[]
        for context in ('Paris is the capital of France.','🌶 Paris is the capital of France.'):
            r=client.post('/ui/predict',headers={'Origin':'http://127.0.0.1:8765','X-Pimientero-UI':'1'},
              json={'id':'stable-ui','context':context,'question':'What is the capital of France?'})
            r.raise_for_status();body=r.json()
            assert set(body)=={'id','answer','start','score'}
            assert (body['answer'] is None)==(body['start'] is None)
            if body['answer'] is not None:
                assert context[body['start']:body['start']+len(body['answer'])]==body['answer']
            predictions.append({'response':body,'expected':'Paris','exact_text_correct':body['answer']=='Paris',
              'normalized_EM':score(body['answer'] or '',['Paris'])[0]})
    (root/'logs/stable_deberta_ui.json').write_text(json.dumps({'info':info,'evaluation':evaluation,'assets':assets,
      'predictions':predictions,'tested_via':'Actual local HTTP UI gateway; no visual browser claim'},indent=2))
    print(json.dumps(predictions));print('UI gateway, assets, model identity, dashboard and prediction contracts passed; semantic outcomes recorded separately.')

if __name__=='__main__':main()
