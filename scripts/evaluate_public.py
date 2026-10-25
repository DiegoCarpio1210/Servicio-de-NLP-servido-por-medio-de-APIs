"""HTTP evaluation only; never trains or calibrates on the public cases."""
import argparse
import json
import os
import time
from pathlib import Path
import httpx


def main():
    p=argparse.ArgumentParser(); p.add_argument('--url',default='http://localhost:8000')
    p.add_argument('--output',default='logs/public.json'); p.add_argument('--repeat',type=int,default=1)
    args=p.parse_args()
    token=os.environ['QA_API_TOKEN']
    items=json.loads(Path('originals/public_sample.json').read_text())['items']
    rows=[]; global_start=time.perf_counter()
    with httpx.Client(base_url=args.url,headers={'Authorization':'Bearer '+token},timeout=5) as client:
        for cycle in range(args.repeat):
            for item in items:
                started=time.perf_counter(); row={'id':item['id'],'cycle':cycle}
                try:
                    response=client.post('/predict',json={k:item[k] for k in ('id','context','question')})
                    row['seconds']=time.perf_counter()-started
                    response.raise_for_status(); r=response.json(); row['response']=r
                    assert r['id']==item['id'] and isinstance(r['score'],(int,float)) and 0<=r['score']<=1
                    assert (r['answer'] is None)==(r['start'] is None)
                    if r['answer'] is not None:
                        assert type(r['start']) is int and r['start']>=0 and r['answer']
                        assert item['context'][r['start']:r['start']+len(r['answer'])]==r['answer']
                    row['contract_valid']=True
                    row['correct']=r['answer'] is None if item['is_impossible'] else r['answer'] in [a['text'] for a in item['answers']]
                except Exception as exc:
                    row.update(correct=False, error=type(exc).__name__)
                rows.append(row)
                print(json.dumps(row),flush=True)
    result={'url':args.url,'elapsed_seconds':time.perf_counter()-global_start,'rows':rows,
            'all_correct':all(r['correct'] for r in rows),
            'timeout_budget_pass':len(rows)==50 and time.perf_counter()-global_start<150 and all(r.get('seconds',6)<5 for r in rows)}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2))
    raise SystemExit(0 if result['all_correct'] else 1)


if __name__=='__main__': main()
