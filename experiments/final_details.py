"""Descriptive final-set analysis only, after locked comparison; never selects settings."""
import json
from pathlib import Path
from datasets import load_from_disk
from qa.metrics import score

def summarize(rows,threshold):
    answered=[r for r in rows if r['start'] is not None and r['margin']<=threshold]
    wrong=sum(score(r['text'],r['references'])[0]==0 for r in answered)
    exact=sum(score('' if r['margin']>threshold else r['text'],r['references'])[0] for r in rows)
    f1=sum(score('' if r['margin']>threshold else r['text'],r['references'])[1] for r in rows)
    return {'count':len(rows),'EM':100*exact/len(rows),'F1':100*f1/len(rows),
      'abstention_rate_percent':100*(len(rows)-len(answered))/len(rows),
      'non_exact_prediction_rate_percent':100*(len(rows)-exact)/len(rows),
      'wrong_nonnull_answers_percent_of_all':100*wrong/len(rows),
      'wrong_nonnull_answers_percent_of_answered':100*wrong/len(answered) if answered else None}

def main():
    root=Path('logs/quality20261029')
    assert (root/'final_comparison.json').exists()
    rows=json.loads((root/'final_predictions.json').read_text())
    lock=json.loads((root/'final_lock.json').read_text())
    raw=load_from_disk('data/squad_v2')['train'];questions=dict(zip(raw['id'],raw['question']))
    result={}
    for name,items in rows.items():
        threshold=lock['profiles'][name]['threshold'];groups={}
        for r in items:
            q=questions[r['id']].lower().strip()
            kind=next((k for k in ('how many','how much','what','which','who','when','where','why','how') if q.startswith(k+' ')),'other')
            groups.setdefault(kind,[]).append(r)
        result[name]={'overall':summarize(items,threshold),'question_types':{k:summarize(v,threshold) for k,v in groups.items()}}
    (root/'final_details.json').write_text(json.dumps({'profiles':result,
      'interpretation':'Wrong means normalized non-exact, including partial matches; question prefixes are descriptive and were not used for tuning. Small subgroups are uncertain.'},indent=2)+'\n')
    print(json.dumps({k:v['overall'] for k,v in result.items()},indent=2))

if __name__=='__main__':main()
