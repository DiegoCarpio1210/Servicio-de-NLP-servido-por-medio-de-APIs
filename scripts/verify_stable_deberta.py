"""Real production-engine compatibility and bounded local HTTP load verification."""
import json
import os
from pathlib import Path
from experiments.final500 import api_checks,atomic
from qa.deployment import load_engine

def main():
    from datasets import load_from_disk
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    os.environ['TORCH_NUM_THREADS']='2';os.environ.pop('MODEL_DIR',None)
    engine=load_engine()
    rows=json.loads((root/'logs/final500/deberta-v3-base-squad2/predictions.json').read_text())
    null_id=next(r['id'] for r in rows if not r['references'] and r['response']['answer'] is None)
    null=next(r for r in load_from_disk(str(root/'data/squad_v2'))['validation'] if r['id']==null_id)
    out=root/'logs/stable_deberta';out.mkdir(exist_ok=True)
    result=api_checks(engine,out,null)
    atomic(out/'http.json',result)
    compact={k:({a:b for a,b in v.items() if a!='rows'} if isinstance(v,dict) else v) for k,v in result.items()}
    atomic(root/'logs/stable_deberta_verification.json',compact)
    assert result['contract_pass'] and result['server_stopped'] and all(result[k]['pass'] for k in ('sequential','concurrent'))
    print(json.dumps(compact,indent=2))

if __name__=='__main__':main()
