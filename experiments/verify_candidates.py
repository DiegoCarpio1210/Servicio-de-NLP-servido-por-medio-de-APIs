"""Real local HTTP/CPU comparison after final lock; never opens a tunnel."""
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import httpx
import psutil
from jsonschema import validate
import yaml
from scripts.http_contract_checks import check

ROOT=Path('logs/quality20261029')

def cpu_seconds(pid):
    parent=psutil.Process(pid)
    return sum(p.cpu_times().user+p.cpu_times().system for p in [parent,*parent.children(recursive=True)])

def verify(name,profile):
    out=ROOT/('cpu_'+name);out.mkdir(exist_ok=True)
    if (out/'result.json').exists():raise RuntimeError('Do not overwrite completed evidence')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',8000))
    token=secrets.token_urlsafe(32);env=os.environ.copy()
    env.update(QA_API_TOKEN=token,TEAM_ID='Pimientero',QA_EXPERIMENT_PROFILE=name,TORCH_NUM_THREADS='4')
    with (out/'server.txt').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-m','uvicorn','experiments.candidate_api:app','--host','127.0.0.1','--port','8000','--workers','1'],env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url='http://127.0.0.1:8000',timeout=5) as client:
                for _ in range(600):
                    if proc.poll() is not None:raise RuntimeError('Candidate server exited')
                    try:
                        if client.get('/health').status_code==200:break
                    except httpx.HTTPError:pass
                    time.sleep(.5)
                else:raise RuntimeError('Readiness deadline exceeded')
                headers={'Authorization':'Bearer '+token}
                info=client.get('/info',headers=headers);info.raise_for_status()
                assert info.json()['checkpoint_sha256']==profile['weights_sha256']
                assert info.json()['no_answer_threshold']==profile['threshold']
                check(client,token,out)
                schema=yaml.safe_load(Path('originals/qa_api_contract.yaml').read_text(encoding='utf-8'))['components']['schemas']['PredictResponse']
                items=json.loads(Path('originals/public_sample.json').read_text())['items']
                rows=[];cpu_start=cpu_seconds(proc.pid);started=time.perf_counter();peak=0
                for cycle in range(10):
                    for item in items:
                        begin=time.perf_counter()
                        r=client.post('/predict',headers=headers,json={k:item[k] for k in ('id','context','question')});r.raise_for_status()
                        elapsed=time.perf_counter()-begin;body=r.json();validate(body,schema)
                        assert body['id']==item['id']
                        assert (body['answer'] is None)==(body['start'] is None)
                        if body['answer'] is not None:assert item['context'][body['start']:body['start']+len(body['answer'])]==body['answer']
                        parent=psutil.Process(proc.pid)
                        peak=max(peak,sum(p.memory_info().rss for p in [parent,*parent.children(recursive=True)]))
                        correct=body['answer'] is None if item['is_impossible'] else body['answer'] in [a['text'] for a in item['answers']]
                        rows.append({'cycle':cycle,'id':item['id'],'seconds':elapsed,'response':body,'correct':correct})
                total=time.perf_counter()-started
                cpu_used=cpu_seconds(proc.pid)-cpu_start
                result={'profile':name,'info':info.json(),'device':'cpu','transport':'real loopback HTTP; not HTTPS or another network',
                  'rows':rows,'public_correct_of_5':sum(r['correct'] for r in rows[:5]),'requests':len(rows),
                  'elapsed_seconds':total,'max_request_seconds':max(r['seconds'] for r in rows),
                  'mean_request_seconds':sum(r['seconds'] for r in rows)/50,'peak_sampled_process_tree_rss_mib':peak/2**20,
                  'server_cpu_seconds':cpu_used,'mean_server_cpu_percent_all_logical_cores':100*cpu_used/total/psutil.cpu_count(),
                  'time_limits_pass':total<=150 and all(r['seconds']<=5 for r in rows),'production_unchanged':True}
        finally:
            parent=psutil.Process(proc.pid)
            owned=[*parent.children(recursive=True),parent]
            for p in reversed(owned):
                try:p.terminate()
                except psutil.NoSuchProcess:pass
            _,alive=psutil.wait_procs(owned,timeout=10)
            for p in alive:p.kill()
            _,alive=psutil.wait_procs(alive,timeout=5)
            proc.wait(timeout=5)
    result['server_stopped']=not alive and proc.poll() is not None
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)

def main():
    profiles=json.loads((ROOT/'final_lock.json').read_text())['profiles']
    for name,profile in profiles.items():verify(name,profile)

if __name__=='__main__':main()
