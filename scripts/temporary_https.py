"""Authorized temporary test: random token in memory, unchanged candidate, guaranteed cleanup."""
import asyncio
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
from types import SimpleNamespace
import httpx
from scripts.rehearse import run


def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    root=Path('logs/continued20k'); stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=root/('https-'+stamp); out.mkdir(parents=True)
    model=Path('models/continued20k')
    before={name:digest(model/name) for name in ('model.safetensors','calibration.json')}
    assert before['model.safetensors']==digest(Path('checkpoints/continued20k/checkpoint-1600/model.safetensors'))
    token=secrets.token_urlsafe(32); env=os.environ.copy()
    env.update(QA_API_TOKEN=token,TEAM_ID='Pimientero',MODEL_DIR=str(model))
    processes=[]; url=None; result={'scope':'Same host via real public HTTPS; mobile/other network NOT tested',
                                  'before_hashes':before,'random_test_token_only':True}
    try:
        with httpx.Client(timeout=1,trust_env=False) as client:
            try: client.get('http://127.0.0.1:8000/health')
            except httpx.TransportError: pass
            else: raise RuntimeError('Port 8000 already serves a process; refusing to reuse or terminate it')
        with (out/'server.txt').open('w') as slog,(out/'tunnel-client.txt').open('w') as tlog:
            processes.append(subprocess.Popen([sys.executable,'-m','uvicorn','qa.api:app','--host','127.0.0.1','--port','8000','--workers','1'],env=env,stdout=slog,stderr=subprocess.STDOUT))
            with httpx.Client(timeout=2,trust_env=False) as client:
                for _ in range(120):
                    if processes[0].poll() is not None: raise RuntimeError('API exited')
                    try:
                        if client.get('http://127.0.0.1:8000/health').status_code==200: break
                    except httpx.HTTPError: pass
                    time.sleep(.5)
                else: raise TimeoutError('Local readiness timeout')
            processes.append(subprocess.Popen(['tools/cloudflared.exe','tunnel','--url','http://127.0.0.1:8000','--protocol','http2'],stdout=tlog,stderr=subprocess.STDOUT))
            with httpx.Client(timeout=5,trust_env=False) as client:
                for _ in range(90):
                    match=re.search(r'https://[a-z0-9-]+\.trycloudflare.com',(out/'tunnel-client.txt').read_text(errors='replace'))
                    if match:
                        url=match.group(0)
                        try:
                            if client.get(url+'/health').status_code==200: break
                        except httpx.HTTPError: pass
                    if processes[-1].poll() is not None: raise RuntimeError('Tunnel exited')
                    time.sleep(1)
                else: raise TimeoutError('HTTPS readiness timeout')
            args=SimpleNamespace(url=url,external_network=False,network_note='Origin laptop through Cloudflare HTTPS',output=str(out/'rehearsal.json'))
            result['rehearsal_exit_code']=asyncio.run(run(args,token))
    except Exception as exc:
        result['error']=type(exc).__name__
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=10)
        result['owned_processes_stopped']=all(p.poll() is not None for p in processes)
        result['after_hashes']={name:digest(model/name) for name in before}
        result['model_and_calibration_unchanged']=result['after_hashes']==before
        checks=[]
        if url:
            with httpx.Client(timeout=5,trust_env=False) as client:
                for _ in range(3):
                    time.sleep(2)
                    try:
                        response=client.get(url+'/health')
                        checks.append({'status':response.status_code,'serves_api':response.status_code==200})
                    except httpx.HTTPError as exc: checks.append({'error':type(exc).__name__,'serves_api':False})
        result.update(url=url,closure_checks=checks,public_api_unavailable=bool(checks) and all(not x['serves_api'] for x in checks))
        (out/'lifecycle.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'evidence_dir':str(out),**result},indent=2),flush=True)


if __name__=='__main__': main()
