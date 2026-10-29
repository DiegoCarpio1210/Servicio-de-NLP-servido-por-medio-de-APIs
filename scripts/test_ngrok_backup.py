"""Authorized bounded backup test; credentials stay out of evidence and logs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import httpx


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    out = Path('logs/continued20k') / ('ngrok-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True)
    model = Path('models/continued20k')
    before = {n: digest(model/n) for n in ('model.safetensors', 'calibration.json')}
    url = 'https://diploma-supper-padded.ngrok-free.dev'
    result = {'url': url, 'scope': 'Origin laptop through real public HTTPS; not a separate mobile-network test',
              'before_hashes': before, 'checks': [], 'random_api_token_in_memory_only': True}
    processes = []
    try:
        assert before['model.safetensors'] == digest(Path('checkpoints/continued20k/checkpoint-1600/model.safetensors'))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 8000))
        token = secrets.token_urlsafe(32)
        env = os.environ.copy()
        env.update(QA_API_TOKEN=token, TEAM_ID='Pimientero', MODEL_DIR=str(model), CUDA_VISIBLE_DEVICES='')
        with (out/'server.txt').open('w') as slog, (out/'tunnel-client.txt').open('w') as tlog:
            processes.append(subprocess.Popen([sys.executable, '-m', 'uvicorn', 'qa.api:app', '--host', '127.0.0.1', '--port', '8000', '--workers', '1'], env=env, stdout=slog, stderr=subprocess.STDOUT))
            with httpx.Client(timeout=5, trust_env=False) as client:
                for _ in range(300):
                    if processes[0].poll() is not None: raise RuntimeError('API exited')
                    try:
                        if client.get('http://127.0.0.1:8000/health').status_code == 200: break
                    except httpx.HTTPError: pass
                    time.sleep(1)
                else: raise TimeoutError('Local readiness')
                print('Local CPU API ready; starting ngrok', flush=True)
                # Do not pass the API token to the tunnel process; ngrok reads its own local config.
                processes.append(subprocess.Popen(['ngrok.exe', 'http', 'http://127.0.0.1:8000', '--url', url, '--inspect=false', '--log=stdout'], stdout=tlog, stderr=subprocess.STDOUT))
                for _ in range(60):
                    if processes[-1].poll() is not None: raise RuntimeError('ngrok exited; inspect local tunnel log')
                    try:
                        response = client.get(url+'/health')
                        if response.status_code == 200 and response.json() == {'status':'ok'}: break
                    except (httpx.HTTPError, ValueError): pass
                    time.sleep(1)
                else: raise TimeoutError('HTTPS readiness')
                for name, method, endpoint, auth, body, expected in [
                    ('health', 'GET', '/health', False, None, 200),
                    ('info_no_token', 'GET', '/info', False, None, 401),
                    ('info_valid_token', 'GET', '/info', True, None, 200),
                    ('predict', 'POST', '/predict', True, {'id':'ngrok-01','context':'Paris is the capital of France.','question':'What is the capital of France?'}, 200),
                ]:
                    start=time.perf_counter()
                    r=client.request(method,url+endpoint,headers={'Authorization':'Bearer '+token} if auth else {},json=body)
                    data=r.json()
                    ok=r.status_code==expected
                    if name=='health': ok=ok and data=={'status':'ok'}
                    elif name=='info_no_token': ok=ok and data=={'error':'unauthorized'}
                    elif name=='info_valid_token':
                        ok=ok and data['checkpoint_sha256']==before['model.safetensors'] and data['training_steps']==1600
                        ok=ok and data['no_answer_threshold']==json.loads((model/'calibration.json').read_text())['threshold']
                    else:
                        ok=ok and set(data)=={'id','answer','start','score'} and data['id']=='ngrok-01'
                        ok=ok and data['answer']=='Paris' and data['start']==0 and 0<=data['score']<=1
                    result['checks'].append({'name':name,'status':r.status_code,'response':data,'seconds':time.perf_counter()-start,'passed':ok})
                result['passed']=all(c['passed'] for c in result['checks'])
    except Exception as exc:
        result.update(passed=False,error=type(exc).__name__)
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=10)
        result['owned_processes_stopped']=all(p.poll() is not None for p in processes)
        result['model_and_calibration_unchanged']=before=={n:digest(model/n) for n in before}
        result['closure_checks']=[]
        with httpx.Client(timeout=5,trust_env=False) as client:
            for _ in range(2):
                time.sleep(2)
                try:
                    r=client.get(url+'/health')
                    result['closure_checks'].append({'status':r.status_code,'api_unavailable':r.status_code!=200})
                except httpx.HTTPError as exc:
                    result['closure_checks'].append({'error':type(exc).__name__,'api_unavailable':True})
            try:
                client.get('http://127.0.0.1:8000/health')
                result['local_api_unavailable']=False
            except httpx.TransportError: result['local_api_unavailable']=True
        (out/'evidence.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'evidence_dir':str(out),**result},indent=2),flush=True)
    return 0 if result.get('passed') and result['owned_processes_stopped'] else 1


if __name__=='__main__':
    raise SystemExit(main())
