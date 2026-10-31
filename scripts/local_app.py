"""Windows local launcher. One owner, random in-memory credential, no public tunnels."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import webbrowser

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'tmp/local-app'
URL='http://127.0.0.1:8765'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stop',action='store_true')
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT); RUN.mkdir(parents=True,exist_ok=True)
    if args.stop:
        (RUN/'STOP').touch();print('Shutdown requested for Pimientero AI.');return 0
    for module in ('torch','transformers','fastapi','uvicorn','httpx'):
        if importlib.util.find_spec(module) is None:
            print('Missing '+module+'. Install requirements.txt in .venv.');return 1
    import httpx
    import msvcrt
    lock=(RUN/'launcher.lock').open('a+b'); lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        print('Pimientero AI is already running. No duplicate instance will be started.')
        if not args.no_browser:webbrowser.open(URL)
        lock.close();return 0
    processes=[];logs=[];before={}
    try:
        for port in (8000,8765):
            with socket.socket() as sock:
                try:sock.bind(('127.0.0.1',port))
                except OSError:raise RuntimeError(f'Port {port} is occupied. Close the existing service first; no unrelated process was stopped.')
        active=json.loads((ROOT/'configs/active_model.json').read_text())
        model=ROOT/active['model_dir']
        for name,expected in active['hashes'].items():
            if not (model/name).is_file():raise RuntimeError(f'Missing {model/name}. Restore the model files; they will not be downloaded automatically.')
            with (model/name).open('rb') as f:before[name]=hashlib.file_digest(f,'sha256').hexdigest()
            if before[name]!=expected:raise RuntimeError(f'File {name} does not match the selected deployment manifest.')
        (RUN/'STOP').unlink(missing_ok=True)
        env=os.environ.copy();env.update(QA_API_TOKEN=secrets.token_urlsafe(32),TEAM_ID='Pimientero',MODEL_DIR=str(model),CUDA_VISIBLE_DEVICES='')
        print('PIMIENTERO AI | Loading '+active['model_dir']+' on CPU. The first load may take a few minutes.',flush=True)
        for app,port in [('qa.api:app',8000),('qa.ui:app',8765)]:
            log=(RUN/f'{port}.log').open('w');logs.append(log)
            processes.append(subprocess.Popen([sys.executable,'-m','uvicorn',app,'--host','127.0.0.1','--port',str(port),'--workers','1'],env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW))
        with httpx.Client(timeout=3,trust_env=False) as client:
            for _ in range(300):
                if (RUN/'STOP').exists():return 0
                if any(p.poll() is not None for p in processes):raise RuntimeError('A service failed to start. Check tmp/local-app/8000.log and 8765.log.')
                try:
                    r=client.get(URL+'/ui/info')
                    if r.status_code==200:
                        data=r.json()
                        if data['checkpoint_sha256']!=before['model.safetensors'] or data['training_steps']!=active['training_steps'] or data['no_answer_threshold']!=active['calibration']['threshold'] or data['extraction_policy']!=active['calibration']['policy']:raise RuntimeError('The API loaded an unexpected model.')
                        break
                except httpx.HTTPError:pass
                time.sleep(1)
            else:raise RuntimeError('The model was not ready within 5 minutes. Check the local logs.')
        (RUN/'status.json').write_text(json.dumps({'state':'ready','url':URL,'info':data,'pids':[p.pid for p in processes]},indent=2))
        print('Ready: '+URL+' | To stop: DETENER_PIMIENTERO.bat or Ctrl+C in this window.',flush=True)
        if not args.no_browser:webbrowser.open(URL)
        while not (RUN/'STOP').exists():
            if any(p.poll() is not None for p in processes):raise RuntimeError('A service stopped. Check the logs and start again.')
            time.sleep(.5)
        return 0
    except KeyboardInterrupt:return 0
    except Exception as exc:
        print('ERROR: '+str(exc),flush=True);return 1
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
        for log in logs:log.close()
        (RUN/'status.json').write_text(json.dumps({'state':'closed','owned_processes_stopped':all(p.poll() is not None for p in processes)}))
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
        print('Local services stopped. Temporary token discarded.',flush=True)


if __name__=='__main__':raise SystemExit(main())
