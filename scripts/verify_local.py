"""Start a real CPU server with an ephemeral in-memory test token and measure HTTP."""
import hashlib
import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
import httpx


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-dir',default='models/subset12k'); p.add_argument('--repeat',type=int,default=10)
    a=p.parse_args(); env=os.environ.copy()
    env.update(QA_API_TOKEN=secrets.token_urlsafe(32),TEAM_ID='Pimientero',MODEL_DIR=a.model_dir)
    out=Path('logs')/Path(a.model_dir).name; out.mkdir(parents=True,exist_ok=True)
    with (out/'server.txt').open('w') as logfile:
        proc=subprocess.Popen([sys.executable,'-m','uvicorn','qa.api:app','--host','127.0.0.1','--port','8000','--workers','1'],
                              env=env,stdout=logfile,stderr=subprocess.STDOUT)
        try:
            with httpx.Client(timeout=2) as client:
                for _ in range(120):
                    if proc.poll() is not None: raise RuntimeError('Server exited; inspect server.txt')
                    try:
                        if client.get('http://127.0.0.1:8000/health').status_code==200: break
                    except httpx.HTTPError: pass
                    time.sleep(0.5)
                else: raise RuntimeError('Readiness timeout; inspect server.txt')
                assert client.get('http://127.0.0.1:8000/info').status_code==401
                assert client.get('http://127.0.0.1:8000/info',headers={'Authorization':'Bearer wrong'}).status_code==401
                info=client.get('http://127.0.0.1:8000/info',headers={'Authorization':'Bearer '+env['QA_API_TOKEN']})
                info.raise_for_status()
                with (Path(a.model_dir)/'model.safetensors').open('rb') as weights:
                    assert info.json()['checkpoint_sha256']==hashlib.file_digest(weights,'sha256').hexdigest()
                assert info.json()['no_answer_threshold']==json.loads((Path(a.model_dir)/'calibration.json').read_text())['threshold']
                assert info.json()['training_steps']==json.loads((Path(a.model_dir)/'training_metadata.json').read_text())['optimizer_steps']
                (out/'info.json').write_text(json.dumps(info.json(),indent=2))
                from scripts.http_contract_checks import check
                check(client,env['QA_API_TOKEN'],out)
            result=subprocess.run([sys.executable,'-m','scripts.evaluate_public','--url','http://127.0.0.1:8000',
                                   '--output',str(out/'public.json'),'--repeat',str(a.repeat)],env=env)
            return result.returncode
        finally:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()


if __name__=='__main__': raise SystemExit(main())
