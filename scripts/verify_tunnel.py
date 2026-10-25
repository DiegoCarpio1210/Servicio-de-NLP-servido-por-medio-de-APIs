"""Temporary HTTPS smoke/latency verification from this host, NOT another network."""
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
import httpx


def main():
    env=os.environ.copy(); env.update(QA_API_TOKEN=secrets.token_urlsafe(32),TEAM_ID='Pimientero',MODEL_DIR='models/pilot')
    out=Path('logs/pilot'); out.mkdir(exist_ok=True,parents=True)
    processes=[]
    try:
        with (out/'tunnel-server.txt').open('w') as serverlog, (out/'tunnel-client.txt').open('w') as tunnellog:
            processes.append(subprocess.Popen([sys.executable,'-m','uvicorn','qa.api:app','--host','127.0.0.1','--port','8000','--workers','1'],env=env,stdout=serverlog,stderr=subprocess.STDOUT))
            with httpx.Client(timeout=3) as c:
                for _ in range(120):
                    try:
                        if c.get('http://127.0.0.1:8000/health').status_code==200: break
                    except httpx.HTTPError: pass
                    time.sleep(.5)
                else: raise RuntimeError('Local model not ready')
            processes.append(subprocess.Popen(['tools/cloudflared.exe','tunnel','--url','http://127.0.0.1:8000','--protocol','http2'],stdout=tunnellog,stderr=subprocess.STDOUT))
            url=None
            for _ in range(120):
                text=(out/'tunnel-client.txt').read_text(errors='replace')
                match=re.search(r'https://[a-z0-9-]+\.trycloudflare.com',text)
                if match:
                    url=match.group(0)
                    try:
                        if httpx.get(url+'/health',timeout=5).status_code==200: break
                    except httpx.HTTPError: pass
                time.sleep(1)
            else: raise RuntimeError('Tunnel readiness timeout')
            with httpx.Client(base_url=url,timeout=5) as c:
                assert c.get('/info').status_code==401
                assert c.get('/info',headers={'Authorization':'Bearer wrong'}).status_code==401
                r=c.get('/info',headers={'Authorization':'Bearer '+env['QA_API_TOKEN']}); r.raise_for_status()
                (out/'tunnel-info.json').write_text(json.dumps(r.json(),indent=2))
            r=subprocess.run([sys.executable,'-m','scripts.evaluate_public','--url',url,'--repeat','10','--output',str(out/'tunnel-public.json')],env=env)
            (out/'tunnel-scope.json').write_text(json.dumps({'scope':'HTTPS round trip from the same host; separate network NOT tested','temporary_url':url,'evaluation_returncode':r.returncode,'tunnel_stopped_after_test':True},indent=2))
            return r.returncode
    finally:
        for proc in reversed(processes):
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()


if __name__=='__main__': raise SystemExit(main())
