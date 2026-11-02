import time
import secrets
from fastapi.testclient import TestClient
from qa.api import create_app


def test_failed_load_never_ready(monkeypatch):
    token=secrets.token_hex(16); monkeypatch.setenv('QA_API_TOKEN',token)
    def fail(): raise RuntimeError('private load details')
    with TestClient(create_app(fail),raise_server_exceptions=False) as c:
        time.sleep(.02)
        assert c.get('/health').status_code==503
        r=c.post('/predict',headers={'Authorization':'Bearer '+token},json={'id':'x','context':'x','question':'q'})
        assert r.status_code==500 and r.json()=={'error':'internal_error'}


def test_auth_precedes_invalid_json(monkeypatch):
    monkeypatch.setenv('QA_API_TOKEN',secrets.token_hex(16))
    with TestClient(create_app(lambda:None)) as c:
        r=c.post('/predict',content='{')
        assert r.status_code==401 and r.json()=={'error':'unauthorized'}


def test_default_model_uses_pinned_deberta_deployment(monkeypatch):
    from test_contract import FakeEngine
    from qa.deployment import configuration
    import qa.inference
    active=configuration()
    # Rule V.1: the served model starts from the base checkpoint, never a SQuAD-fine-tuned one.
    assert active['metadata']['base_model']=='microsoft/deberta-v3-base'
    assert active['source']['repo_id']=='microsoft/deberta-v3-base'
    assert active['source']['revision']=='8ccc9b6f36199bec6961081d44eb72fb3f7353f3'
    assert 'squad' not in active['model_dir'].lower() and 'backup_model_dir' not in active
    seen=[]
    def loader(path, deployment=None): seen.append((path,deployment)); return FakeEngine()
    monkeypatch.setenv('QA_API_TOKEN',secrets.token_hex(16)); monkeypatch.delenv('MODEL_DIR',raising=False)
    monkeypatch.setattr(qa.inference,'Engine',loader)
    with TestClient(create_app()) as c:
        for _ in range(100):
            if c.get('/health').status_code==200: break
            time.sleep(.01)
        assert len(seen)==1
        assert seen[0][0].as_posix().endswith('/'+active['model_dir'])
        assert seen[0][1]['source']['revision']==active['source']['revision']
        assert seen[0][1]['calibration']['threshold']==active['calibration']['threshold']
