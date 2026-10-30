import httpx
import pytest
from fastapi.testclient import TestClient
from qa.ui import create_app

ORIGIN={'Origin':'http://127.0.0.1:8765','X-Pimientero-UI':'1'}
BODY={'id':'ui-test','context':'Paris is the capital of France.','question':'What is the capital?'}


def client_for(monkeypatch, handler):
    monkeypatch.setenv('QA_API_TOKEN','test-only-ui-secret')
    return TestClient(create_app(httpx.MockTransport(handler)),base_url='http://127.0.0.1:8765')


def test_proxy_keeps_bearer_server_side_and_preserves_unicode(monkeypatch):
    seen=[]
    def handler(request):
        seen.append(request)
        assert request.headers['Authorization']=='Bearer test-only-ui-secret'
        return httpx.Response(200,json={'id':'ui-test','answer':'Paris','start':2,'score':.8})
    with client_for(monkeypatch,handler) as client:
        r=client.post('/ui/predict',headers=ORIGIN,json={**BODY,'context':'🌶 Paris'})
        assert r.json()=={'id':'ui-test','answer':'Paris','start':2,'score':.8}
        assert 'test-only-ui-secret' not in client.get('/').text
        assert 'test-only-ui-secret' not in client.get('/assets/app.js').text
        assert 'Authorization' not in client.get('/assets/app.js').text
        assert len(seen)==1


@pytest.mark.parametrize('headers',[{}, {'Origin':'https://evil.example','X-Pimientero-UI':'1'},{'Origin':'http://127.0.0.1:8765'}])
def test_cross_site_write_blocked(monkeypatch,headers):
    with client_for(monkeypatch,lambda r:pytest.fail('Must not reach API')) as client:
        assert client.post('/ui/predict',json=BODY,headers=headers).status_code==403


def test_rebinding_and_csp(monkeypatch):
    with client_for(monkeypatch,lambda r:pytest.fail('Must not reach API')) as client:
        assert client.get('/',headers={'Host':'evil.example:8765'}).status_code==403
        r=client.get('/')
        assert "frame-ancestors 'none'" in r.headers['content-security-policy']
        assert 'access-control-allow-origin' not in r.headers


def test_api_down_and_null(monkeypatch):
    def offline(r):raise httpx.ConnectError('offline')
    with client_for(monkeypatch,offline) as client:
        assert client.get('/ui/status').status_code==503
        assert client.post('/ui/predict',headers=ORIGIN,json=BODY).status_code==503
    null={'id':'ui-test','answer':None,'start':None,'score':.9}
    with client_for(monkeypatch,lambda r:httpx.Response(200,json=null)) as client:
        assert client.post('/ui/predict',headers=ORIGIN,json=BODY).json()==null


def test_limits_and_candidate_metrics(monkeypatch):
    with client_for(monkeypatch,lambda r:pytest.fail('Must not reach API')) as client:
        for body in ({**BODY,'context':'x'*5001},{**BODY,'question':'x'*501},{**BODY,'context':''}):
            assert client.post('/ui/predict',headers=ORIGIN,json=body).status_code==422
        data=client.get('/ui/evaluation').json()
        assert data['metrics']['count']==1024
        assert data['metrics']['EM']==61.81640625
        assert data['metrics']['F1']==65.52553295229076
        assert data['answerable']['EM']==48.046875
        assert data['unanswerable']['EM']==75.5859375
        assert data['weighted_75_25']['EM']==54.931640625
        assert 'modelo activo' in data['label']
        for path in ('/','/assets/app.js','/assets/style.css'):
            assert client.get(path).status_code==200
