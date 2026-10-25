import json
import secrets
import time
from pathlib import Path
from threading import Event
import pytest
import yaml
from jsonschema import validate
from fastapi.testclient import TestClient
from qa.api import create_app

SCHEMAS=yaml.safe_load(Path('originals/qa_api_contract.yaml').read_text(encoding='utf-8'))['components']['schemas']

class FakeEngine:
    # Test-only fixture, never imported by production.
    metadata={'base_model':'test-base'}
    params=1
    max_length=384
    stride=128
    def predict(self, context, question):
        if question=='fail': raise RuntimeError('sensitive diagnostic')
        if question=='impossible': return {'answer':None,'start':None,'score':0.7}
        return {'answer':context[-1:],'start':len(context)-1,'score':0.8}

@pytest.fixture
def client(monkeypatch):
    token=secrets.token_urlsafe(24)
    monkeypatch.setenv('QA_API_TOKEN',token); monkeypatch.setenv('TEAM_ID','test-team')
    with TestClient(create_app(FakeEngine),raise_server_exceptions=False) as c:
        for _ in range(100):
            if c.get('/health').status_code==200: break
            time.sleep(0.01)
        yield c, {'Authorization':'Bearer '+token}

@pytest.mark.parametrize('route,method',[('/info','get'),('/predict','post')])
@pytest.mark.parametrize('auth',['','Bearer wrong','bearer wrong','BEARER wrong','Bearer','Basic abc'])
def test_auth(client,route,method,auth):
    c,_=client; r=getattr(c,method)(route,headers={'Authorization':auth})
    assert r.status_code==401 and r.json()=={'error':'unauthorized'}
    assert r.headers['www-authenticate']=='Bearer'


def test_exact_auth(client):
    c,h=client
    for value in [h['Authorization'].replace('Bearer','bearer'),h['Authorization']+' ',h['Authorization'].swapcase()]:
        assert c.get('/info',headers={'Authorization':value}).status_code==401
    r=c.get('/info',headers=h); assert r.status_code==200; validate(r.json(),SCHEMAS['Info'])

@pytest.mark.parametrize('field,values', [('id',['','a'*65,3,None,True]),('context',['','a'*5001,[],None]),('question',['','a'*501,{},False])])
def test_invalid(client,field,values):
    c,h=client
    for v in values:
        body={'id':'x','context':'abc','question':'q'}; body[field]=v
        r=c.post('/predict',headers=h,json=body)
        assert r.status_code in (400,422); validate(r.json(),SCHEMAS['Error']); assert r.json()['error']=='bad_request'

@pytest.mark.parametrize('body',[{}, {'id':'x'}, {'id':'x','context':'c','question':'q','extra':1}])
def test_missing_extra(client,body):
    c,h=client; r=c.post('/predict',headers=h,json=body)
    assert r.status_code==422 and r.json()['error']=='bad_request'


def test_broken_json(client):
    c,h=client; r=c.post('/predict',headers=h,content='{')
    assert r.status_code==422 and r.json()['error']=='bad_request'

@pytest.mark.parametrize('context,question',[('repeat repeat é\n🌎','q'),('a'*5000,'b'*500),('text','impossible')])
def test_success(client,context,question):
    c,h=client; body={'id':'x'*64,'context':context,'question':question}
    validate(body,SCHEMAS['PredictRequest'])
    r=c.post('/predict',headers=h,json=body); assert r.status_code==200
    assert r.headers['content-type']=='application/json'
    result=r.json(); validate(result,SCHEMAS['PredictResponse']); assert result['id']==body['id']
    assert (result['answer'] is None)==(result['start'] is None)
    if result['answer'] is not None: assert context[result['start']:result['start']+len(result['answer'])]==result['answer']


def test_error(client):
    c,h=client; r=c.post('/predict',headers=h,json={'id':'x','context':'x','question':'fail'})
    assert r.status_code==500 and r.json()=={'error':'internal_error'}


def test_loading(monkeypatch):
    monkeypatch.setenv('QA_API_TOKEN',secrets.token_hex()); gate=Event()
    def load(): gate.wait(5); return FakeEngine()
    with TestClient(create_app(load)) as c:
        r=c.get('/health'); assert r.status_code==503 and r.json()=={'status':'loading'}
        gate.set()
        for _ in range(100):
            if c.get('/health').status_code==200: break
            time.sleep(0.01)
        assert c.get('/health').json()=={'status':'ok'}


def test_missing_token(monkeypatch):
    monkeypatch.delenv('QA_API_TOKEN',raising=False)
    with TestClient(create_app(FakeEngine)) as c:
        assert c.get('/info',headers={'Authorization':'Bearer '}).status_code==401
