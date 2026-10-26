import asyncio
import json
from pathlib import Path
import httpx
import yaml
from scripts.rehearse import batch

SCHEMA=yaml.safe_load(Path('originals/qa_api_contract.yaml').read_text(encoding='utf-8'))['components']['schemas']['PredictResponse']
ITEM={'id':'q','context':'Paris','question':'Where?','is_impossible':False,'answers':[{'text':'Paris'}]}


def execute(handler,**budgets):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler),base_url='https://test.invalid') as client:
            return await batch(client,[ITEM]*50,SCHEMA,**budgets)
    return asyncio.run(go())


def test_fifty_valid_responses():
    r=execute(lambda req:httpx.Response(200,json={'id':'q','answer':'Paris','start':0,'score':.8}))
    assert r['completed_requests']==50 and r['technical_pass'] and r['accuracy_pass']


def test_bad_offsets_fail_contract():
    r=execute(lambda req:httpx.Response(200,json={'id':'q','answer':'Paris','start':1,'score':.8}))
    assert not r['technical_pass']


def test_null_is_contract_valid_but_wrong_for_answerable():
    r=execute(lambda req:httpx.Response(200,json={'id':'q','answer':None,'start':None,'score':.8}))
    assert r['technical_pass'] and not r['accuracy_pass']


def test_wall_deadline_cancels_and_preserves_global_budget():
    async def slow(req):
        await asyncio.sleep(1)
        return httpx.Response(200,json={})
    r=execute(slow,per_request=.01,global_cap=.035)
    assert not r['technical_pass'] and r['completed_requests']<50
    assert all(x['error']=='TimeoutError' for x in r['rows'])
    assert r['elapsed_seconds']<.3
