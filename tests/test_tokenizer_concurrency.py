from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import SimpleNamespace
import time
import pytest
import torch
from qa.inference import Engine
from experiments.run_pretrained_benchmark import BenchmarkEngine

@pytest.mark.parametrize('engine_type',[Engine,BenchmarkEngine])
def test_tokenizer_and_forward_share_request_lock(monkeypatch,engine_type):
    active=0
    class Batch(dict):
        def sequence_ids(self,i):return [None,1,None]
    def encode(*args):
        nonlocal active
        assert active==0,'Tokenizer concurrently borrowed'
        active+=1;time.sleep(.01);active-=1
        return Batch(input_ids=[[0,1,2]],offset_mapping=[[(0,0),(0,1),(0,0)]])
    monkeypatch.setattr('qa.inference.encode',encode)
    monkeypatch.setattr('qa.features.encode',encode)
    engine=engine_type.__new__(engine_type)
    engine.tokenizer=SimpleNamespace(model_input_names=['input_ids'],cls_token_id=0)
    engine.model=lambda **kwargs:SimpleNamespace(start_logits=torch.tensor([[0.,2.,0.]]),end_logits=torch.tensor([[0.,2.,0.]]))
    engine.lock=Lock();engine.max_length=384;engine.stride=128
    engine.policy=None if engine_type is Engine else {'max_length':30,'length_penalty':0.}
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(lambda _:engine.candidate('x','What?'),range(8)))
    assert all(r['text']=='x' and r['start']==0 for r in rows)
