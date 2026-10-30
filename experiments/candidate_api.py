"""Loopback-only evaluation adapter; production default/model files are untouched."""
import json
import os
from pathlib import Path
from qa.api import create_app
from qa.inference import Engine
from experiments.inference import candidate
from qa.features import encode
import torch

class ExperimentalEngine(Engine):
    def __init__(self,profile):
        super().__init__(profile['model_dir'],require_calibration=False)
        assert self.weights_sha256==profile['weights_sha256']
        self.threshold=profile['threshold'];self.policy=profile['policy']
        self.calibration={'objective':'experimental_75_25_EM_null_floor70'}

    def candidate(self,context,question):
        batch=encode(self.tokenizer,[question],[context],self.max_length,self.stride)
        windows=[]
        with self.lock,torch.inference_mode():
            for lo in range(0,len(batch['input_ids']),8):
                inputs={k:torch.tensor(batch[k][lo:lo+8]) for k in self.tokenizer.model_input_names}
                out=self.model(**inputs)
                for j in range(len(inputs['input_ids'])):
                    i=lo+j
                    offsets=[o if s==1 else None for o,s in zip(batch['offset_mapping'][i],batch.sequence_ids(i))]
                    windows.append((out.start_logits[j].numpy(),out.end_logits[j].numpy(),offsets,batch['input_ids'][i].index(self.tokenizer.cls_token_id)))
        return candidate(context,windows,self.policy)

def load():
    if not os.getenv('QA_API_TOKEN'):raise RuntimeError('Temporary token required')
    frozen=json.loads(Path('logs/quality20261029/final_lock.json').read_text())
    engine=ExperimentalEngine(frozen['profiles'][os.environ['QA_EXPERIMENT_PROFILE']])
    engine.predict('Warmup context.','What is this?')
    return engine

app=create_app(loader=load)
