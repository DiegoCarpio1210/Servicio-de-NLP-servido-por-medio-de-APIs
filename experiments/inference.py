"""Experimental postprocessing; production qa.spans is unchanged."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from qa.features import encode
from qa.metrics import summarize, prior_weighted_metrics, weighted_threshold_sweep


class ResearchEngine:
    def __init__(self,model_dir,device='cpu'):
        torch.set_num_threads(4)
        self.root=Path(model_dir);self.device=device
        with (self.root/'model.safetensors').open('rb') as f:self.sha=hashlib.file_digest(f,'sha256').hexdigest()
        self.tokenizer=AutoTokenizer.from_pretrained(self.root,local_files_only=True)
        self.model=AutoModelForQuestionAnswering.from_pretrained(self.root,local_files_only=True).to(device).eval()

    def windows(self,context,question):
        batch=encode(self.tokenizer,[question],[context],384,128);windows=[]
        with torch.inference_mode():
            for lo in range(0,len(batch['input_ids']),8):
                hi=lo+8
                inputs={k:torch.tensor(batch[k][lo:hi],device=self.device) for k in self.tokenizer.model_input_names}
                out=self.model(**inputs)
                for j in range(len(inputs['input_ids'])):
                    i=lo+j
                    offsets=[o if s==1 else None for o,s in zip(batch['offset_mapping'][i],batch.sequence_ids(i))]
                    windows.append((out.start_logits[j].cpu().numpy(),out.end_logits[j].cpu().numpy(),offsets,batch['input_ids'][i].index(self.tokenizer.cls_token_id)))
        return windows


from qa.postprocessing import candidate


def metrics(rows,threshold):
    groups=summarize(rows,threshold)
    return {'groups':groups,'weighted_75_25':prior_weighted_metrics(groups)}


def fit(rows):
    nulls=np.array([r['margin'] for r in rows if not r['references']])
    points=weighted_threshold_sweep(rows,.75)
    eligible=[p for p in points if 100*np.mean(nulls>p['threshold'])>=70.]
    return max(eligible,key=lambda p:(round(p['EM'],10),round(p['F1'],10),-abs(p['threshold'])))['threshold']
