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


def candidate(context,windows,policy,nbest=0):
    best=-math.inf;start=end=None;null=math.inf;alternatives=[]
    for sl,el,offsets,cls in windows:
        if not np.isfinite(sl).all() or not np.isfinite(el).all():raise ValueError('Nonfinite logits')
        null=min(null,float(sl[cls]+el[cls]));n=len(sl)
        valid=np.array([o is not None and o[1]>o[0] for o in offsets])
        lengths=np.arange(n)[None,:]-np.arange(n)[:,None]+1
        mask=valid[:,None]&valid[None,:]&(lengths>=1)&(lengths<=policy['max_length'])
        gaps=np.cumsum(np.array([o is None for o in offsets]))
        mask &= (gaps[None,:]-gaps[:,None])==0
        scores=(sl[:,None]+el[None,:]).astype(np.float64)-policy['length_penalty']*np.log(np.maximum(lengths,1))
        scores[~mask]=-np.inf
        i,j=np.unravel_index(np.argmax(scores),scores.shape)
        if scores[i,j]>best:
            best=float(scores[i,j]);start=offsets[i][0];end=offsets[j][1]
        if nbest:
            indices=np.argsort(scores.ravel())[-nbest:][::-1]
            for index in indices:
                a,b=np.unravel_index(index,scores.shape)
                if not np.isfinite(scores[a,b]):continue
                x,y=offsets[a][0],offsets[b][1]
                alternatives.append({'text':context[x:y],'start':x,'score':float(scores[a,b]),'start_logit':float(sl[a]),'end_logit':float(el[b]),'tokens':int(lengths[a,b])})
    result={'text':context[start:end] if start is not None else '', 'start':start,'margin':null-best if start is not None else 1e6}
    if nbest:result.update(null_score=null,best_span_score=best,top_spans=sorted(alternatives,key=lambda r:-r['score'])[:nbest],windows=len(windows))
    return result


def metrics(rows,threshold):
    groups=summarize(rows,threshold)
    return {'groups':groups,'weighted_75_25':prior_weighted_metrics(groups)}


def fit(rows):
    nulls=np.array([r['margin'] for r in rows if not r['references']])
    points=weighted_threshold_sweep(rows,.75)
    eligible=[p for p in points if 100*np.mean(nulls>p['threshold'])>=70.]
    return max(eligible,key=lambda p:(round(p['EM'],10),round(p['F1'],10),-abs(p['threshold'])))['threshold']
