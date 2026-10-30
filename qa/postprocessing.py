"""Frozen general span policy validated before quality40k activation."""
import math
import numpy as np

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

