"""Read-only sample-04 logit inspection; never fits or modifies inference settings."""
import json
from pathlib import Path
import torch
from qa.features import encode
from qa.inference import Engine


def main():
    engine=Engine('models/continued20k')
    ex=next(x for x in json.loads(Path('originals/public_sample.json').read_text())['items'] if x['id']=='sample-04')
    batch=encode(engine.tokenizer,[ex['question']],[ex['context']],engine.max_length,engine.stride)
    windows=[]
    with torch.inference_mode():
        for i in range(len(batch['input_ids'])):
            inputs={k:torch.tensor([batch[k][i]]) for k in engine.tokenizer.model_input_names}
            output=engine.model(**inputs); start=output.start_logits[0]; end=output.end_logits[0]
            cls=batch['input_ids'][i].index(engine.tokenizer.cls_token_id)
            context=[j for j,s in enumerate(batch.sequence_ids(i)) if s==1]
            spans=[]
            for a in context:
                for b in context:
                    if not a<=b<a+30: continue
                    lo,hi=batch['offset_mapping'][i][a][0],batch['offset_mapping'][i][b][1]
                    if lo<hi: spans.append({'text':ex['context'][lo:hi],'start':lo,'end':hi,
                                           'score':float(start[a]+end[b]),'tokens':b-a+1})
            spans.sort(key=lambda x:x['score'],reverse=True)
            windows.append({'active_tokens':sum(batch['attention_mask'][i]),
                            'context_end':max(batch['offset_mapping'][i][j][1] for j in context),
                            'cls_start_logit':float(start[cls]),'cls_end_logit':float(end[cls]),
                            'null_score':float(start[cls]+end[cls]),'top_spans':spans[:5],
                            'all_logits_finite':bool(torch.isfinite(start).all() and torch.isfinite(end).all())})
    best=max((span for w in windows for span in w['top_spans']),key=lambda s:s['score'])
    margin=min(w['null_score'] for w in windows)-best['score']
    result={'id':ex['id'],'weights_sha256':engine.weights_sha256,'device':str(next(engine.model.parameters()).device),
            'threshold':engine.threshold,'context_characters':len(ex['context']),'windows':windows,
            'margin':margin,'returns_null':margin>engine.threshold,'offset_valid':ex['context'][best['start']:best['end']]==best['text'],
            'diagnostic_only':True,'weights_or_parameters_changed':False}
    Path('logs/continued20k/sample04_diagnosis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
