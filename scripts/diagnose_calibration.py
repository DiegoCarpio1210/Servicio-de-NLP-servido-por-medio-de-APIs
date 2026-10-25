"""Logit and span diagnostics restricted to the 512 calibration examples."""
import json,math
from pathlib import Path
import numpy as np
import torch
from datasets import load_from_disk
from qa.inference import Engine
from qa.features import encode
from qa.spans import aggregate
from qa.metrics import score,summarize


def main():
    out=Path('logs/recalibration')
    manifest=json.loads(Path('logs/subset12k/data_manifest.json').read_text())
    permitted=set(manifest['calibration_ids'])
    cached={r['id']:r for r in json.loads(Path('logs/subset12k/validation_predictions.json').read_text()) if r['calibration']}
    assert set(cached)==permitted and not permitted & set(manifest['holdout_ids'])
    raw=load_from_disk('data/squad_v2')['validation']
    ds=raw.select([i for i,id in enumerate(raw['id']) if id in permitted])
    engine=Engine('models/subset12k'); details=[]
    for count,ex in enumerate(ds):
        batch=encode(engine.tokenizer,[ex['question']],[ex['context']],engine.max_length,engine.stride)
        windows=[]; gold_pairs=[]; max_end=0; invalid_argmax=0; question_logits_top=0
        start_hit=end_hit=False; gold_lengths=[]; gold_scores=[]
        with torch.inference_mode():
            for wi in range(len(batch['input_ids'])):
                inputs={k:torch.tensor([batch[k][wi]]) for k in engine.tokenizer.model_input_names}
                output=engine.model(**inputs)
                starts=output.start_logits[0].numpy(); ends=output.end_logits[0].numpy()
                assert np.isfinite(starts).all() and np.isfinite(ends).all()
                offsets=[o if sid==1 else None for o,sid in zip(batch['offset_mapping'][wi],batch.sequence_ids(wi))]
                cls=batch['input_ids'][wi].index(engine.tokenizer.cls_token_id)
                windows.append((starts,ends,offsets,cls))
                indexes=[j for j,o in enumerate(offsets) if o is not None and o[1]>o[0]]
                if not indexes: continue
                max_end=max(max_end,max(offsets[j][1] for j in indexes))
                s_top=max(indexes,key=lambda j:starts[j]); e_top=max(indexes,key=lambda j:ends[j])
                invalid_argmax+=int(e_top<s_top or e_top-s_top+1>30)
                question_logits_top+=int(batch.sequence_ids(wi)[int(np.argmax(starts))]==0 or batch.sequence_ids(wi)[int(np.argmax(ends))]==0)
                for a,text in zip(ex['answers']['answer_start'],ex['answers']['text']):
                    b=a+len(text)
                    if offsets[indexes[0]][0]<=a and offsets[indexes[-1]][1]>=b:
                        si=next(j for j in reversed(indexes) if offsets[j][0]<=a)
                        ei=next(j for j in indexes if offsets[j][1]>=b)
                        gold_pairs.append((si,ei)); gold_lengths.append(ei-si+1)
                        start_hit |= si==s_top; end_hit |= ei==e_top
                        gold_scores.append(float(starts[si]+ends[ei]))
        candidate=aggregate(ex['context'],windows)
        assert candidate['text']==cached[ex['id']]['text'] and candidate['start']==cached[ex['id']]['start']
        assert abs(candidate['margin']-cached[ex['id']]['margin'])<1e-4
        raw_em,raw_f1=score(candidate['text'],ex['answers']['text'])
        # Context may end in whitespace, which has no tokenizer offset.
        covers_tail=max_end>=len(ex['context'].rstrip())
        details.append({'id':ex['id'],'answerable':bool(ex['answers']['text']),'windows':len(windows),'context_chars':len(ex['context']),
                        'tail_covered':covers_tail,'finite_logits':True,'gold_window_available':bool(gold_pairs),'gold_fits_30':any(n<=30 for n in gold_lengths),
                        'raw_span_EM':raw_em,'raw_span_F1':raw_f1,'start_argmax_matches_any_gold':bool(start_hit),
                        'end_argmax_matches_any_gold':bool(end_hit),'invalid_independent_argmax_windows':invalid_argmax,
                        'question_argmax_windows':question_logits_top,'question_was_capped':len(engine.tokenizer(ex['question'],add_special_tokens=False)['input_ids'])>64,
                        'margin':candidate['margin']})
        if (count+1)%64==0: print(f'Calibration logits inspected: {count+1}/512',flush=True)
    yes=[r for r in details if r['answerable']]
    result={'partition':'calibration only','count':len(details),'weights_sha256':engine.weights_sha256,
            'all_finite':all(r['finite_logits'] for r in details),'all_tails_covered':all(r['tail_covered'] for r in details),
            'multiwindow_examples':sum(r['windows']>1 for r in details),'answerable_count':len(yes),
            'answerable_missing_gold_window':sum(not r['gold_window_available'] for r in yes),
            'answerable_no_gold_within_30_tokens':sum(not r['gold_fits_30'] for r in yes),
            'question_capped_count':sum(r['question_was_capped'] for r in details),
            'start_argmax_hit_count':sum(r['start_argmax_matches_any_gold'] for r in yes),
            'end_argmax_hit_count':sum(r['end_argmax_matches_any_gold'] for r in yes),
            'invalid_unconstrained_argmax_windows':sum(r['invalid_independent_argmax_windows'] for r in details),
            'question_argmax_windows':sum(r['question_argmax_windows'] for r in details),
            'single_window_raw_span_F1':100*np.mean([r['raw_span_F1'] for r in yes if r['windows']==1]),
            'multiwindow_raw_span_F1':100*np.mean([r['raw_span_F1'] for r in yes if r['windows']>1]) if any(r['windows']>1 for r in yes) else None}
    (out/'calibration_logit_details.json').write_text(json.dumps(details,indent=2))
    (out/'calibration_logit_audit.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
