"""Verify saved weights, full recorded validation evaluation and fresh inference samples."""
import hashlib,json,random
from pathlib import Path
from qa.inference import Engine
from qa.data import load_splits,validation_partitions
from qa.metrics import summarize,threshold_sweep,weighted_threshold_sweep


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    root=Path('logs/subset12k'); model=Path('models/subset12k')
    m=json.loads((model/'training_metadata.json').read_text())
    assert m['completed'] and m['actual_epochs']==2 and m['optimizer_steps']==1522 and m['train_size']==12000
    weights=digest(model/'model.safetensors')
    assert weights==digest(Path(m['best_checkpoint'])/'model.safetensors')
    assert weights!=digest('models/pilot/model.safetensors')
    for name in ('tokenizer.json','tokenizer_config.json','vocab.txt'):
        assert digest(model/name)==digest(Path(m['best_checkpoint'])/name)
    raw=load_splits(m); parts=validation_partitions(raw['validation'],m)
    ids={k:set(v['id']) for k,v in parts.items()}; ids['train']=set(raw['train']['id'])
    assert all(not a&b for i,a in enumerate(ids.values()) for b in list(ids.values())[i+1:])
    rows=json.loads((root/'validation_predictions.json').read_text())
    cal=[r for r in rows if r['calibration']]; held=[r for r in rows if not r['calibration']]
    assert len(cal)==512 and len(held)==1024
    assert {r['id'] for r in cal}==ids['calibration'] and {r['id'] for r in held}==ids['holdout']
    examples={ex['id']:ex for ex in raw['validation']}
    for row in rows:
        ex=examples[row['id']]
        assert row['references']==ex['answers']['text']
        if row['start'] is not None:
            assert ex['context'][row['start']:row['start']+len(row['text'])]==row['text']
    e=json.loads((root/'evaluation.json').read_text())
    if e.get('objective')=='teacher_em':
        sweep=weighted_threshold_sweep(cal,e['answerable_prior'])
        best=max(sweep,key=lambda r:(round(r['EM'],10),round(r['F1'],10),-abs(r['threshold']-e['old_threshold'])))
    else:
        sweep=threshold_sweep(cal)
        best=max(sweep,key=lambda r:(r['F1'],r['EM'],-abs(r['threshold'])))
    assert best['threshold']==e['threshold']
    assert summarize(held,best['threshold'])==e['held_out']
    assert summarize(cal,best['threshold'])==e['calibration']
    engine=Engine(model)
    assert engine.threshold==e['threshold']
    assert str(next(engine.model.parameters()).device)=='cpu' and engine.weights_sha256==weights
    sampled=random.Random(42).sample(rows,32)
    for row in sampled:
        ex=examples[row['id']]; fresh=engine.candidate(ex['context'],ex['question'])
        assert fresh['text']==row['text'] and fresh['start']==row['start']
        assert abs(fresh['margin']-row['margin'])<1e-4
    predicted_null=[r for r in held if r['margin']>e['threshold']]
    correct_null=sum(not r['references'] for r in predicted_null)
    result={'training_complete':True,'selected_checkpoint_matches_export':True,'pilot_weights_differ':True,
            'weights_sha256':weights,'params':engine.params,'device':'cpu','threshold':e['threshold'],
            'validation_records_verified':len(rows),'fresh_forward_predictions_matched':len(sampled),
            'calibration_count':len(cal),'held_out_count':len(held),'disjoint_splits':True,'held_out':e['held_out'],
            'no_answer_precision_percent':100*correct_null/len(predicted_null),
            'method':'Recompute exact threshold sweep and all 1024 held-out metrics from recorded predictions; verify every reference/offset and 32 deterministic fresh model forwards.'}
    (root/'verification.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
