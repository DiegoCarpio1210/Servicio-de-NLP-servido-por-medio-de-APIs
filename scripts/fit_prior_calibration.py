"""Fit only calibration rows; lock policy before any independent comparison."""
import datetime,hashlib,json,random
from pathlib import Path
from qa.metrics import summarize,threshold_sweep,weighted_threshold_sweep,prior_weighted_metrics,score


def select(rows,old):
    sweep=weighted_threshold_sweep(rows,.75)
    best=max(sweep,key=lambda x:(round(x['EM'],10),round(x['F1'],10),-abs(x['threshold']-old)))
    return best,sweep


def main():
    root=Path('logs/recalibration')
    rows=[r for r in json.loads(Path('logs/subset12k/validation_predictions.json').read_text()) if r['calibration']]
    manifest=json.loads(Path('logs/subset12k/data_manifest.json').read_text())
    assert len(rows)==512 and {r['id'] for r in rows}==set(manifest['calibration_ids'])
    old=json.loads((root/'baseline_calibration.json').read_text())['threshold']
    best,sweep=select(rows,old)
    fold_ids=[set() for _ in range(5)]; rng=random.Random(42)
    for flag in (False,True):
        indexes=[r['id'] for r in rows if bool(r['references'])==flag]; rng.shuffle(indexes)
        for i,id in enumerate(indexes): fold_ids[i%5].add(id)
    cross_old=[]; cross_new=[]; folds=[]
    for fold,ids in enumerate(fold_ids):
        train=[r for r in rows if r['id'] not in ids]; test=[r for r in rows if r['id'] in ids]
        before=max(threshold_sweep(train),key=lambda r:(r['F1'],r['EM'],-abs(r['threshold'])))['threshold']
        after,_=select(train,old)
        for row in test:
            cross_old.append({**row,'text':'' if row['margin']>before else row['text'],'margin':-1})
            cross_new.append({**row,'text':'' if row['margin']>after['threshold'] else row['text'],'margin':-1})
        folds.append({'fold':fold,'size':len(test),'balanced_F1_threshold':before,'teacher_prior_EM_threshold':after['threshold']})
    audit=json.loads((root/'calibration_logit_audit.json').read_text())
    if not audit['all_finite'] or not audit['all_tails_covered']:
        raise RuntimeError('Resolve structural inference failures before fitting')
    result={'locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fit_partition':'calibration only','metric_version':'official SQuAD v2 normalized-empty-reference filtering',
            'calibration_ids':[r['id'] for r in rows],'answerable_prior':.75,'threshold':best['threshold'],
            'old_threshold':old,'primary_objective':'75/25 weighted EM, conservative proxy for exact-match/correct-null credit',
            'tie_break':'weighted F1, then closest to old threshold','span_policy':'unchanged: exhaustive valid spans, max 30 tokens, min CLS across all windows',
            'weights_sha256':audit['weights_sha256'],'before':summarize(rows,old),'after':summarize(rows,best['threshold']),
            'before_weighted':prior_weighted_metrics(summarize(rows,old)),
            'after_weighted':prior_weighted_metrics(summarize(rows,best['threshold'])),
            'cross_validation':{'folds':folds,'before_weighted':prior_weighted_metrics(summarize(cross_old,0)),
                                'after_weighted':prior_weighted_metrics(summarize(cross_new,0))},
            'sweep':sweep,'held_out_used_for_selection':False,'public_used_for_selection':False,
            'limitations':'Actual partial-credit LLM judge is unavailable; weighted F1 is only a proxy. Weighted results assume class-conditional performance transfers to the stated prior.'}
    (root/'locked_policy.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('calibration_ids','sweep')},indent=2))


if __name__=='__main__': main()
