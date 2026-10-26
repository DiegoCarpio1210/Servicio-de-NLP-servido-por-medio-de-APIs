"""Apply the calibration-only locked policy, then evaluate holdout without selecting anything."""
import datetime,hashlib,json,random,subprocess
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from qa.inference import Engine
from qa.metrics import summarize,prior_weighted_metrics,score


def main():
    root=Path('logs/recalibration'); policy=json.loads((root/'locked_policy.json').read_text())
    committed=subprocess.check_output(['git','show','759bcb3:logs/recalibration/locked_policy.json'])
    assert json.loads(committed)==policy, 'Policy changed after lock'
    model=Path('models/subset12k')
    with (model/'model.safetensors').open('rb') as f: weights=hashlib.file_digest(f,'sha256').hexdigest()
    assert weights==policy['weights_sha256']
    active={'threshold':policy['threshold'],'old_threshold':policy['old_threshold'],'objective':'teacher_em',
            'answerable_prior':.75,'weights_sha256':weights,'fit_partition':'512 calibration examples only',
            'policy_commit':'759bcb3','calibration_ids':policy['calibration_ids'],'calibration':policy['after']}
    temp=model/'calibration.tmp'; temp.write_text(json.dumps(active,indent=2)); temp.replace(model/'calibration.json')
    evaluation_started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows=json.loads(Path('logs/subset12k/validation_predictions.json').read_text())
    held=[r for r in rows if not r['calibration']]; cal=[r for r in rows if r['calibration']]
    manifest=json.loads(Path('logs/subset12k/data_manifest.json').read_text())
    assert len(held)==1024 and {r['id'] for r in held}==set(manifest['holdout_ids'])
    assert not set(policy['calibration_ids']) & set(manifest['holdout_ids'])
    before=summarize(held,policy['old_threshold']); after=summarize(held,policy['threshold'])
    # Fixed seeded paired class-stratified bootstrap: uncertainty only, never policy selection.
    rng=np.random.default_rng(42); differences=[]
    for flag in (True,False):
        subset=[r for r in held if bool(r['references'])==flag]
        differences.append(np.array([score('' if r['margin']>policy['threshold'] else r['text'],r['references'])[0]-
                                     score('' if r['margin']>policy['old_threshold'] else r['text'],r['references'])[0] for r in subset]))
    bootstrap=[100*(.75*rng.choice(differences[0],len(differences[0]),replace=True).mean()+.25*rng.choice(differences[1],len(differences[1]),replace=True).mean()) for _ in range(2000)]
    engine=Engine(model); sampled=random.Random(81).sample(held,32)
    needed={r['id'] for r in sampled}; ds=load_from_disk('data/squad_v2')['validation']
    samples={x['id']:x for x in ds if x['id'] in needed}
    for row in sampled:
        ex=samples[row['id']]; prediction=engine.candidate(ex['context'],ex['question'])
        assert prediction['text']==row['text'] and abs(prediction['margin']-row['margin'])<1e-4
    def breakdown(threshold):
        a=[r for r in held if r['references']]; n=[r for r in held if not r['references']]
        return {'answerable_abstained':sum(r['margin']>threshold for r in a),
                'answerable_answered':sum(r['margin']<=threshold for r in a),
                'unanswerable_correct_null':sum(r['margin']>threshold for r in n),
                'unanswerable_false_span':sum(r['margin']<=threshold for r in n)}
    result={'policy_commit':'759bcb3','evaluation_started_utc':evaluation_started,'weights_unchanged':True,
            'weights_sha256':weights,'old_threshold':policy['old_threshold'],'new_threshold':policy['threshold'],
            'metric_version':policy['metric_version'],'before':before,'after':after,
            'before_weighted_75_25':prior_weighted_metrics(before),'after_weighted_75_25':prior_weighted_metrics(after),
            'weighted_EM_delta_95_percent_bootstrap_interval':np.percentile(bootstrap,[2.5,97.5]).tolist(),
            'before_errors':breakdown(policy['old_threshold']),'after_errors':breakdown(policy['threshold']),
            'fresh_holdout_forward_matches':32,'evaluation_method':'Fixed post-adjustment evaluation using cached span logits/margins for all 1024 examples plus 32 fresh forwards. Span algorithm and weights unchanged.',
            'held_out_used_for_selection':False,'public_used_for_selection':False}
    (root/'holdout_comparison.json').write_text(json.dumps(result,indent=2))
    current={'threshold':policy['threshold'],'old_threshold':policy['old_threshold'],'objective':'teacher_em','answerable_prior':.75,
             'criterion':policy['primary_objective'],'policy_commit':'759bcb3','calibration':policy['after'], 'held_out':after,
             'weighted_75_25':prior_weighted_metrics(after),'sensitivity':policy['sweep'],'dataset':'rajpurkar/squad_v2 validation',
             'independent_of_checkpoint_selection':True,'calibration_ids':policy['calibration_ids'],'held_out_ids':[r['id'] for r in held]}
    Path('logs/subset12k/evaluation.json').write_text(json.dumps(current,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
