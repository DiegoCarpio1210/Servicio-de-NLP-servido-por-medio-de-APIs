"""Calibration-only selection; previously observed development diagnostics are separate."""
import json,hashlib
from pathlib import Path
import numpy as np
from qa.metrics import score,summarize,prior_weighted_metrics,weighted_threshold_sweep
from qa.spans import decide

ROOT=Path(__file__).resolve().parents[1];PILOT=ROOT/'logs/deberta_lora/pilot200';OUT=ROOT/'logs/deberta_lora/calibration_analysis'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')

def measure(rows,t):
    groups=summarize(rows,t)
    return {'threshold':t,'groups':groups,'weighted_75_25':prior_weighted_metrics(groups),
      'abstention_rate':100*sum(decide(r,t)['answer'] is None for r in rows)/len(rows)}

def select_threshold(rows):
    sweep=[measure(rows,x['threshold']) for x in weighted_threshold_sweep(rows,.75)]
    feasible=[m for m in sweep if m['groups']['unanswerable']['EM']>=95.]
    def key(m):return (round(m['weighted_75_25']['EM'],10),round(m['weighted_75_25']['F1'],10),round(m['groups']['unanswerable']['EM'],10),-m['threshold'])
    selected=max(feasible,key=key)
    frontier=[]
    for m in sweep:
        a,n=m['groups']['answerable']['EM'],m['groups']['unanswerable']['EM']
        if any((x['groups']['answerable']['EM']>=a and x['groups']['unanswerable']['EM']>=n) and
          (x['groups']['answerable']['EM']>a or x['groups']['unanswerable']['EM']>n) for x in sweep):continue
        matching=next((i for i,x in enumerate(frontier) if x['groups']['answerable']['EM']==a and x['groups']['unanswerable']['EM']==n),None)
        if matching is None:frontier.append(m)
        elif key(m)>key(frontier[matching]):frontier[matching]=m
    return selected,sweep,frontier

def breakdown(rows,t):
    yes=[r for r in rows if r['references']];no=[r for r in rows if not r['references']]
    buckets={'answerable_null':0,'answerable_exact':0,'answerable_partial_nonnull':0,'answerable_zero_overlap_nonnull':0}
    for r in yes:
        answer=decide(r,t)['answer'];em,f1=score(answer or '',r['references'])
        k='answerable_null' if answer is None else ('answerable_exact' if em else ('answerable_partial_nonnull' if f1 else 'answerable_zero_overlap_nonnull'))
        buckets[k]+=1
    buckets.update(answerable_nonnull_inexact=buckets['answerable_partial_nonnull']+buckets['answerable_zero_overlap_nonnull'],
      impossible_correct_null=sum(decide(r,t)['answer'] is None for r in no),
      candidate_answerable_EM=100*np.mean([score(r['text'],r['references'])[0] for r in yes]),
      candidate_answerable_F1=100*np.mean([score(r['text'],r['references'])[1] for r in yes]),
      exact_candidate_rejected=sum(decide(r,t)['answer'] is None and score(r['text'],r['references'])[0]==1 for r in yes),
      margin_quantiles={name:np.percentile([r['margin'] for r in subset],[0,25,50,75,100]).tolist() for name,subset in [('answerable',yes),('unanswerable',no)]})
    return buckets

def confidence(systems):
    first=systems[0][0];yes=np.array([i for i,r in enumerate(first) if r['references']]);no=np.array([i for i,r in enumerate(first) if not r['references']])
    rng=np.random.default_rng(20261031);iy=rng.choice(yes,(3000,len(yes)));inn=rng.choice(no,(3000,len(no)))
    samples=[];intervals=[]
    for rows,t in systems:
        assert [r['id'] for r in rows]==[r['id'] for r in first]
        values=np.array([score(decide(r,t)['answer'] or '',r['references']) for r in rows])*100
        a=values[iy].mean(axis=1);n=values[inn].mean(axis=1)
        draws={'EM':(a[:,0]+n[:,0])/2,'F1':(a[:,1]+n[:,1])/2,'answerable_EM':a[:,0],
          'unanswerable_EM':n[:,0],'weighted_EM':.75*a[:,0]+.25*n[:,0],'weighted_F1':.75*a[:,1]+.25*n[:,1]}
        samples.append(draws);intervals.append({k:np.percentile(v,[2.5,97.5]).tolist() for k,v in draws.items()})
    paired={k:np.percentile(samples[2][k]-samples[1][k],[2.5,97.5]).tolist() for k in samples[0]}
    return {'systems_95CI':intervals,'step200_minus_original_both_calibrated_95CI':paired}

def main():
    assert read(OUT/'completed.json')['completed']
    assert hashlib.sha256((ROOT/'experiments/lora_calibration_protocol.json').read_bytes()).hexdigest()==read(OUT/'partition_audit.json')['protocol_sha256']
    protocol=read(ROOT/'experiments/lora_calibration_protocol.json');t0=protocol['original_threshold']
    cal={k:read(OUT/(k+'_calibration.json')) for k in ('original','step200')}
    dev={s:read(PILOT/f'validation-{s:04d}.json')['predictions'] for s in (0,100,200)}
    thresholds={};frontiers={};calibration={}
    for name,rows in cal.items():
        best,sweep,frontier=select_threshold(rows);thresholds[name]=best['threshold'];frontiers[name]=frontier
        write(OUT/(name+'_threshold_sweep.json'),sweep);calibration[name]={'original_policy':measure(rows,t0),'calibrated':best}
    write(OUT/'frozen_experimental_thresholds.json',{'protocol':protocol,'thresholds':thresholds,'weights_frozen':True,'selection_data':'calibration_reserved only'})
    diagnostics={str(s):breakdown(rows,t0) for s,rows in dev.items()}
    delta={'changed_candidate_text_or_start':sum((a['text'],a['start'])!=(b['text'],b['start']) for a,b in zip(dev[0],dev[200])),
      'changed_answerable_candidate_spans':sum(bool(a['references']) and (a['text'],a['start'])!=(b['text'],b['start']) for a,b in zip(dev[0],dev[200])),
      'changed_decisions_at_original_threshold':sum((decide(a,t0)['answer'] is None)!=(decide(b,t0)['answer'] is None) for a,b in zip(dev[0],dev[200])),
      'margin_change_mean':{group:float(np.mean([b['margin']-a['margin'] for a,b in zip(dev[0],dev[200]) if bool(a['references'])==flag])) for group,flag in [('answerable',True),('unanswerable',False)]}}
    exploratory={'original_policy':measure(dev[0],t0),'original_calibrated':measure(dev[0],thresholds['original']),
      'step200_original_policy':measure(dev[200],t0),'step200_calibrated':measure(dev[200],thresholds['step200'])}
    uncertainty={
      'calibration':confidence([(cal['original'],t0),(cal['original'],thresholds['original']),(cal['step200'],thresholds['step200'])]),
      'observed_development':confidence([(dev[0],t0),(dev[0],thresholds['original']),(dev[200],thresholds['step200'])])}
    summary={'pilot_verified':True,'thresholds':thresholds,'calibration':calibration,'development_original_policy_breakdown':diagnostics,
      'development_calibrated_breakdown':{'original':breakdown(dev[0],thresholds['original']),'step200':breakdown(dev[200],thresholds['step200'])},
      'calibration_margin_distributions':{k:breakdown(rows,t0)['margin_quantiles'] for k,rows in cal.items()},
      'development_changes_step0_to_200':delta,'observed_development':exploratory,'confidence':uncertainty,'pareto':frontiers,
      'loading':read(OUT/'loading_verification.json'),'integrity':read(OUT/'completed.json'),'new_training_updates':0,
      'independent_final_evaluation':None,'production_changed':False}
    write(ROOT/'logs/deberta_lora_calibration_summary.json',summary)
    lines=['# DeBERTa LoRA answerability and calibration analysis','',
      'This is an experimental calibration study, not a production activation or an independent final test. No training occurred. Main, original DeBERTa, quality40k, historical predictions, all LoRA checkpoints, API and dashboard are preserved.',
      '', '## Completed pilot verified','',
      'The saved pilot completed 200/200 optimizer updates: 461.20 seconds optimizer-loop time, 523.11 seconds training-phase time. Checkpoints 0000, 0025, 0028, 0050, 0075, 0100, 0125, 0150, 0175 and 0200 exist. Recovery from update 28 restored optimizer, scheduler, cursor and RNG. The predeclared development weighted-EM selector retained step 0. See [pilot report](DEBERTA_LORA_PILOT200.md) for measured memory, losses and provenance.',
      '', '## Data roles and limitations','',
      'Training used 1,024 examples. Previously observed development contains 128 examples (64 answerable/64 impossible); reserved calibration contains a different 128 (64/64). All 1,280 selected normalized contexts/questions and IDs are distinct. The training implementation consumes train only; checkpoint scoring consumes development only, confirmed by prediction IDs. Calibration was not used for the pilot or checkpoint selection. In this new study it is used only for threshold selection.',
      '', 'Answerability is confounded with source: AdversarialQA supplies all positive examples and SQuAD2 all negatives. SQuAD2 was used by the vendor; AdversarialQA shares Wikipedia/SQuAD contexts. Historical benchmark IDs/context exclusions are retained, but universal vendor exposure or near-duplicate independence cannot be certified. The old 500/1,024 benchmarks and public samples were not used for this tuning or rerun. No credible independent final set is available locally; none is invented here.',
      '', '## Actual error breakdown at the frozen threshold','',
      '| Step | Answerable null | Exact non-null | Partial non-null | Zero-overlap non-null | All inexact non-null | Correct impossible null | Exact candidate rejected |',
      '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s,b in diagnostics.items():lines.append(f"| {s} | {b['answerable_null']}/64 | {b['answerable_exact']}/64 | {b['answerable_partial_nonnull']}/64 | {b['answerable_zero_overlap_nonnull']}/64 | {b['answerable_nonnull_inexact']}/64 | {b['impossible_correct_null']}/64 | {b['exact_candidate_rejected']} |")
    lines += ['', 'The null/exact/partial/zero-overlap answerable columns are mutually exclusive; the inexact column sums partial and zero-overlap. Partial means positive token F1 below 1. The threshold rejects useful spans, but extraction quality is also limiting:',
      '', '| Step | Candidate answerable EM before abstention | Candidate answerable F1 before abstention |','|---:|---:|---:|']
    for s,b in diagnostics.items():lines.append(f"| {s} | {b['candidate_answerable_EM']:.2f}% | {b['candidate_answerable_F1']:.2f}% |")
    lines += ['',f"Step 200 changed {delta['changed_candidate_text_or_start']} candidate text/start pairs and {delta['changed_decisions_at_original_threshold']} null decisions versus step 0. Mean margin shifts: answerable {delta['margin_change_mean']['answerable']:+.4f}, impossible {delta['margin_change_mean']['unanswerable']:+.4f}. These measured shifts must not be mistaken for improved span accuracy.",
      '', 'All 64 answerable candidate text/start spans were identical at steps 0 and 200; the one changed candidate belongs to an impossible question. Every stored candidate span and returned offset was checked against the supplied original development context. Current GPU inference reproduced three saved examples per state (one previously abstained answerable, one answered example and one impossible) within 0.0002 margin tolerance. All 50 saved adapter/QA-head tensors at steps 100 and 200 were checked against active loaded tensors. The loaded QA head differs from the original at both trained checkpoints; step 0 has the original head and zero LoRA B matrices. There is no silent fallback to the base model.',
      '', '## Margin convention and threshold selection','',
      'The project computes margin = minimum window CLS(start+end) score minus the best valid span score. `qa.spans.decide` returns null when `start is None OR margin > threshold`; equality answers. Increasing the threshold reduces abstention. The API score is sigmoid(abs(margin-threshold)), a distance signal rather than a calibrated correctness probability.',
      '', 'Before reserved inference, commit `a9ed983` declared: maximize calibration 75/25 weighted EM subject to empirical unanswerable EM >=95%; ties maximize weighted F1, then unanswerable EM, then choose the lower threshold. Comparisons round metrics to 10 decimals to avoid numerical tie noise. The existing exact weighted_threshold_sweep tests minimum margin minus one and each distinct margin. Models, span policy and candidate lists stay fixed; thresholds are stored only under the ignored experimental directory.',
      '',f"Selected original threshold: **{thresholds['original']:.12f}**. Selected step-200 threshold: **{thresholds['step200']:.12f}**. Production remains **{t0:.12f}**. With 64 negatives, the 95% floor permits at most three errors; it is an empirical constraint, not a population guarantee.",
      '', '## Calibration results — 128 examples, used for threshold fitting','',
      '| System | Overall EM | Overall F1 | Answerable EM | Answerable F1 | Impossible EM | Weighted EM | Weighted F1 |','|---|---:|---:|---:|---:|---:|---:|']
    def row(label,m):
        g=m['groups'];return f"| {label} | {g['all']['EM']:.2f} | {g['all']['F1']:.2f} | {g['answerable']['EM']:.2f} | {g['answerable']['F1']:.2f} | {g['unanswerable']['EM']:.2f} | {m['weighted_75_25']['EM']:.2f} | {m['weighted_75_25']['F1']:.2f} |"
    for name,m in calibration.items():
        lines += [row(name+' · original threshold',m['original_policy']),row(name+' · calibrated',m['calibrated'])]
    for name,m in calibration.items():
        before,after=m['original_policy']['groups']['all'],m['calibrated']['groups']['all']
        lines.append(f"\n{name} calibration-set effect: EM {after['EM']-before['EM']:+.2f} points; F1 {after['F1']-before['F1']:+.2f} points. These are fitted-set changes.")
    lines += ['', '## Previously observed development diagnostics — 128 examples','',
      'These examples were previously inspected and used for checkpoint selection. They did not select the new thresholds and are not a fresh independent final test.',
      '', '| System | Overall EM | Overall F1 | Answerable EM | Answerable F1 | Impossible EM | Weighted EM | Weighted F1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in exploratory.items():lines.append(row(name,m))
    for name,before,after in [('Original',exploratory['original_policy'],exploratory['original_calibrated']),('Step200',exploratory['step200_original_policy'],exploratory['step200_calibrated'])]:
        lines.append(f"\n{name} calibration effect on the same observed development examples: EM {after['groups']['all']['EM']-before['groups']['all']['EM']:+.2f} points; F1 {after['groups']['all']['F1']-before['groups']['all']['F1']:+.2f} points.")
    lines += ['', '## Pareto tradeoff on calibration','', '| Model | Threshold | Answerable EM | Impossible EM | Meets 95% floor |','|---|---:|---:|---:|---|']
    for name,points in frontiers.items():
        for m in points:lines.append(f"| {name} | {m['threshold']:.6f} | {m['groups']['answerable']['EM']:.2f} | {m['groups']['unanswerable']['EM']:.2f} | {'yes' if m['groups']['unanswerable']['EM']>=95 else 'no'} |")
    lines += ['', 'Only nondominated answerable-EM / impossible-EM operating points are shown; tied pairs retain the declared F1/conservatism preference. Points below the floor are not selected. Full threshold sweeps and raw predictions remain local.',
      '', '## Margin distributions (min / Q1 / median / Q3 / max)','', '| Cohort / model / class | Margin quantiles |','|---|---|']
    for label,distributions in [(f'development step {s}',b['margin_quantiles']) for s,b in diagnostics.items()]+[(f'calibration {k}',v) for k,v in summary['calibration_margin_distributions'].items()]:
        for group,v in distributions.items():lines.append('| '+label+' / '+group+' | '+' / '.join(f'{x:.3f}' for x in v)+' |')
    lines += ['', '## Uncertainty and direct paired comparisons','',
      'Stratified paired bootstrap: 3,000 resamples, seed 20261031, retaining 64 examples per class and sharing sampled indices across systems. Intervals hold fitted thresholds fixed. Calibration intervals therefore omit threshold-selection uncertainty; development intervals also cannot erase earlier model selection or source confounding. These are descriptive, not independent generalization guarantees.',
      '', '| Cohort | Calibrated LoRA minus calibrated original EM 95% CI | F1 95% CI | Weighted EM 95% CI |','|---|---|---|']
    for name,u in uncertainty.items():
        ci=u['step200_minus_original_both_calibrated_95CI'];lines.append('| '+name+' | '+' | '.join(f'[{ci[k][0]:+.2f}, {ci[k][1]:+.2f}]' for k in ('EM','F1','weighted_EM'))+' |')
    a=exploratory['original_calibrated'];b=exploratory['step200_calibrated']
    lines += ['',f"On observed development, calibrated LoRA minus calibrated original: EM {b['groups']['all']['EM']-a['groups']['all']['EM']:+.2f} points; F1 {b['groups']['all']['F1']-a['groups']['all']['F1']:+.2f} points; weighted EM {b['weighted_75_25']['EM']-a['weighted_75_25']['EM']:+.2f} points.",
      '', '## Conclusion and next step','',
      'Best experimental candidate by the declared calibration objective is the original model at threshold 14.58487319946289: calibration weighted EM 50.00%, versus 49.61% for calibrated LoRA. This preference uses calibration only, not the development comparison. It remains a provisional experimental policy, not a deployment recommendation.',
      'The frozen threshold over-abstains on this hard mixed-source cohort, but no threshold can fix an incorrect candidate span. Use the measured tables, not falling training loss, to judge the adapter. The pilot did not improve candidate-span EM/F1 on development and did not win its preregistered checkpoint criterion. There is no evidence here warranting production LoRA activation or another immediate training run.',
      '', 'Keep the stable model and threshold. Treat the fitted original and LoRA policies as experimental candidates only. Obtain a new source-balanced, independently annotated QA set with both answerable and impossible questions within each source, audit context/question overlap, lock one candidate policy before opening it, and evaluate once. Include SQuAD positives in a future mixed-source training design rather than confounding source with answerability. Another LoRA run should wait for a defensible data design and predeclared external evaluation.',
      '', 'The complete regression suite, including deterministic selection, strict threshold equality, empirical null-floor constraints, error partitions and paired-bootstrap checks, passed 72 tests (logs/deberta_lora_calibration_tests.txt). Frozen calibration inference used a recorded peak of 790.01 MiB allocated / 800 MiB reserved on the P1000. No gradient updates were performed.',
      '', 'Full results and all fixed-threshold confidence intervals: logs/deberta_lora_calibration_summary.json. Raw predictions, threshold sweeps, loaded-tensor verification and experimental thresholds are local under logs/deberta_lora/calibration_analysis/. No historical evaluation was overwritten, no training was restarted, and no adapter or experimental threshold was activated.']
    (ROOT/'docs/DEBERTA_LORA_CALIBRATION_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'thresholds':thresholds,'calibration':calibration,'development':exploratory,'paired_dev':uncertainty['observed_development']['step200_minus_original_both_calibrated_95CI']},indent=2))

if __name__=='__main__':main()
