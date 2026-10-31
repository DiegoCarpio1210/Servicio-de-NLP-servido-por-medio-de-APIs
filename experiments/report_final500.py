"""Descriptive reporting only after all fixed-profile predictions are complete."""
import csv,json
from pathlib import Path
import numpy as np
from experiments.final500 import ROOT,OUT,MODELS,read,atomic,sha

def confidence(rows_by_model):
    first=rows_by_model[MODELS[0]]
    yes=np.array([i for i,r in enumerate(first) if r['references']]);no=np.array([i for i,r in enumerate(first) if not r['references']])
    assert len(yes)==len(no)==250
    rng=np.random.default_rng(50020261030)
    iy=rng.choice(yes,(3000,250));inn=rng.choice(no,(3000,250))
    intervals={};samples={}
    for key,rows in rows_by_model.items():
        assert [r['id'] for r in rows]==[r['id'] for r in first]
        em=np.array([r['EM'] for r in rows]);f1=np.array([r['F1'] for r in rows]);abst=np.array([100*(r['response']['answer'] is None) for r in rows])
        a=em[iy].mean(axis=1);n=em[inn].mean(axis=1);af=f1[iy].mean(axis=1);nf=f1[inn].mean(axis=1)
        draws={'EM':(a+n)/2,'F1':(af+nf)/2,'answerable_EM':a,'answerable_F1':af,'unanswerable_EM':n,
          'weighted_EM_75_25':.75*a+.25*n,'abstention_rate':(abst[iy].mean(axis=1)+abst[inn].mean(axis=1))/2}
        intervals[key]={name:np.percentile(v,[2.5,97.5]).tolist() for name,v in draws.items()};samples[key]=draws['weighted_EM_75_25']
    paired={key:np.percentile(v-samples['quality40k'],[2.5,97.5]).tolist() for key,v in samples.items() if key!='quality40k'}
    return intervals,paired

def main():
    manifest=read(OUT/'manifest.json');results={key:read(OUT/key/'summary.json') for key in MODELS};rows={key:read(OUT/key/'predictions.json') for key in MODELS}
    for k,v in results.items():
        assert v['manifest_sha256']==sha(OUT/'manifest.json') and len(rows[k])==500
        assert [r['id'] for r in rows[k]]==manifest['ids']
        assert v['all_offsets_valid'] and all(r['offset_valid'] for r in rows[k])
        for metric in ('EM','F1'):
            assert abs(sum(r[metric] for r in rows[k])/500-v['metrics']['groups']['all'][metric])<1e-8
    active=read(ROOT/'configs/active_model.json')
    assert all(sha(ROOT/active['model_dir']/name)==digest for name,digest in active['hashes'].items())
    intervals,paired=confidence(rows);previous=read(ROOT/'logs/pretrained_benchmark/comparison.json')['models']
    for k,v in results.items():
        v['confidence_intervals_95']=intervals[k]
        api=read(OUT/k/'http.json');v['concurrent_outputs_equal_sequential']=all(a.get('body')==b.get('body') for a,b in zip(api['sequential']['rows'],api['concurrent']['rows']))
        v['api_compatible']=bool(api['contract_pass'] and api['sequential']['pass'] and api['concurrent']['pass'] and api['server_stopped'])
    recommendation='RECOMMENDED FOR ACTIVATION' if results[MODELS[0]]['all_targets'] and results[MODELS[0]]['api_compatible'] else 'DeBERTa is NOT recommended for activation under the requested all-targets rule.'
    if not any(v['all_targets'] for v in results.values()):recommendation+=' None of the three models fully satisfies all four requested requirements.'
    export=OUT/'per_question.csv'
    with export.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','question','expected_answers',*sum(([k+'_answer',k+'_start',k+'_score',k+'_EM',k+'_F1',k+'_seconds'] for k in MODELS),[])])
        for i,r in enumerate(rows[MODELS[0]]):
            line=[r['id'],r['question'],json.dumps(r['references'],ensure_ascii=False)]
            for k in MODELS:
                p=rows[k][i];line += [json.dumps(p['response']['answer'],ensure_ascii=False),p['response']['start'],p['response']['score'],p['EM'],p['F1'],p['seconds']]
            writer.writerow(line)
    summary={'models':results,'audit':manifest['audit'],'manifest_sha256':sha(OUT/'manifest.json'),
      'paired_weighted_EM_gain_vs_quality40k_95CI':paired,'recommendation':recommendation,'no_recalibration':True,
      'all_offsets_valid':True,'production_unchanged_during_benchmark':True,'raw_csv_sha256':sha(export),'limitations':manifest['limitations']}
    atomic(ROOT/'logs/final500_summary.json',summary)
    lines=['# Final 500-question evaluation','',recommendation,'',
      'During the benchmark quality40k remained active; no model settings were changed or training performed. Subsequently the user explicitly authorized DeBERTa weighted_EM as the stable baseline despite the unmet targets. See STABLE_DEBERTA.md.',
      '', '## Dataset and overlap audit','',
      f"Exactly **500 English SQuAD 2.0 validation examples: 250 answerable and 250 unanswerable**. Seed `{manifest['seed']}`; fingerprint `{manifest['fingerprint']}`. Selection was frozen in commit `bb9cacd`, before predictions. Ground truth is the original answer text and answer_start annotations, never generated labels.",
      '',f"The audit scanned {manifest['audit']['scanned_json_files']} available JSON records and found {manifest['audit']['historical_validation_ids']} previously referenced validation IDs. The eligible pool contained {manifest['audit']['eligible_unseen_id_and_normalized_question_count']} examples with unused IDs and normalized questions. The selected sample has **zero prior ID overlap, zero normalized-question overlap, but 500/500 previously used contexts**. Each selected example has a different context. No unused validation context remained. Context novelty was preferred before seeded sampling; none was available.",
      '', 'This is not a completely unseen or official instructor test. Vendor model selection and previous local work used SQuAD validation. The old abandoned full-data run has unknown consumed IDs. JSON records cannot establish universal historical non-exposure. Using the official training holdout instead would contaminate comparison with the SQuAD-pretrained models.',
      '', '## Frozen configurations','',
      'The prior benchmark primary profiles are retained: weighted_EM for both Deepset models and active for quality40k. The alternative null95 calibration operating points were not substituted after seeing this sample. No threshold is fitted to these 500 examples. All models use 384-token windows, stride 128 and two CPU threads, in fresh sequential processes.',
      '', '| Model | Threshold | Span policy |','|---|---:|---|']
    for k in MODELS:
        p=manifest['profiles'][k];lines.append(f"| {k} | {p['threshold']:.12f} | {json.dumps(p['policy'])} |")
    lines+=['','Weights and source-policy SHA-256 values are in `configs/final500_manifest.json`; local raw records are under `logs/final500/`. All three use precisely the same example IDs. Production settings and dashboard were preserved.','',
      '## Metric definitions and comparison','',
      'EM uses official SQuAD normalization (case, punctuation, articles, whitespace). F1 measures normalized token overlap, taking the maximum over original references. Null references are scored against an empty answer. Group scores use their own 250-example denominators; overall metrics weight both groups equally. Weighted EM is 0.75 × answerable EM + 0.25 × unanswerable EM. Abstention rate is the fraction of null predictions, not null-class accuracy. Scores are threshold-distance signals, not calibrated correctness probabilities.',
      '', '| Model | Overall EM | Overall F1 | Answerable EM | Answerable F1 | Unanswerable EM | Weighted EM 75/25 | Abstention |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for k,v in results.items():
        m=v['metrics'];g=m['groups'];lines.append(f"| {k} | {g['all']['EM']:.2f}% | {g['all']['F1']:.2f}% | {g['answerable']['EM']:.2f}% | {g['answerable']['F1']:.2f}% | {g['unanswerable']['EM']:.2f}% | {m['weighted_75_25']['EM']:.2f}% | {100*m['abstention_rate']:.2f}% |")
    lines+=['','## Independent target checks','','Every required target must pass; a high average does not compensate for a failed class target. These are the thresholds requested by the user for this decision.','',
      '| Model | EM ≥85 | F1 ≥85 | Answerable EM ≥80 | Unanswerable EM ≥95 | All four |','|---|---|---|---|---|---|']
    for k,v in results.items():lines.append('| '+k+' | '+' | '.join('PASS' if b else 'FAIL' for b in [*v['targets'].values(),v['all_targets']])+' |')
    lines+=['','## Confidence intervals','',
      '95% percentile bootstrap intervals use 3,000 resamples, seed 50020261030, stratified to retain 250 examples per class. Each question has a distinct context. The same resampled indices are used for paired model comparisons. These describe sampling uncertainty conditional on this historically exposed validation population, not protection against domain shift or prior selection bias.',
      '', '| Model | EM 95% CI | F1 95% CI | Answerable EM 95% CI | Unanswerable EM 95% CI | Weighted EM 95% CI |','|---|---|---|---|---|---|']
    for k in MODELS:lines.append('| '+k+' | '+' | '.join(f'[{intervals[k][n][0]:.2f}, {intervals[k][n][1]:.2f}]' for n in ['EM','F1','answerable_EM','unanswerable_EM','weighted_EM_75_25'])+' |')
    for k,ci in paired.items():lines.append(f'\nPaired weighted-EM gain versus quality40k, {k}: 95% CI [{ci[0]:.2f}, {ci[1]:.2f}] percentage points.')
    lines+=['','## Comparison with the previous 1,024-example benchmark','',
      'These cohorts differ. Changes below are descriptive, not paired improvement estimates or evidence of retraining. The same primary thresholds and policies were used.',
      '', '| Model | Previous EM / F1 | Current EM / F1 | EM / F1 difference (points) |','|---|---|---|---|']
    for k,v in results.items():
        old=previous[k]['metrics']['groups']['all'];new=v['metrics']['groups']['all']
        lines.append(f"| {k} | {old['EM']:.2f} / {old['F1']:.2f} | {new['EM']:.2f} / {new['F1']:.2f} | {new['EM']-old['EM']:+.2f} / {new['F1']-old['F1']:+.2f} |")
    lines+=['','## CPU latency and RAM','',
      'Every one of the 500 predictions was timed after an untimed warmup, including tokenization, span extraction and the decision. Model loading is excluded. RAM is peak process working set, including Python and dataset overhead. This is one measured run per model, not a speed confidence interval.',
      '', '| Model | Mean / p95 / max seconds | Peak RAM MiB |','|---|---|---:|']
    for k,v in results.items():
        t=v['latency'];lines.append(f"| {k} | {t['mean_seconds']:.3f} / {t['p95_seconds']:.3f} / {t['max_seconds']:.3f} | {v['peak_rss_mib']:.1f} |")
    lines+=['','## API reliability and concurrency','',
      'The previous quality40k bounded-concurrency run recorded one HTTP failure. A direct two-thread tokenizer reproduction produced RuntimeError: Already borrowed. The inference lock was extended to cover tokenization as well as model forward execution, for both production and benchmark engines. This is a thread-safety correction; weights, mathematical span selection and thresholds were not changed. Regression tests cover simultaneous calls.',
      '', 'Tests use the real shared FastAPI service on loopback port 8000 and a random in-memory token. Health, authenticated info with model hash, exact authentication rejection, schema errors, null responses, Unicode and long-context offsets are checked. Each performance mode sends 50 requests; bounded concurrency means two simultaneous clients, not fifty simultaneous clients. All prediction JSONs must have exactly id, answer, start and score. Public samples are repeated for HTTP load only and never tune the models.',
      '', '| Model | Contract | Sequential valid / total; seconds; max | Concurrent valid / total; seconds; max | Limits pass |','|---|---|---|---|---|']
    for k,v in results.items():
        a=v['http'];s=a['sequential'];c=a['concurrent'];lines.append(f"| {k} | {'PASS' if a['contract_pass'] else 'FAIL'} | {s['valid']}/50; {s['seconds']:.2f}; {s['maximum_seconds']:.3f} | {c['valid']}/50; {c['seconds']:.2f}; {c['maximum_seconds']:.3f} | {'PASS' if v['api_compatible'] else 'FAIL'} |")
    lines+=['','Five seconds per request and 150 seconds per mode are checked independently of accuracy. HTTP response details, including any failure status/body, are retained. Temporary servers were stopped. No public tunnel was opened.','',
      '## Per-question predictions and raw artifacts','',
      'All 500 individual rows, original expected answers, predicted text/null, offsets, scores, EM/F1 and latency are available locally in `logs/final500/per_question.csv`. Each model also has `predictions.json`, `http.json` and `summary.json`. These files are deliberately excluded from Git along with the dataset and weights; they are not missing generated results. Run the committed scripts with the local artifacts to regenerate the report. The compact public summary is `logs/final500_summary.json`.',
      '',f'CSV SHA-256: `{sha(export)}`.',
      '', 'The first five rows in the frozen order (not selected for correctness) are shown below. The complete local CSV contains every question and expected answer.',
      '', '| Question ID | Original references | DeBERTa answer | RoBERTa answer | quality40k answer |','|---|---|---|---|---|']
    def cell(x):return json.dumps(x,ensure_ascii=False).replace('|','\\|').replace('\n',' ')
    for i in range(5):lines.append('| '+rows[MODELS[0]][i]['id']+' | '+cell(rows[MODELS[0]][i]['references'])+' | '+' | '.join(cell(rows[k][i]['response']['answer']) for k in MODELS)+' |')
    lines+=['','## Recommendation and remaining limitations','',recommendation,'',
      'The user subsequently approved DeBERTa as a stable baseline; quality40k remains operational as a rollback option. A statistical gain over quality40k does not imply meeting all four absolute targets. No thresholds will be changed after this evaluation. Known domain mismatch, historical context exposure and imperfect no-answer recognition remain relevant. No submission tag was created.']
    (ROOT/'docs/FINAL_500_EVALUATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'recommendation':recommendation,'results':{k:{'metrics':v['metrics'],'targets':v['targets'],'http':v['api_compatible']} for k,v in results.items()}},indent=2))

if __name__=='__main__':main()
