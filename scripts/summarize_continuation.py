"""Publish completed evidence, retaining the calibration-locked candidate decision."""
import json
import re
import statistics
from pathlib import Path


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    root=Path('logs/continued20k'); lock=read(root/'locked_candidate.json'); comparison=read(root/'comparison.json')
    training=read(root/'training_metadata.json'); selected=lock['selected']; profile=lock['profiles'][selected]
    if not training.get('completed'): raise RuntimeError('Stopped run needs an explicit partial-run report')
    candidate_dir=Path(profile['model_dir']); evidence=Path('logs')/candidate_dir.name
    public=read(evidence/'public.json'); info=read(evidence/'info.json')
    assert info['no_answer_threshold']==profile['threshold']
    scores=comparison['profiles'][selected]['fresh_holdout']; groups=scores['groups']
    latencies=[r['seconds'] for r in public['rows']]
    correct=sum(r['correct'] for r in public['rows'][:5])
    summary={'selected':selected,'model_dir':str(candidate_dir),'selection_source':'locked calibration decision',
             'fresh_holdout':scores,'public_correct':correct,'cpu_mean_seconds':statistics.mean(latencies),
             'cpu_max_seconds':max(latencies),'http_50_seconds':public['elapsed_seconds'],
             'timeout_budget_pass':public['timeout_budget_pass'],'continued_training_seconds':training['elapsed_seconds'],
             'continued_training_completed':training['completed'],'no_further_training_started':True,'submission_ready':False}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    adequate=groups['all']['F1']>=60 and groups['answerable']['F1']>=50 and groups['unanswerable']['EM']>=70
    Path('configs/candidate.json').write_text(json.dumps({'model_dir':str(candidate_dir).replace('\\','/'),
         'profile':selected,'threshold':profile['threshold'],'decision':'provisional_candidate' if adequate else 'needs_improvement',
         'evidence':'logs/continued20k/summary.json','selection_evidence':'logs/continued20k/locked_candidate.json',
         'submission_ready':False},indent=2))
    table=['| Profile | Overall EM | Overall F1 | Answerable EM | Answerable F1 | Impossible EM | 75/25 EM | 75/25 F1 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,result in comparison['profiles'].items():
        m=result['fresh_holdout']; g=m['groups']; w=m['weighted_75_25']
        values=[g['all']['EM'],g['all']['F1'],g['answerable']['EM'],g['answerable']['F1'],g['unanswerable']['EM'],w['EM'],w['F1']]
        table.append('| '+name+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    result_text='\n'.join(table)
    experiment=Path('docs/CONTINUATION_EXPERIMENT.md')
    experiment.write_text(experiment.read_text(encoding='utf-8').replace(
        'Status: training launched; results pending. Do not confuse the planned run with completed evidence.',
        'Status: the single authorized continuation completed. Measured results appear below.'),encoding='utf-8')
    with experiment.open('a',encoding='utf-8') as f:
        f.write(f"\n## Completed results\n\nThe run completed flag is {training['completed']}; actual epochs {training['actual_epochs']:.3f}, "
                f"{training['optimizer_steps']} steps, {training['elapsed_seconds']/60:.2f} minutes. Best checkpoint: {training['best_checkpoint']}.\n\n"
                f"Candidate fixed on calibration before final evaluation: **{selected}**. Fresh holdout comparison:\n\n{result_text}\n\n"
                f"Public cases: {correct}/5. CPU 50 requests: {public['elapsed_seconds']:.2f}s; max {max(latencies):.3f}s. "
                "See comparison.json for the previously consulted holdout and paired bootstrap intervals. No configuration was adjusted from these final results.\n")
    report=Path('REPORT.md').read_text(encoding='utf-8')
    def section(number,body):
        nonlocal report
        report=re.sub(rf'(^## {number}\.[^\n]*\n).*?(?=^## \d+\.|\Z)',lambda m:m.group(1)+'\n'+body.strip()+'\n\n',report,flags=re.M|re.S)
    report=re.sub(r'Draft with completed.*?Final submission is PENDING\.',
                  'Draft with measured baseline and continuation results. Final submission is PENDING.',report,count=1)
    section(2,f'''Initial run: 12,000 SQuAD 2.0 examples, two epochs, 1,522 steps, 26m22s.
Continuation: nested 20,000 examples (12,000 previous + 8,000 new; 6,675 impossible),
{training['actual_epochs']:.2f} additional epochs, {training['optimizer_steps']} steps, {training['elapsed_seconds']/60:.2f} minutes.
Both use the Quadro P1000 4 GiB and train encoder plus QA head. Original weights/checkpoints are preserved.
The continuation export uses the minimum-loss checkpoint at step {training.get('selected_optimizer_steps',training['optimizer_steps'])}
(epoch {training.get('selected_epoch',training['actual_epochs']):.3f}), which can precede the end of the completed run.

| Setting | Initial / continuation |
|---|---|
| Learning rate / warmup | 3e-5 / 10%; then 1e-5 / 5%; linear decay, weight decay .01 |
| Batch / accumulation / precision | 16 / 1 / FP32; length-grouped dynamic padding |
| max_seq_length / doc_stride / seed | 384 / 128 / 42 |
| Validation | 512 selection, 512 calibration, old 1,024 holdout, fresh 1,024 context-disjoint holdout |

[Continuation loss curve](logs/continued20k/loss.png). The candidate **{selected}** was fixed using calibration.
Fresh holdout EM/F1: {groups['all']['EM']:.2f}/{groups['all']['F1']:.2f}; answerable
{groups['answerable']['EM']:.2f}/{groups['answerable']['F1']:.2f}; impossible {groups['unanswerable']['EM']:.2f}.
75/25 weighted EM/F1: {scores['weighted_75_25']['EM']:.2f}/{scores['weighted_75_25']['F1']:.2f}.
The old holdout was already consulted and was not used to optimize this experiment.''')
    section(3,f'''Return null when minimum window CLS start-plus-end minus best valid span score exceeds
**{profile['threshold']:.8f}**. The continuation threshold maximizes calibration 75/25 weighted EM,
with weighted F1 tie-break, subject to impossible calibration EM >=70%. This is a calibration safeguard,
not a guaranteed test result. The candidate and thresholds were locked before final evaluations.
Public examples never train or calibrate the model. EM/F1 are proxies, not the instructor's partial-credit judge.
See docs/CONTINUATION_EXPERIMENT.md for all three profiles and the historical baseline scoring correction.''')
    report=report.replace("$env:MODEL_DIR = 'models/subset12k'",f"$env:MODEL_DIR = '{candidate_dir.as_posix()}'")
    report=report.replace('and 1,522 training steps','and actual training steps')
    lines=['| Item | Expected | Your answer | `start` | Correct? |','|---|---|---|---|---|']
    for item,row in zip(read('originals/public_sample.json')['items'],public['rows'][:5]):
        response=row['response']; expected='null' if item['is_impossible'] else ' / '.join(a['text'] for a in item['answers'])
        answer=str(response['answer']).replace('|','/').replace('\n',' ') if response['answer'] is not None else 'null'
        lines.append(f"| {item['id']} | {expected} | {answer} | {response['start']} | {'Yes' if row['correct'] else 'No'} |")
    section(5,'\n'.join(lines)+f'\n\nReal HTTP: {correct}/5. Results were observed only after candidate locking; no case-specific fixes.')
    size=sum(p.stat().st_size for p in candidate_dir.rglob('*') if p.is_file())
    section(6,f'''66,364,418 parameters; {size/2**20:.2f} MiB including tokenizer/metadata.
CPU mean {statistics.mean(latencies):.3f}s, maximum {max(latencies):.3f}s; 50 sequential calls {public['elapsed_seconds']:.2f}s.
This repeats five public cases ten times, not fifty hidden questions. Laptop: Intel family 6/model 158,
31.72 GiB RAM. Current-policy HTTPS and independent-network rehearsal remain PENDING;
historical baseline HTTPS took 20.47s and does not validate changed weights/settings.''')
    section(9,'''Model accuracy remains dataset-dependent; weighted scores assume the stated 75/25 mix transfers.
The impossible-class calibration floor is not a deployment guarantee. Confidence is not probability-calibrated;
questions cap at 64 tokens, spans at 30. Training is English-only. The new holdout is now observed and must
not become a tuning set. Instructor dry run, backup tunnel, independent-network checks, weight sharing and
final review remain PENDING. No second continuation or submission tag was created.''')
    Path('REPORT.md').write_text(report,encoding='utf-8')
    Path('docs/STATUS.md').write_text(f'''# Current status — Pimientero

One continuation completed; no additional run or submission tag. Candidate: **{selected}**, `{candidate_dir.as_posix()}`.
Selection was fixed on calibration before either final holdout was evaluated. All metrics below use the fresh 1,024.

{result_text}

Public accuracy: {correct}/5. Fifty local CPU requests: {public['elapsed_seconds']:.2f}s; maximum {max(latencies):.3f}s.
Training: original 26m22s plus continuation {training['elapsed_seconds']/60:.2f} minutes. Original weights are preserved.
See CONTINUATION_EXPERIMENT.md, logs/continued20k/comparison.json and COMPLIANCE.md.

Pending: instructor token entered locally, current-model HTTPS and another-network test, backup tunnel,
official dry run and active URL, authorized separate weight distribution, pitch/Q&A rehearsal and final review.
The API test process was stopped after verification. Start it with the README commands and the candidate model path.
No further training is authorized automatically. Submission is not declared complete.
''',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
