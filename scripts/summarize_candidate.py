"""Publish measured 12k evidence into the draft report; no synthetic or inferred scores."""
import json,re,statistics
from pathlib import Path


def main():
    root=Path('logs/subset12k')
    m=json.loads((root/'training_metadata.json').read_text())
    e=json.loads((root/'evaluation.json').read_text())
    public=json.loads((root/'public.json').read_text())
    if not m.get('completed') or m['train_size']!=12000 or m['epochs']!=2:
        raise ValueError('Only completed authorized 12k/2-epoch evidence can update the report')
    manifest=json.loads((root/'data_manifest.json').read_text())
    ids=[set(manifest[k+'_ids']) for k in ('train','selection','calibration','holdout')]
    assert all(not a & b for i,a in enumerate(ids) for b in ids[i+1:])
    latency=[r['seconds'] for r in public['rows'] if 'seconds' in r]
    size=sum(p.stat().st_size for p in Path(m['model_dir']).rglob('*') if p.is_file())
    held=e['held_out']; adequate=held['all']['F1']>=60 and held['answerable']['F1']>=50 and held['unanswerable']['EM']>=70
    epoch_minutes=m['elapsed_seconds']/120
    history=json.loads((root/'history.json').read_text())
    losses=[r['eval_loss'] for r in history if 'eval_loss' in r]
    improving=len(losses)>1 and losses[-1]<losses[-2]-.03
    recommendation=('Retain as a provisional candidate; verify the instructor dry run before submission.' if adequate else
                    ('Consider one additional epoch on the same 12k examples before expanding the dataset.' if improving else
                     'Consider a larger representative subset (for example 20k), or inspect validation errors before another epoch.'))
    summary={'training_examples':12000,'epochs':2,'training_seconds':m['elapsed_seconds'],'optimizer_steps':m['optimizer_steps'],
             'validation':held,'threshold':e['threshold'],'cpu_latency_mean_seconds':statistics.mean(latency),
             'cpu_latency_max_seconds':max(latency),'http_50_seconds':public['elapsed_seconds'],
             'timeout_budget_pass':public['timeout_budget_pass'],'public_correct_first_five':sum(r['correct'] for r in public['rows'][:5]),
             'model_disk_bytes':size,'decision':'provisional_candidate' if adequate else 'needs_improvement',
             'decision_basis':'Engineering heuristic on independent validation only; not the professor grading formula. F1 >=60, answerable F1 >=50, unanswerable EM >=70.',
             'recommendation':recommendation,'additional_epoch_estimate_minutes':epoch_minutes,
             'fresh_20k_two_epoch_estimate_minutes':m['elapsed_seconds']/60*20000/12000,
             'automatic_extension':False}
    Path('configs/candidate.json').write_text(json.dumps({'model_dir':m['model_dir'],'decision':summary['decision'],'evidence':'logs/subset12k/summary.json','submission_ready':False},indent=2))
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    report=Path('REPORT.md').read_text(encoding='utf-8')
    report=report.replace('Draft with real pilot results; full-data training is RUNNING, not completed. Final submission is PENDING.',
                          'Draft with completed 12,000-example training and independent validation. Final submission is PENDING.')
    report=report.replace('Draft: the full-data run was stopped; the authorized 12,000-example run is in progress. Final submission is PENDING.',
                          'Draft with completed 12,000-example training and independent validation. Final submission is PENDING.')
    def replace_section(number,body):
        nonlocal report
        pattern=rf'(^## {number}\.[^\n]*\n).*?(?=^## \d+\.|\Z)'
        report=re.sub(pattern,lambda match:match.group(1)+'\n'+body.strip()+'\n\n',report,flags=re.S|re.M)
    replace_section(2,f"""| Item | Value |
|---|---|
| Training data | 12,000 SQuAD 2.0 train examples; {m['splits']['train']['unanswerable']} unanswerable; seed 42 |
| Validation data | 512 checkpoint selection, 512 calibration, 1,024 independent holdout; disjoint IDs |
| Epochs | 2; {m['optimizer_steps']} optimizer steps |
| Learning rate and schedule (warmup?) | 3e-5; linear decay; 10% warmup; weight decay .01 |
| Batch size | 16; accumulation 1; FP32; length-grouped dynamic padding |
| max_seq_length / doc_stride | 384 / 128 |
| Hardware and training time | Quadro P1000 4 GiB; {m['elapsed_seconds']/60:.2f} minutes |

[Real loss curve](logs/subset12k/loss.png). Logs and metadata: `logs/subset12k/`.
Holdout EM/F1: {held['all']['EM']:.2f}/{held['all']['F1']:.2f}; answerable EM/F1:
{held['answerable']['EM']:.2f}/{held['answerable']['F1']:.2f}; unanswerable EM/F1:
{held['unanswerable']['EM']:.2f}/{held['unanswerable']['F1']:.2f}. The earlier full-data run was stopped before its first checkpoint;
its logs and all pilot artifacts were preserved. No full-data completion is claimed.""")
    replace_section(3,f"""Return null when minimum window CLS start-plus-end minus the best valid span score exceeds
**{e['threshold']:.8f}**. The exact sweep maximizes calibration F1, with EM as a tie-break.
The 512 calibration examples are separate from checkpoint selection and the 1,024-example final holdout.
Public cases are never used to fit the model or threshold. Lower thresholds increase abstention;
higher thresholds increase unsupported-span risk. The threshold-sensitivity sweep is saved in evaluation.json.""")
    lines=['| Item | Expected | Your answer | `start` | Correct? |','|---|---|---|---|---|']
    items=json.loads(Path('originals/public_sample.json').read_text())['items']
    for item,row in zip(items,public['rows'][:5]):
        response=row.get('response',{}); answer=response.get('answer')
        expected='null' if item['is_impossible'] else ' / '.join(a['text'] for a in item['answers'])
        value='null' if answer is None else str(answer).replace('|','/').replace('\n',' ')
        lines.append(f"| {item['id']} | {expected} | {value} | {response.get('start')} | {'Yes' if row['correct'] else 'No'} |")
    replace_section(5,'\n'.join(lines)+f"\n\nReal HTTP result: {summary['public_correct_first_five']}/5. Wrong cases remain prediction failures; no public-answer tuning was performed. Full contexts, including the last window, were processed.")
    replace_section(6,f"""| Item | Value |
|---|---|
| Parameters | {m['params']:,} |
| Size on disk | {size/2**20:.2f} MiB including tokenizer and metadata |
| Mean CPU latency | {statistics.mean(latency):.3f} s |
| Maximum CPU latency / 50 sequential requests | {max(latency):.3f} s / {public['elapsed_seconds']:.2f} s |
| Laptop CPU and RAM | Intel family 6/model 158; 31.72 GiB |

This repeats five cases ten times, not the hidden test. Earlier pilot HTTPS timing is preserved separately;
HTTPS rehearsal with this candidate and an independent network remains PENDING.""")
    replace_section(9,f"""{recommendation} These are measured validation results, not a promised grade.
An extra epoch is estimated at {epoch_minutes:.1f} minutes; no extension was started automatically.
Instructor dry run, backup tunnel and final report review remain PENDING. Questions are capped internally at
64 tokens, answers at 30 tokens; confidence is not probability-calibrated. Training is English-only.
Malformed inputs fail validation. The final submission tag has not been created.""")
    report=report.replace("$env:MODEL_DIR = 'models/final'","$env:MODEL_DIR = 'models/subset12k'")
    Path('REPORT.md').write_text(report,encoding='utf-8')
    status=Path('docs/STATUS.md').read_text(encoding='utf-8')
    marker='<!-- SUBSET-RUN-STATUS -->'
    message=f"""{marker}
## Current 12k candidate (supersedes earlier pilot/full-run status)

Completed exactly 12,000 examples / two epochs in {m['elapsed_seconds']/60:.2f} minutes.
Independent holdout EM/F1: {held['all']['EM']:.2f}/{held['all']['F1']:.2f}.
Public cases: {summary['public_correct_first_five']}/5. CPU budget passed: {public['timeout_budget_pass']}.
Decision: **{summary['decision']}**. {recommendation}
No additional training or submission tag was created. See `logs/subset12k/summary.json`.
"""
    status=status.split(marker)[0]
    status=status.replace('- State: RUNNING.', '- State: COMPLETED (candidate still requires review).')
    status=status.replace('| Full two-epoch 12k run | PENDING — RUNNING |', '| Full two-epoch 12k run | PASSED |')
    status=status.replace('| Independent EM/F1 and subgroup accuracy | PENDING |', '| Independent EM/F1 and subgroup accuracy | MEASURED |')
    status=status.replace('| Candidate CPU latency/public five cases | PENDING |', '| Candidate CPU latency/public five cases | MEASURED |')
    status=status.rstrip()+'\n\n'+message
    Path('docs/STATUS.md').write_text(status,encoding='utf-8')
    readme=Path('README.md').read_text(encoding='utf-8')
    readme=readme.replace('12,000-example / two-epoch training is running','12,000-example / two-epoch training completed').replace('A 12,000-example, two-epoch background job is running.','The 12,000-example, two-epoch job completed; measured results are in logs/subset12k/summary.json.')
    Path('README.md').write_text(readme,encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
