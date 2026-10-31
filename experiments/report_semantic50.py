"""Summarize completed HTTP cohorts without fabricating semantic judgments."""
import json
from collections import Counter
from experiments.run_semantic50 import ROOT,OUT,read
from qa.metrics import normalize

def main():
    protocol=read(ROOT/'configs/semantic50_protocol.json');diagnostics=read(OUT/'development_diagnostics.json')
    cohorts={name:read(OUT/(name+'_results.json')) for name in ('development','final')}
    compact={}
    for name,result in cohorts.items():
        assert result['server_stopped']
        exports=read(OUT/(name+'_grading_export.json'))
        compact[name]={k:v for k,v in result.items() if k!='modes'}
        compact[name]['modes']={k:{a:b for a,b in v.items() if a!='rows'} for k,v in result['modes'].items()}
        compact[name]['provisional_nonexact_categories']=dict(Counter(r['provisional_error_category'] for r in exports if r['EM']<100))
        compact[name]['nonexact_ids']=[r['id'] for r in exports if r['EM']<100]
        relations=Counter()
        for r in exports:
            if r['EM']==100 or not r['answers']['text'] or not r.get('response') or r['response']['answer'] is None:continue
            actual=normalize(r['response']['answer']);expected=normalize(r['answers']['text'][0])
            relations['shorter reference span; essential omission undecided' if (' '+actual+' ') in (' '+expected+' ') else
              ('extra surrounding words; semantic validity undecided' if (' '+expected+' ') in (' '+actual+' ') else 'different span; semantic review required')]+=1
        compact[name]['nonexact_boundary_relations']=dict(relations)
        compact[name]['lexical_45_target']=result['modes']['sequential']['exact_correct_out_of_50']>=45
        compact[name]['composition']=protocol['cohorts'][name]
    total=sum(m['sent'] for c in compact.values() for m in c['modes'].values())
    summary={'unique_questions':100,'evaluation_post_requests':total,'all_recorded_HTTP_requests':sum(c['all_wire_requests'] for c in compact.values()),
      'cohorts':compact,'semantic_correct_out_of_50':None,'claude_used':False,'semantic_target_verified':False,
      'improvements_selected':None,'production_changed':False,'dataset_publication':False,'integrity':read(OUT/'integrity.json')}
    (ROOT/'logs/semantic50_readiness_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Readiness for a 45/50 semantic QA evaluation','',
      '**Local HTTP readiness was measured; the Claude-semantic 45/50 target is not verified.** No Claude credential was configured, no external judge was called, and no semantic scores were invented. Exact Match/F1 below are separate lexical metrics. Production remains the original DeBERTa with threshold 2.7772940397262573.',
      '', '## Verified professor requirements','',
      'Authority: originals/qa_api_contract.yaml and originals/guide_extracted.txt (extracted from the preserved instructor PDF). The brief specifies approximately 25% unanswerable items, 50 hidden non-SQuAD questions sent sequentially, 5 seconds per request and 150 seconds total. Its rubric awards full credit for exact/correct-null answers and permits 1 / 0.5 / 0 for judged partial overlap. The user reports Claude will be the judge; its private prompt/version and exact adjudication policy are unknown.',
      '', '| Requirement | Verified behavior |','|---|---|',
      '| /health | Public; 503 loading, 200 with status ok only after model loading |',
      '| /info | Exact case-sensitive Bearer token; team/model/positive parameter count, plus optional metadata |',
      '| /predict input | UTF-8 JSON; id 1–64 characters, context 1–5000, question 1–500; no additional input fields |',
      '| /predict output | id, answer, start, score; answer is an original-context span or null; start is a zero-based character offset and null iff answer is null |',
      '| Response extras | YAML explicitly allows additional response fields; the harness does not invent a prohibition |',
      '| Errors | 401 unauthorized; 400 or 422 bad_request; 500 internal_error counts as failure |',
      '| Time / concurrency | Official evaluation is sequential. Two-client concurrency is an additional stress test, not an invented instructor requirement |',
      '| Network | Instructor requires a public HTTPS endpoint; this task intentionally tested loopback only and opened no tunnel |',
      '', 'All successful outputs were JSON-schema checked, id checked, finite-score checked and paired-null/Unicode-substring checked. Authorization failures and malformed/oversized requests were exercised separately from evaluated questions. The full regression suite covers additional boundaries. Existing unchanged-model 5,000-character HTTP checks remain in logs/stable_deberta_verification.json; current authored cohorts include overflow contexts up to 4,063 characters.',
      '', '## Cohorts, provenance and leakage safeguards','',
      'Two assistant-authored fictional-scenario cohorts were written before predictions. Development has 35 answerable / 15 impossible questions; held-out final has 38 / 12, bracketing the brief’s approximate 75/25 distribution rather than assuming 50/50. Each has 50 questions over 10 distinct contexts. Categories cover names, locations, dates, numbers, multi-part/longer answers, multi-sentence evidence, distractors, Unicode and long overflow contexts. Long contexts deliberately include repeated maintenance prose as a window-handling stressor.',
      '', 'Cohort hashes, seeded order (20261031 and 20261032), composition and overlap methodology were frozen in commit 17b61a0 before HTTP evaluation. Exact normalized context/question overlap and near duplicates were audited against the recorded 500/1,024 evaluations, earlier partitions, all LoRA train/development/calibration records and public samples. Near-duplicate detection uses word-shingle Jaccard >=0.8 (3-word questions, 5-word contexts). No matches were found. Development and final share no normalized context or question.',
      '', 'This audit is limited to available records and a specified similarity detector; it cannot prove universal pretraining independence. The synthetic questions may be easier or less representative than the instructor’s private texts. Both cohorts have a common author and recurring templates; five questions share each context. They are not independent external benchmark samples. Their content and answerability mix both differ, so score differences cannot be attributed solely to the distribution. No dataset or raw predictions are published to Git.',
      '', '## Real HTTP methodology','',
      'All evaluated predictions traveled through a real localhost TCP connection to the actual FastAPI application and pinned CPU Engine. No evaluation used a direct model call. An experiment-only wrapper records the candidate margins produced inside those requests, without changing inference or the response contract. Each cohort sends exactly 50 sequential evaluation POSTs, then replays those same 50 with two clients to check reliability. The concurrent replay is not another independent accuracy cohort. Each mode has its own 150-second cap and measured 5-second request limit.',
      '',f"Measured totals: **{total} evaluated POST /predict requests**, covering 100 unique questions. Including readiness polling, metadata, authentication/error checks and the UI gateway check, the server recorded **{summary['all_recorded_HTTP_requests']} total HTTP requests**. Expected 401/422/503 responses in these diagnostic calls are not counted as failed evaluation questions. Route/status counts are in logs/semantic50_readiness_summary.json.",
      '', '## Results: reliability versus lexical accuracy','',
      '| Cohort / mode | Successful HTTP | Within time caps | Total s | Mean s | Maximum s | Exact correct /50 | EM | F1 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,result in compact.items():
        for mode,m in result['modes'].items():lines.append(f"| {name} / {mode} | {m['http_valid']}/50 | {m['within_limits']}/50 | {m['seconds']:.2f} | {m['mean_seconds']:.3f} | {m['max_seconds']:.3f} | {m['exact_correct_out_of_50']} | {m['EM']:.2f}% | {m['F1']:.2f}% |")
    lines += ['', '| Cohort (sequential accuracy only) | Answerable EM | Impossible EM | Correct nulls | Incorrect nulls | Semantic correct /50 |','|---|---:|---:|---:|---:|---|']
    for name,result in compact.items():
        m=result['modes']['sequential'];lines.append(f"| {name} | {m['answerable_EM']:.2f}% | {m['unanswerable_EM']:.2f}% | {m['correct_nulls']} | {m['incorrect_nulls']} | Not graded |")
    lines += ['', 'EM uses existing official SQuAD normalization; F1 uses normalized token overlap against the authored extractive reference. A non-exact span can be semantically valid or incomplete. Neither F1 nor a high model score is treated as semantic correctness. The historical 83.00/86.69 SQuAD results refer to a different cohort and are not a paired comparison with these scenarios.',
      '', '## Error analysis and defensible changes','',
      'Development had four non-exact spans: two shortened noun phrases and two spans with an added preposition. They are exported for semantic review rather than automatically marked fully wrong or fully correct. All reference spans were covered by at least one token window. Observed margins on these four were -23.796, -17.464, -19.065 and -12.807, all already below the production threshold; increasing the threshold cannot repair their boundaries.',
      '', '| Proposed change | Evidence and decision |','|---|---|',
      '| Raise no-answer threshold | No development incorrect-null errors; all four mismatches are already non-null. Not justified; not tested or activated |',
      '| Expand/replace short answers | Could help some phrases but risks adding unsupported or nonessential text. No semantic judge or demonstrated general bug; no substitutions or question-specific rules |',
      '| Trim leading whitespace / adjust offsets | Existing answers satisfy exact original-context slicing; whitespace is normalized by EM. Cosmetic benefit does not justify changing the frozen policy |',
      '| Fix context windows or Unicode | Reference coverage and offset checks passed. No demonstrated truncation/offset bug found; no change |',
      '| LoRA / training | Previous experiments did not establish benefit; not repeated |',
      '', 'No inference improvement was selected. Commit 3e1d49e locked the unchanged original policy, active manifest and source hashes before the final cohort. The final accuracy result was examined once, followed only by the required technical replay; no final-set tuning or answer substitutions occurred.',
      '', '## Final non-exact/error categories','', '| Provisional category | Count |','|---|---:|']
    for category,count in compact['final']['provisional_nonexact_categories'].items():lines.append(f'| {category} | {count} |')
    if not compact['final']['provisional_nonexact_categories']:lines.append('| None under lexical scoring | 0 |')
    lines += ['', '| Non-exact boundary relation (not a semantic grade) | Count |','|---|---:|']
    for category,count in compact['final']['nonexact_boundary_relations'].items():lines.append(f'| {category} | {count} |')
    lines += ['', 'These are reference-based diagnostic categories, not fabricated semantic judgments. Wrong nulls and non-null outputs for declared impossible items are reported explicitly; remaining non-exact spans require a grader to distinguish incomplete answers, harmless wording/boundary differences, ambiguity or incorrect spans. No API/offset failure should be inferred merely from EM <100.',
      '', '## Reproducible independent semantic grading','',
      'No ANTHROPIC_API_KEY was configured. No credential was requested, obtained, printed or sent, and no data was transmitted to an external model. Local JSON and CSV exports include all contexts, questions, original references, actual responses, EM/F1 and provisional categories. Semantic score/justification/evidence/grader fields are blank. No proxy judge was silently substituted for Claude.',
      '', 'Use logs/semantic50/final_grading_export.json (or the CSV for review). An independent grader should verify: the answer addresses the actual question, is supported by the context, does not contradict it, omits no essential information and adds no unsupported facts. Impossible questions require null. Judge all 50, not only EM mismatches. Record semantic_score 1 / 0.5 / 0, a brief semantic_justification, supporting_evidence, grader_id and grading_method. Keep context/question/reference/response unchanged.',
      '', '`python -m experiments.import_semantic50_grades final <graded-copy.json>` validates completeness and input integrity and counts full-credit answers separately from partial-credit points. It will not turn missing grades into success or authenticate a claimed Claude session. A proxy review must identify itself as a proxy and cannot be described as the professor’s private Claude evaluation. Transmitting data to an external grading provider remains a separate user-controlled action.',
      '', '## Reproduction, integrity and submission recommendation','',
      'With the locally preserved, hash-matching cohorts, `python -m experiments.run_semantic50 development` / `final` runs the HTTP harness. It refuses to overwrite completed results; the final run requires the frozen local lock. `python -m experiments.report_semantic50` regenerates this report from saved evidence without rerunning predictions. Datasets are intentionally local; a fresh clone alone cannot reconstruct the privately preserved authored cohorts from their hashes.',
      '', 'The actual UI gateway, assets and unchanged 83.00% dashboard metric were checked against the running API. API servers were stopped after each cohort and ephemeral tokens discarded. No public tunnel was opened. Protected model/tokenizer/calibration hashes, stable production source and main remain unchanged. The complete regression suite passed **77 tests** (logs/semantic50_regression.txt), including Unicode offsets, contract-allowed response extras, finite scores and refusal to fabricate missing semantic grades.',
      '', 'The development lexical count reached 46/50; the held-out lexical count was 40/50 and did not reach 45/50. The ten non-exact spans are not automatically ten semantic failures, but F1 95.29% cannot be substituted for 45 semantically correct answers.',
      '', 'Additional uncovered limit: shared preprocessing caps questions at 64 tokens even though the API accepts up to 500 characters. These cohorts use short questions, so semantic behavior when essential information occurs beyond that cap is not established. This existing limitation was not changed based on final-cohort results. Future development-only edge cases should test it before proposing a different token budget.',
      '', 'Technical local readiness and lexical correctness are measured separately above. The 45/50 Claude-semantic objective remains **unverified pending legitimate grading**; even a local 45/50 lexical result is not a guarantee on private instructor questions. HTTPS/external-network testing was intentionally not repeated in this task. Preserve production and obtain an independent rubric-based grading pass before declaring final semantic readiness or submission complete.']
    (ROOT/'docs/CLAUDE_50_QUESTION_READINESS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('unique_questions','evaluation_post_requests','all_recorded_HTTP_requests','semantic_target_verified')},indent=2))

if __name__=='__main__':main()
