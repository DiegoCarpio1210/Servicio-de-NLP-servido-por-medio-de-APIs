"""Analyze observed development only; do not select policies from the final cohort."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from qa.features import encode
from qa.metrics import score
from experiments.run_semantic50 import ROOT,OUT,read,key,sha

def main():
    items=read(OUT/'development.json');results=read(OUT/'development_results.json')['modes']['sequential']['rows']
    traces=read(OUT/'development_traces.json');tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/pretrained/deberta-v3-base-squad2',local_files_only=True)
    records=[]
    for item,row in zip(items,results):
        c=traces[key(item['context'],item['question'])];enc=encode(tokenizer,[item['question']],[item['context']],384,128)
        coverage=[]
        for answer,start in zip(item['answers']['text'],item['answers']['answer_start']):
            coverage.append(any(any(s==1 and a<=start for (a,b),s in zip(enc['offset_mapping'][i],enc.sequence_ids(i))) and
              max(b for (a,b),s in zip(enc['offset_mapping'][i],enc.sequence_ids(i)) if s==1)>=start+len(answer) for i in range(len(enc['input_ids']))))
        assert all(coverage)
        if row['EM']<100:
            answer=(row.get('response') or {}).get('answer')
            reason='API failure' if not row['within_limits'] else ('incorrect abstention' if item['answers']['text'] and answer is None else
              ('unsupported non-null answer' if not item['answers']['text'] else 'grading uncertainty: inexact span requires semantic review'))
            records.append({'id':item['id'],'category':reason,'actual_answer':answer,'references':item['answers']['text'],
              'margin':c['margin'],'candidate':c['text'],'candidate_EM_F1':list(score(c['text'],item['answers']['text'])),
              'windows':len(enc['input_ids']),'gold_window_coverage':coverage,'F1':row['F1']})
    report={'development_only':True,'non_exact_records':records,'all_answer_spans_covered_by_windows':True,
      'threshold_changed':False,'selected_change':None,'reason':'No demonstrated extraction/offset/window bug; semantic grading unavailable. Do not tune a threshold to these final-like author-created examples.'}
    (OUT/'development_diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
