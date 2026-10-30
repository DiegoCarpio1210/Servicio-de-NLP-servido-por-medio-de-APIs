"""Describe calibration/development errors; never inspect the frozen final set."""
import json
from pathlib import Path
from qa.metrics import score

def main():
    root=Path('logs/quality20261029');selection=json.loads((root/'stable_selection.json').read_text())
    result={}
    for split in ('calibration','development'):
        rows=json.loads((root/f'stable_{split}_predictions.json').read_text())['standard']
        threshold=selection['profiles']['production']['threshold']
        yes=[r for r in rows if r['references']];no=[r for r in rows if not r['references']]
        rejected=[r for r in yes if r['margin']>threshold]
        result[split]={'answerable_count':len(yes),'impossible_count':len(no),
          'answerable_false_abstentions':len(rejected),
          'false_abstentions_with_exact_span_available':sum(score(r['text'],r['references'])[0] for r in rejected),
          'answerable_wrong_span_when_answered':sum(score(r['text'],r['references'])[0]==0 for r in yes if r['margin']<=threshold),
          'answerable_raw_best_span_EM_without_abstention':100*sum(score(r['text'],r['references'])[0] for r in yes)/len(yes),
          'impossible_false_answers':sum(r['margin']<=threshold for r in no)}
    (root/'error_breakdown.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
