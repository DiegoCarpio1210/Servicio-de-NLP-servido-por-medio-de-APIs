"""Post-selection ISS diagnostics; no policy or threshold changes."""
import json
from pathlib import Path
from experiments.inference import ResearchEngine,candidate
from qa.spans import decide

def main():
    root=Path('logs/quality20261029');assert (root/'final_comparison.json').exists()
    profile=json.loads((root/'final_lock.json').read_text())['profiles']['trained']
    source=json.loads(Path('experiments/iss_diagnostic.json').read_text())
    engine=ResearchEngine(profile['model_dir'],'cpu');assert engine.sha==profile['weights_sha256']
    rows=[]
    for item in source['questions']:
        best=candidate(source['context'],engine.windows(source['context'],item['question']),profile['policy'],nbest=5)
        rows.append({**item,'response':decide(best,profile['threshold']),'diagnostic':best})
    result={'purpose':'Diagnostics after final evaluation; never used for optimization','profile':profile,'results':rows}
    (root/'iss_candidate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{'question':r['question'],'response':r['response']} for r in rows],indent=2))

if __name__=='__main__':main()
