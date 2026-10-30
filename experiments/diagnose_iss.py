"""Inspect stable logits without fitting anything to the user's manual examples."""
import json
from pathlib import Path
from experiments.inference import ResearchEngine,candidate
from qa.spans import decide


def main():
    raw=json.loads(Path('experiments/iss_diagnostic.json').read_text())
    engine=ResearchEngine('models/continued20k','cpu')
    threshold=json.loads(Path('models/continued20k/calibration.json').read_text())['threshold']
    out=[]
    for item in raw['questions']:
        windows=engine.windows(raw['context'],item['question'])
        best=candidate(raw['context'],windows,{'max_length':30,'length_penalty':0},nbest=10)
        response=decide(best,threshold)
        out.append({**item,'response':response,'diagnostic':best,'offset_valid':response['answer'] is None or raw['context'][response['start']:response['start']+len(response['answer'])]==response['answer']})
    result={'purpose':raw['purpose'],'model_sha256':engine.sha,'threshold':threshold,'context_characters':len(raw['context']),'results':out,'parameters_changed':False}
    Path('logs/quality20261029/iss_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{'question':r['question'],'response':r['response'],'windows':r['diagnostic']['windows'],'null_score':r['diagnostic']['null_score'],'span_score':r['diagnostic']['best_span_score']} for r in out],indent=2))

if __name__=='__main__':main()
