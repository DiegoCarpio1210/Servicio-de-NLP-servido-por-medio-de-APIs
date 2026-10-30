"""Check frozen IDs/contexts without computing final-set predictions."""
import hashlib
import json
from pathlib import Path
from datasets import load_from_disk

def main():
    root=Path('logs/quality20261029')
    manifest=json.loads((root/'manifest.json').read_text())
    raw=load_from_disk('data/squad_v2')
    hashes={split:dict(zip(ds['id'],[hashlib.sha256(c.encode()).hexdigest() for c in ds['context']])) for split,ds in raw.items()}
    parts={key:set(manifest[key+'_ids']) for key in ('train','selection','calibration','development','final')}
    for key,ids in parts.items():assert len(ids)==len(manifest[key+'_ids'])
    for key,ids in parts.items():
        for other,others in parts.items():
            if key!=other:assert not ids&others,(key,other)
    contexts={k:{hashes['validation' if k in ('selection','calibration') else 'train'][id] for id in ids} for k,ids in parts.items()}
    for key in ('development','final'):
        for other in parts:
            if key!=other:assert not contexts[key]&contexts[other],(key,other)
    result={'counts':{k:len(v) for k,v in parts.items()},'all_ids_disjoint':True,
      'final_and_development_contexts_disjoint_from_other_partitions':True,
      'selection_calibration_shared_contexts':len(contexts['selection']&contexts['calibration']),
      'final_predictions_computed':False,
      'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
    (root/'partition_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
