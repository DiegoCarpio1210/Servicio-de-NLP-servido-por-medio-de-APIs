"""Read-only integrity review of completed training and retained checkpoints."""
import hashlib,json
from pathlib import Path
from safetensors import safe_open

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    root=Path('logs/quality20261029');model=Path('models/quality40k')
    meta=json.loads((model/'training_metadata.json').read_text())
    history=json.loads(Path('logs/quality40k/history.json').read_text())
    evaluations=[{'step':r['step'],'loss':r['eval_loss']} for r in history if 'eval_loss' in r]
    best=min(evaluations,key=lambda r:r['loss']);assert best['step']==2000
    assert meta['completed'] and meta['optimizer_steps']==2531 and meta['selected_optimizer_steps']==2000
    checkpoints={}
    for folder in Path('checkpoints/quality40k').iterdir():
        files={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in folder.iterdir() if p.is_file()}
        for required in ('model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','tokenizer.json','config.json'):
            assert files[required]['bytes']>0
        with safe_open(folder/'model.safetensors',framework='pt') as tensors:
            assert len(list(tensors.keys()))>0
        checkpoints[folder.name]=files
    exported={p.name:digest(p) for p in model.iterdir() if p.is_file()}
    for name in ('model.safetensors','tokenizer.json','vocab.txt','config.json'):
        assert exported[name]==checkpoints['checkpoint-2000'][name]['sha256']
    stable={'weights':digest(Path('models/continued20k/model.safetensors')),'calibration':digest(Path('models/continued20k/calibration.json'))}
    assert stable['weights']=='7d66ffcff40157ed8f9f728f51ffe3c1b5b3f249ac8c100ece9666d1f7007848'
    assert stable['calibration']=='52289cfdd93d7f37ef035715720e4afea67cdd1fac435b7d7f89476a7664a6dd'
    result={'metadata':meta,'checkpoint_selection':'Minimum loss on the original 512 selection examples; not final accuracy selection',
      'selection_losses':evaluations,'best':best,'exported_hashes':exported,'checkpoints':checkpoints,'stable_unchanged':stable,
      'optimizer_files_present_and_hashed':True,'resume_execution_tested':False}
    (root/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'best':best,'exported_weights':exported['model.safetensors'],'stable_unchanged':True}))

if __name__=='__main__':main()
