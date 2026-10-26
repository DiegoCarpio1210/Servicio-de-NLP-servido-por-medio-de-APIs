"""Verify exported best weights and retain separate run/selected checkpoint step counts."""
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    root=Path('models/continued20k'); logs=Path('logs/continued20k')
    metadata=json.loads((root/'training_metadata.json').read_text())
    selected=Path(metadata['best_checkpoint'])
    state=json.loads((selected/'trainer_state.json').read_text())
    current=digest(root/'model.safetensors'); best=digest(selected/'model.safetensors')
    original=digest(Path('models/subset12k/model.safetensors'))
    original_checkpoint=digest(Path('checkpoints/subset12k/checkpoint-1522/model.safetensors'))
    assert current==best and original==original_checkpoint==metadata['warm_start_sha256']
    metadata.update(selected_optimizer_steps=state['global_step'],selected_epoch=state['epoch'])
    if not metadata.get('base_revision'):
        metadata['base_revision']=json.loads(Path('configs/continued20k.json').read_text())['base_revision']
    # The running process predates these metadata-only fields; add verified values without changing weights.
    for path in (root/'training_metadata.json',logs/'training_metadata.json'):
        path.write_text(json.dumps(metadata,indent=2))
    result={'export_matches_best_checkpoint':True,'original_matches_preserved_checkpoint':True,
            'weights_sha256':current,'original_sha256':original,'run_steps':metadata['optimizer_steps'],
            'selected_steps':state['global_step'],'selected_epoch':state['epoch'],
            'optimizer_scheduler_rng_present':all((selected/name).exists() for name in ('optimizer.pt','scheduler.pt','rng_state.pth'))}
    assert result['optimizer_scheduler_rng_present']
    (logs/'artifact_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
