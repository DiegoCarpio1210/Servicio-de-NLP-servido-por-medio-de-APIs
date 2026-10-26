from pathlib import Path
from datasets import load_dataset, load_from_disk
import numpy as np


def subset(dataset, size, seed):
    if not size or size >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    groups = [[i for i, a in enumerate(dataset['answers']) if bool(a['text']) == flag]
              for flag in (False, True)]
    n0 = round(size * len(groups[0]) / len(dataset))
    indices = list(rng.choice(groups[0], n0, replace=False)) + list(rng.choice(groups[1], size-n0, replace=False))
    rng.shuffle(indices)
    return dataset.select([int(i) for i in indices])


def load_splits(config):
    snapshot = Path('data/squad_v2')
    raw = load_from_disk(str(snapshot)) if snapshot.exists() else load_dataset('rajpurkar/squad_v2', cache_dir='data/hf')
    if config.get('data_manifest'):
        import json
        manifest=json.loads(Path(config['data_manifest']).read_text())
        result={}
        for split,key in (('train','train_ids'),('validation','selection_ids')):
            lookup={item:i for i,item in enumerate(raw[split]['id'])}
            ids=manifest[key]
            if len(ids)!=len(set(ids)): raise ValueError('Duplicate manifest IDs')
            result[split]=raw[split].select([lookup[item] for item in ids])
        return result
    return {k: subset(raw[k], config[k+'_size'], config['seed']) for k in ('train','validation')}


def stats(dataset):
    n = sum(not a['text'] for a in dataset['answers'])
    return {'count': len(dataset), 'unanswerable': n, 'unanswerable_ratio': n/len(dataset),
            'fingerprint': dataset._fingerprint}


def validation_partitions(dataset, config):
    """Disjoint stratified examples for checkpoint selection, threshold fit and final scoring."""
    sizes=config['validation_partition']
    if sum(sizes.values()) != len(dataset):
        raise ValueError('Validation pool size must equal the three partition sizes')
    selection=subset(dataset,sizes['selection'],config['seed']+101)
    selected=set(selection['id'])
    remaining=dataset.select([i for i,item in enumerate(dataset['id']) if item not in selected])
    calibration=subset(remaining,sizes['calibration'],config['seed']+102)
    calibrated=set(calibration['id'])
    holdout=remaining.select([i for i,item in enumerate(remaining['id']) if item not in calibrated])
    assert len(holdout)==sizes['holdout']
    return {'selection':selection,'calibration':calibration,'holdout':holdout}
