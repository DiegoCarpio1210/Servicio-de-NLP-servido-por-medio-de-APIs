from datasets import load_dataset
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
    raw = load_dataset('rajpurkar/squad_v2', cache_dir='data/hf')
    return {k: subset(raw[k], config[k+'_size'], config['seed']) for k in ('train','validation')}


def stats(dataset):
    n = sum(not a['text'] for a in dataset['answers'])
    return {'count': len(dataset), 'unanswerable': n, 'unanswerable_ratio': n/len(dataset),
            'fingerprint': dataset._fingerprint}
