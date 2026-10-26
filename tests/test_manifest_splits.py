"""Guard exact continuation partitions against silently re-sampling validation."""
import json
from datasets import Dataset, DatasetDict
import pytest
from qa import data


def test_manifest_uses_exact_ids_and_selection_only(tmp_path,monkeypatch):
    raw=DatasetDict({'train':Dataset.from_dict({'id':['t1','t2','t3']}),
                     'validation':Dataset.from_dict({'id':['s','c','h']})})
    monkeypatch.chdir(tmp_path)
    (tmp_path/'data/squad_v2').mkdir(parents=True)
    monkeypatch.setattr(data,'load_from_disk',lambda _:raw)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'train_ids':['t3','t1'],'selection_ids':['s'],
                                   'calibration_ids':['c'],'holdout_ids':['h']}))
    splits=data.load_splits({'data_manifest':str(manifest)})
    assert splits['train']['id']==['t3','t1']
    assert splits['validation']['id']==['s']
    manifest.write_text(json.dumps({'train_ids':['t1','t1'],'selection_ids':['s']}))
    with pytest.raises(ValueError,match='Duplicate'): data.load_splits({'data_manifest':str(manifest)})
