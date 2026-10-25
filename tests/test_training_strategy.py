import pytest
from datasets import Dataset
from qa.data import subset,validation_partitions
from qa.training_control import training_lock
from qa.features import trim_training_padding


def dataset(n=100):
    return Dataset.from_dict({'id':[str(i) for i in range(n)],'answers':[{'text':[] if i%3==0 else ['answer'],'answer_start':[] if i%3==0 else [0]} for i in range(n)]})


def test_representative_reproducible_subset():
    a=subset(dataset(),30,42); b=subset(dataset(),30,42)
    assert a['id']==b['id'] and len(set(a['id']))==30
    assert sum(not x['text'] for x in a['answers'])==10


def test_three_disjoint_validation_partitions():
    c={'seed':42,'validation_partition':{'selection':20,'calibration':30,'holdout':50}}
    parts=validation_partitions(dataset(),c)
    ids={k:set(v['id']) for k,v in parts.items()}
    assert [len(ids[k]) for k in ('selection','calibration','holdout')]==[20,30,50]
    assert not ids['selection']&ids['holdout'] and not ids['calibration']&ids['holdout'] and not ids['selection']&ids['calibration']
    assert len(set.union(*ids.values()))==100
    assert parts['holdout']['id']==validation_partitions(dataset(),c)['holdout']['id']


def test_training_lock_excludes_second_job(tmp_path):
    with training_lock(tmp_path/'job.lock'):
        with pytest.raises(RuntimeError):
            with training_lock(tmp_path/'job.lock'): pass
    with training_lock(tmp_path/'job.lock'): pass


def test_trimming_keeps_all_real_tokens_and_labels():
    features={'input_ids':[[101,102,7,0,0]],'attention_mask':[[1,1,1,0,0]],'start_positions':[2],'end_positions':[2]}
    result=trim_training_padding(features,['input_ids','attention_mask'])
    assert result['input_ids']==[[101,102,7]]
    assert result['start_positions']==[2] and features['input_ids'][0][-1]==0


def test_stop_requests_checkpoint_and_evaluation(tmp_path):
    from scripts.train import EvidenceCallback
    from transformers import TrainerControl,TrainerState
    cb=EvidenceCallback(tmp_path,100); (tmp_path/'STOP').touch()
    result=cb.on_step_end(None,TrainerState(global_step=3),TrainerControl())
    assert result.should_save and result.should_evaluate and result.should_training_stop and cb.stop_requested
