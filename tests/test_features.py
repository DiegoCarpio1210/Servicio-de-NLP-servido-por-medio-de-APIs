from pathlib import Path
import pytest
from transformers import AutoTokenizer
from qa.features import encode,training_features

@pytest.fixture(scope='module')
def tokenizer():
    path=Path('models/smoke')
    if not path.exists(): pytest.skip('Run real smoke training to create local tokenizer')
    return AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True)


def test_overflow_labels_and_final_answer(tokenizer):
    context='ordinary text '*250+'final answer'
    answer_start=context.index('final answer')
    examples={'context':[context],'question':['What is at the end?'],
              'answers':[{'text':['final answer'],'answer_start':[answer_start]}]}
    batch=encode(tokenizer,examples['question'],examples['context'])
    labels=training_features(examples,tokenizer)
    assert len(batch['input_ids'])>1
    assert labels['start_positions'][0]==0
    assert labels['start_positions'][-1]>0
    i=len(batch['input_ids'])-1
    start=batch['offset_mapping'][i][labels['start_positions'][i]][0]
    end=batch['offset_mapping'][i][labels['end_positions'][i]][1]
    assert context[start:end]=='final answer'
    assert max(o[1] for o,s in zip(batch['offset_mapping'][-1],batch.sequence_ids(i)) if s==1)==len(context)


def test_impossible_and_max_question(tokenizer):
    examples={'context':['é🌎 context '*300], 'question':['x '*250],
              'answers':[{'text':[],'answer_start':[]}]}
    f=training_features(examples,tokenizer)
    assert len(f['input_ids'])>1
    assert all(x==0 for x in f['start_positions']+f['end_positions'])


def test_repeated_unicode_offset(tokenizer):
    context='é🌎 repeat repeat'
    start=context.rindex('repeat')
    ex={'context':[context],'question':['Which word?'],'answers':[{'text':['repeat'],'answer_start':[start]}]}
    batch=encode(tokenizer,ex['question'],ex['context']); labels=training_features(ex,tokenizer)
    assert batch['offset_mapping'][0][labels['start_positions'][0]][0]==start
