import numpy as np
import pytest
from qa.spans import aggregate,decide
from qa.metrics import score


def test_last_window_unicode_repeated():
    context='é🌎 one one\nlast'
    windows=[([3,1,1],[3,1,1],[None,(3,6),(7,10)],0),
             ([0,5],[0,5],[None,(11,15)],0)]
    c=aggregate(context,windows); assert c['text']=='last' and c['start']==11
    assert decide(c,0)['answer']=='last'


def test_null_and_non_context():
    c=aggregate('text',[([9,1],[9,1],[None,(0,4)],0)])
    assert decide(c,0)['answer'] is None and decide(c,0)['start'] is None
    assert 0<=decide(c,0)['score']<=1


def test_invalid_spans():
    c=aggregate('abc',[([0,2,8],[0,8,2],[None,(0,1),(2,3)],0)],max_answer_length=1)
    assert c['text'] in ('a','c')
    with pytest.raises(ValueError): aggregate('x',[([np.nan],[0],[None],0)])


def test_metrics():
    assert score('The cat.', ['cat'])==(1,1)
    assert score('',[])==(1,1)
    assert score('a cat dog',['cat'])==(0,2/3)


def test_fast_threshold_sweep_matches_brute_force():
    from qa.metrics import threshold_sweep,summarize
    rows=[{'text':'cat','references':['cat'],'margin':1},
          {'text':'cat dog','references':['cat'],'margin':1},
          {'text':'x','references':[],'margin':2}]
    for row in threshold_sweep(rows):
        brute=summarize(rows,row['threshold'])['all']
        assert row['EM']==pytest.approx(brute['EM'])
        assert row['F1']==pytest.approx(brute['F1'])
