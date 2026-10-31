import random
import pytest
from qa.spans import decide
from qa.metrics import score
from experiments.analyze_lora_calibration import select_threshold,measure,breakdown,confidence

def example(identifier,margin,text='answer',refs=None):
    return {'id':identifier,'margin':margin,'text':text,'start':0,'references':['answer'] if refs is None else refs}

def test_strict_threshold_equality_and_structural_null():
    row=example('x',5.)
    assert decide(row,5.)['answer']=='answer'
    assert decide(row,4.999)['answer'] is None
    assert decide({**row,'start':None},100.)['answer'] is None

def test_calibration_floor_and_deterministic_tie_break():
    rows=[example('a',1.),example('b',20.),example('n',10.,refs=[])]
    selected,_,_=select_threshold(rows)
    assert selected['threshold']==1. and selected['groups']['unanswerable']['EM']==100.
    assert measure(rows,20.)['weighted_75_25']['EM']>selected['weighted_75_25']['EM']
    shuffled=rows.copy();random.Random(5).shuffle(shuffled)
    assert select_threshold(shuffled)[0]==selected

def test_f1_breaks_equal_em_tie_using_calibration_only():
    rows=[example('a',1.),example('b',2.,text='answer extra'),example('n',10.,refs=[])]
    assert select_threshold(rows)[0]['threshold']==2.

def test_breakdown_partitions_and_decision_metric_agreement():
    rows=[example('null',9.),example('exact',-1.),example('partial',0.,'answer extra'),example('wrong',0.,'different'),example('negative',10.,refs=[])]
    b=breakdown(rows,2.)
    assert [b[k] for k in ('answerable_null','answerable_exact','answerable_partial_nonnull','answerable_zero_overlap_nonnull')]==[1,1,1,1]
    expected=100*sum(score(decide(r,2.)['answer'] or '',r['references'])[0] for r in rows)/len(rows)
    assert measure(rows,2.)['groups']['all']['EM']==expected

def test_identical_system_paired_intervals_are_zero():
    rows=[example('a',-1.),example('n',10.,refs=[])]
    result=confidence([(rows,2.),(rows,2.),(rows,2.)])
    assert all(v==[0.,0.] for v in result['step200_minus_original_both_calibrated_95CI'].values())
