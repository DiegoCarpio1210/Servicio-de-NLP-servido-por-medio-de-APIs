import pytest
from qa.metrics import weighted_threshold_sweep,prior_weighted_metrics,summarize


def test_weighted_sweep_matches_direct_metrics_and_ties():
    rows=[{'margin':1,'text':'cat','references':['cat']},{'margin':1,'text':'cat dog','references':['cat']},
          {'margin':2,'text':'wrong','references':[]},{'margin':3,'text':'x','references':[]}]
    sweep=weighted_threshold_sweep(rows,.75)
    assert len(sweep)==4
    for item in sweep:
        expected=prior_weighted_metrics(summarize(rows,item['threshold']),.75)
        assert item['EM']==pytest.approx(expected['EM'])
        assert item['F1']==pytest.approx(expected['F1'])


def test_weighted_objective_uses_target_not_sample_count():
    a={'margin':0,'text':'yes','references':['yes']}; n={'margin':1,'text':'x','references':[]}
    one=weighted_threshold_sweep([a,n],.75)
    many=weighted_threshold_sweep([a,a,a,n],.75)
    assert [x['EM'] for x in one]==pytest.approx([x['EM'] for x in many])


def test_prior_validation():
    with pytest.raises(ValueError): weighted_threshold_sweep([],0.75)
    with pytest.raises(ValueError): weighted_threshold_sweep([],1)


def test_official_empty_reference_filtering():
    from qa.metrics import score
    assert score('', ['a', 'cat'])==(0,0)
    assert score('cat', ['a','cat'])==(1,1)
    assert score('', ['a','the'])==(1,1)
