import copy
from pathlib import Path
import pytest,yaml
from experiments.run_semantic50 import check_response
from experiments.import_semantic50_grades import validate_grades

SCHEMA=yaml.safe_load(Path('originals/qa_api_contract.yaml').read_text(encoding='utf-8'))['components']['schemas']['PredictResponse']

def test_unicode_offsets_and_contract_allowed_extra_response_fields():
    item={'id':'x','context':'🌍 text'}
    body={'id':'x','answer':'text','start':2,'score':.8,'optional_extra':'allowed by YAML'}
    check_response(body,item,SCHEMA)
    with pytest.raises(AssertionError):check_response({**body,'start':3},item,SCHEMA)

def test_null_pair_and_finite_scores():
    item={'id':'x','context':'text'};body={'id':'x','answer':None,'start':None,'score':.8}
    check_response(body,item,SCHEMA)
    with pytest.raises(AssertionError):check_response({**body,'start':0},item,SCHEMA)
    with pytest.raises(AssertionError):check_response({**body,'score':float('nan')},item,SCHEMA)

def records():
    source=[{'id':str(i),'context':'text','question':'q','answers':{'text':['text']},'response':{'answer':'text'},'semantic_score':None} for i in range(50)]
    graded=[{**r,'semantic_score':1,'semantic_justification':'Test fixture','supporting_evidence':'text','grader_id':'unit-test','grading_method':'unit-test fixture'} for r in source]
    return source,graded

def test_pending_semantic_scores_cannot_be_reported_as_completed():
    source,graded=records()
    assert validate_grades(source,graded)['fully_correct']==50
    graded[0]['semantic_score']=None
    with pytest.raises(AssertionError):validate_grades(source,graded)

def test_grader_cannot_change_the_question_or_reward_api_failure():
    source,graded=records();graded[0]['question']='changed'
    with pytest.raises(AssertionError):validate_grades(source,graded)
    source,graded=records();source[0]['provisional_error_category']='API failure'
    with pytest.raises(AssertionError):validate_grades(source,graded)

def test_partial_credit_does_not_count_as_a_fully_correct_answer():
    source,graded=records()
    for r in graded[:6]:r['semantic_score']=.5
    summary=validate_grades(source,graded)
    assert summary['weighted_points_out_of_50']==47
    assert summary['fully_correct']==44 and not summary['target_45_full_credit']
