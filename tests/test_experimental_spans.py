import numpy as np
from experiments.inference import candidate
from qa.spans import aggregate


def test_standard_vectorized_decoder_matches_production():
    rng=np.random.default_rng(42)
    for _ in range(20):
        windows=[(rng.normal(size=10).astype('float32'),rng.normal(size=10).astype('float32'),[None]+[(i,i+1) for i in range(8)]+[None],0)]
        assert candidate('abcdefgh',windows,{'max_length':30,'length_penalty':0})==aggregate('abcdefgh',windows)


def test_length_penalty_is_general_not_question_specific():
    offsets=[None,(0,1),(2,3),(4,5)]
    windows=[(np.array([-5.,3.,0.,0.]),np.array([-5.,2.,0.,2.1]),offsets,0)]
    assert candidate('a b c',windows,{'max_length':30,'length_penalty':0})['text']=='a b c'
    assert candidate('a b c',windows,{'max_length':30,'length_penalty':.25})['text']=='a'
