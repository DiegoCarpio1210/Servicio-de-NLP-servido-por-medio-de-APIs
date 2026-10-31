from experiments.report_final500 import confidence
from experiments.final500 import MODELS

def test_paired_bootstrap_direction_and_constant_scores():
    rows={}
    for key in MODELS:
        value=0 if key=='quality40k' else 100
        rows[key]=[{'id':str(i),'references':['x'] if i<250 else [],'EM':value,'F1':value,
          'response':{'answer':'x' if i<250 else None}} for i in range(500)]
    ci,paired=confidence(rows)
    assert ci['quality40k']['EM']==[0.,0.]
    assert ci[MODELS[0]]['F1']==[100.,100.]
    assert paired[MODELS[0]]==[100.,100.]
