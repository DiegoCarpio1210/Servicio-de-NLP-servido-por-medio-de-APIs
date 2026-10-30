import json
from experiments.run_pretrained_benchmark import atomic,partition,threshold

def test_context_disjoint_reproducible_partition():
    rows=[{'id':f'{i}-{j}','context':str(i),'answers':{'text':['yes'] if j else []}} for i in range(12) for j in (0,1)]
    a=partition(rows,42,4,8)
    assert a==partition(rows,42,4,8)
    assert not set(a['calibration']['context_hashes'])&set(a['final']['context_hashes'])
    assert len(a['calibration']['ids'])==4 and len(a['final']['ids'])==8

def test_atomic_progress_and_frozen_threshold(tmp_path):
    path=tmp_path/'progress.json';atomic(path,{'step':1});atomic(path,{'step':2})
    assert json.loads(path.read_text())=={'step':2}
    assert not path.with_suffix('.json.tmp').exists()
    rows=[{'text':'a','references':['a'],'margin':-2},{'text':'x','references':[],'margin':2}]
    assert threshold(rows)==-2 and threshold(rows,.95)==-2
