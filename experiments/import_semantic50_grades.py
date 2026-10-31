"""Validate externally supplied semantic grades; never infer grades from EM/F1."""
import argparse,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'logs/semantic50'
def validate_grades(original,graded):
    assert len(original)==len(graded)==50,'Exactly 50 grading records required'
    assert len({r['id'] for r in graded})==50
    lookup={r['id']:r for r in original}
    for row in graded:
        source=lookup[row['id']]
        for field in ('context','question','answers','response'):
            assert row[field]==source[field],f'Grading must not alter {field}'
        assert type(row['semantic_score']) in (int,float) and row['semantic_score'] in (0,.5,1)
        assert all(isinstance(row.get(k),str) and row[k].strip() for k in ('semantic_justification','supporting_evidence','grader_id','grading_method'))
        if source.get('provisional_error_category')=='API failure':assert row['semantic_score']==0
    return {'count':50,'fully_correct':sum(r['semantic_score']==1 for r in graded),
      'partially_correct':sum(r['semantic_score']==.5 for r in graded),'incorrect':sum(r['semantic_score']==0 for r in graded),
      'weighted_points_out_of_50':sum(r['semantic_score'] for r in graded),
      'target_45_full_credit':sum(r['semantic_score']==1 for r in graded)>=45,
      'provenance':'Externally supplied grades; this importer does not authenticate a Claude API session'}

def main():
    p=argparse.ArgumentParser();p.add_argument('cohort',choices=['development','final']);p.add_argument('graded_json',type=Path);a=p.parse_args()
    source=json.loads((OUT/(a.cohort+'_grading_export.json')).read_text(encoding='utf-8'))
    graded=json.loads(a.graded_json.read_text(encoding='utf-8-sig'));summary=validate_grades(source,graded)
    (OUT/(a.cohort+'_external_semantic_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
