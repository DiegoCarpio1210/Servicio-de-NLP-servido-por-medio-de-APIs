"""Official SQuAD normalization and maximum-over-references EM/F1."""
from collections import Counter
import re
import string


def normalize(s):
    s = ''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r"\b(a|an|the)\b", ' ', s).split())


def score(prediction, references):
    p = normalize(prediction).split()
    # Official SQuAD v2 drops empty normalized references unless all references are empty.
    references = [ref for ref in references if normalize(ref)] or ['']
    results = []
    for ref in references or [""]:
        r = normalize(ref).split()
        overlap = sum((Counter(p) & Counter(r)).values())
        f1 = float(p == r) if not p or not r else 2 * overlap / (len(p) + len(r))
        results.append((float(p == r), f1))
    return tuple(max(x[i] for x in results) for i in range(2))


def summarize(rows, threshold):
    output = {}
    for group in ('all', 'answerable', 'unanswerable'):
        subset = [r for r in rows if group == 'all' or bool(r['references']) == (group == 'answerable')]
        values = [score('' if r['margin'] > threshold else r['text'], r['references']) for r in subset]
        output[group] = {'count': len(subset), 'EM': 100*sum(v[0] for v in values)/len(values) if values else None,
                         'F1': 100*sum(v[1] for v in values)/len(values) if values else None}
    return output


def threshold_sweep(rows):
    """Exact O(n log n) sweep: switch tied margins from null to span together."""
    ordered=sorted(rows,key=lambda r:r['margin'])
    if not ordered: raise ValueError('Empty calibration set')
    totals=[0.,0.]
    for r in ordered:
        values=score('',r['references'])
        for j in range(2): totals[j]+=values[j]
    def snapshot(threshold):
        return {'threshold':threshold,'count':len(rows),'EM':100*totals[0]/len(rows),'F1':100*totals[1]/len(rows)}
    results=[snapshot(ordered[0]['margin']-1)]
    for i,r in enumerate(ordered):
        old=score('',r['references']); new=score(r['text'],r['references'])
        for j in range(2): totals[j]+=new[j]-old[j]
        if i+1==len(ordered) or ordered[i+1]['margin']!=r['margin']:
            results.append(snapshot(r['margin']))
    return results


def prior_weighted_metrics(metrics, answerable_prior=0.75):
    """Class-conditional performance reweighted to an explicitly specified target mix."""
    if not 0 < answerable_prior < 1: raise ValueError('Prior must be strictly between zero and one')
    if not metrics['answerable']['count'] or not metrics['unanswerable']['count']:
        raise ValueError('Both classes are required')
    return {key:answerable_prior*metrics['answerable'][key]+(1-answerable_prior)*metrics['unanswerable'][key]
            for key in ('EM','F1')}


def weighted_threshold_sweep(rows, answerable_prior=0.75):
    """O(n log n) exact sweep; equal margins switch together to match margin > threshold."""
    if not 0 < answerable_prior < 1: raise ValueError('Invalid target prior')
    n_yes=sum(bool(r['references']) for r in rows); n_no=len(rows)-n_yes
    if not n_yes or not n_no: raise ValueError('Both classes required for prior weighting')
    ordered=sorted(rows,key=lambda r:r['margin'])
    total=[0.,0.]
    weights=[answerable_prior/n_yes if r['references'] else (1-answerable_prior)/n_no for r in ordered]
    for row,weight in zip(ordered,weights):
        old=score('',row['references'])
        for j in range(2): total[j]+=weight*old[j]
    def snapshot(threshold): return {'threshold':threshold,'EM':100*total[0],'F1':100*total[1]}
    results=[snapshot(ordered[0]['margin']-1)]
    for i,(row,weight) in enumerate(zip(ordered,weights)):
        old=score('',row['references']); new=score(row['text'],row['references'])
        for j in range(2): total[j]+=weight*(new[j]-old[j])
        if i+1==len(ordered) or row['margin']!=ordered[i+1]['margin']:
            results.append(snapshot(row['margin']))
    return results
