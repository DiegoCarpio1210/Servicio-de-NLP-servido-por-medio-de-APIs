"""Official SQuAD normalization and maximum-over-references EM/F1."""
from collections import Counter
import re
import string


def normalize(s):
    s = ''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r"\b(a|an|the)\b", ' ', s).split())


def score(prediction, references):
    p = normalize(prediction).split()
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
