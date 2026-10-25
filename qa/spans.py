"""SQuAD v2 minimum-null aggregation; offsets always index original text."""
import math
import numpy as np


def aggregate(context, windows, max_answer_length=30):
    best_score, best_start, best_end = -math.inf, None, None
    null_score = math.inf
    for starts, ends, offsets, cls in windows:
        starts, ends = np.asarray(starts), np.asarray(ends)
        if not np.isfinite(starts).all() or not np.isfinite(ends).all():
            raise ValueError("Non-finite model output")
        null_score = min(null_score, float(starts[cls] + ends[cls]))
        # Exhaustive bounded spans avoid the invalid top-k token pitfall.
        for i, off in enumerate(offsets):
            if off is None or off[1] <= off[0]:
                continue
            for j in range(i, min(i + max_answer_length, len(offsets))):
                other = offsets[j]
                if other is None:
                    break
                a, b = off[0], other[1]
                if 0 <= a < b <= len(context):
                    score = float(starts[i] + ends[j])
                    if score > best_score:
                        best_score, best_start, best_end = score, a, b
    if best_start is None:
        return {"text": "", "start": None, "margin": 1e6}
    return {"text": context[best_start:best_end], "start": best_start,
            "margin": null_score - best_score}


def decide(candidate, threshold):
    no_answer = candidate["start"] is None or candidate["margin"] > threshold
    distance = abs(candidate["margin"] - threshold)
    confidence = 1 / (1 + math.exp(-min(distance, 700)))
    return {"answer": None if no_answer else candidate["text"],
            "start": None if no_answer else candidate["start"], "score": confidence}
