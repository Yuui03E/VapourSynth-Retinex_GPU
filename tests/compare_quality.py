"""Shared range normalization and output-difference metrics for the comparison harness."""
from __future__ import annotations


import numpy as np
import vapoursynth as vs

PEAK = 65535.0


def array(node, n):
    with node.get_frame(int(n)) as f:
        return np.array(f[0], dtype=np.float32)


def normalize(a, limited=False):
    return np.clip((a - 4096) / 56064 if limited else a / PEAK, 0, 1)


def difference(a, b, regions=None):
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite output invalidates comparison metrics')
    d = np.abs(a - b)
    row = dict(mae=float(d.mean()), rmse=float(np.sqrt(np.mean(d*d))),
               p99=float(np.quantile(d, .99)), maximum=float(d.max()),
               fraction_over_1pct=float(np.mean(d > .01)))
    if regions:
        row['regions'] = {k: dict(pixels=int(m.sum()), mae=float(d[m].mean()))
                          for k, m in regions.items() if m.any()}
    return row


def binary_agreement(a, b, threshold):
    x, y = a >= threshold, b >= threshold
    union = np.count_nonzero(x | y)
    return dict(threshold=threshold, disagree=float(np.mean(x != y)),
                iou=float(np.count_nonzero(x & y) / union) if union else 1.,
                reference_coverage=float(x.mean()), candidate_coverage=float(y.mean()))


def summarize(rows):
    out = {key: float(np.mean([r[key] for r in rows]))
           for key in ('mae', 'fraction_over_1pct')}
    out['rmse'] = float(np.sqrt(np.mean([r['rmse']**2 for r in rows])))
    out['mean_frame_p99'] = float(np.mean([r['p99'] for r in rows]))
    out['maximum'] = max(r['maximum'] for r in rows)
    out['frames'] = len(rows)
    regions = sorted({k for r in rows for k in r.get('regions', {})})
    out['regions'] = {}
    for k in regions:
        vals = [r['regions'][k] for r in rows if k in r.get('regions', {})]
        count = sum(v['pixels'] for v in vals)
        out['regions'][k] = dict(pixels=count,
            mae=sum(v['pixels'] * v['mae'] for v in vals) / count)
    return out
