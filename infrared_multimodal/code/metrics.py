"""CMC, mAP and mINP; SYSU uses unique-identity CMC and camera-3/2 exclusion."""
import numpy as np


def retrieval(distances, query, gallery, sysu=False, max_rank=20):
    qids, gids = np.array([r.pid for r in query]), np.array([r.pid for r in gallery])
    gcams = np.array([r.camera for r in gallery])
    curves, aps, inps = [], [], []
    for i, order in enumerate(np.argsort(distances, axis=1, kind='stable')):
        if sysu and query[i].camera == 3:
            order = order[gcams[order] != 2]
        labels = gids[order]
        matches = labels == qids[i]
        if not matches.any():
            continue
        cumulative = matches.cumsum()
        aps.append(float(((cumulative / np.arange(1, len(matches) + 1)) * matches).sum() / matches.sum()))
        last = np.flatnonzero(matches)[-1]
        inps.append(float(cumulative[last] / (last + 1)))
        if sysu:
            first = np.sort(np.unique(labels, return_index=True)[1])
            matches = labels[first] == qids[i]
        curve = np.minimum(matches.cumsum(), 1)
        curves.append(np.pad(curve[:max_rank], (0, max(0, max_rank-len(curve))), mode='edge'))
    if not curves:
        raise ValueError('No valid query has a match after protocol exclusions.')
    cmc = np.mean(curves, axis=0)
    return {'rank1': float(cmc[0]), 'rank5': float(cmc[4]), 'rank10': float(cmc[9]),
            'mAP': float(np.mean(aps)), 'mINP': float(np.mean(inps)),
            'queries': len(query), 'valid_queries': len(curves), 'gallery': len(gallery)}
