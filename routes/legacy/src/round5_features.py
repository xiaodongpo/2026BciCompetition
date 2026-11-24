"""Window-local ordinal complexity; no fitted parameters or record identity."""
import math
import numpy as np
from src.round2_features import validate, robust_signal, filterbank_features

DELAYS = (1, 2, 4, 8, 16)


def entropy_pair(z, order, delay):
    """Normalized PE and variance-weighted PE, excluding vectors with exact ties.

    All-tied/constant windows return zero. Ties are excluded to avoid artificial
    patterns from clipping plateaus and to preserve polarity invariance.
    This explicitly differs from stable-index tie breaking in some PE libraries.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or order not in (3, 4) or delay < 1:
        raise ValueError('Expected a batch, order 3/4 and positive delay.')
    n = z.shape[1] - (order - 1) * delay
    if n <= 0 or not np.isfinite(z).all():
        raise ValueError('Not enough finite samples for embedding.')
    vectors = np.stack([z[:, j * delay:j * delay + n] for j in range(order)], -1)
    sorted_values = np.sort(vectors, axis=-1)
    valid = (np.diff(sorted_values, axis=-1) > 0).all(-1)
    ranks = np.argsort(vectors, axis=-1, kind='stable')
    codes = (ranks * (order ** np.arange(order))).sum(-1)
    bins = order ** order
    codes += np.arange(len(z))[:, None] * bins
    counts = np.bincount(codes[valid], minlength=len(z) * bins).reshape(len(z), bins)
    variance = vectors.var(-1)
    weighted = np.bincount(codes[valid], weights=variance[valid],
                           minlength=len(z) * bins).reshape(len(z), bins)
    out = []
    for histogram in (counts, weighted):
        total = histogram.sum(-1, keepdims=True)
        p = np.divide(histogram, total, out=np.zeros_like(histogram, dtype=float), where=total > 0)
        log = np.zeros_like(p)
        np.log(p, out=log, where=p > 0)
        out.append(-(p * log).sum(-1) / math.log(math.factorial(order)))
    return np.stack(out, -1)


def ordinal_features(x):
    z, _ = robust_signal(validate(x))
    return np.concatenate([entropy_pair(z, order, delay)
                           for order in (3, 4) for delay in DELAYS], axis=1).astype('float32')


def get_features(x, kind):
    if kind == 'base':
        return filterbank_features(x)
    if kind == 'ordinal':
        return ordinal_features(x)
    if kind == 'fusion':
        return np.concatenate([filterbank_features(x), ordinal_features(x)], axis=1)
    raise ValueError(kind)


def feature_names():
    return [f'{metric}_m{order}_delay{delay}' for order in (3, 4)
            for delay in DELAYS for metric in ('pe', 'weighted_pe')]
