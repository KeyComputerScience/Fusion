"""Evidence and observation-quality helpers for the DA-RF reference code.

Arrays are chronological. Missing rows remain NaN during block resampling;
they are not treated as evidence of no change or of a zero target.
"""
from __future__ import annotations
import numpy as np


def normalized_js_divergence(previous, current):
    """Jensen--Shannon diagnostic in [0,1]; None for an empty distribution."""
    p, q = np.asarray(previous, float), np.asarray(current, float)
    if p.ndim != 1 or p.shape != q.shape or len(p) == 0:
        raise ValueError("equal, nonempty count/distribution vectors required")
    if not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any() or (q < 0).any():
        raise ValueError("counts must be finite and nonnegative")
    if p.sum() == 0 or q.sum() == 0:
        return None
    p, q = p / p.sum(), q / q.sum()
    mean = (p + q) / 2
    def kl(x):
        m = x > 0
        return np.sum(x[m] * np.log(x[m] / mean[m]))
    return float(np.clip((kl(p) + kl(q)) / (2 * np.log(2)), 0, 1))


def operating_change(previous, current, prefix_mean, prefix_std,
                     min_rows=2, scale_floor=1e-6):
    """Adjacent operating means, using only fixed prefix standardization."""
    old, new = np.asarray(previous, float), np.asarray(current, float)
    mean, std = np.asarray(prefix_mean, float), np.asarray(prefix_std, float)
    if old.ndim != 2 or new.ndim != 2 or old.shape[1:] != new.shape[1:]:
        raise ValueError("previous/current must have matching feature dimensions")
    if mean.shape != (old.shape[1],) or std.shape != mean.shape:
        raise ValueError("prefix normalizer dimension mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any() or scale_floor <= 0:
        raise ValueError("invalid fixed prefix normalizer")
    if min_rows < 1:
        raise ValueError("min_rows must be positive")
    counts_old, counts_new = np.isfinite(old).sum(0), np.isfinite(new).sum(0)
    if (counts_old < min_rows).any() or (counts_new < min_rows).any():
        return None
    old_mean = np.where(np.isfinite(old), old, 0).sum(0) / counts_old
    new_mean = np.where(np.isfinite(new), new, 0).sum(0) / counts_new
    return float(np.mean(np.minimum(1, np.abs(new_mean-old_mean)/np.maximum(std,scale_floor))))


def matched_service_change(previous: dict, current: dict,
                           min_context_count=2, epsilon=1e-6):
    """Input context -> (valid count, gross-utility mean); no common support => None."""
    if min_context_count < 1 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("invalid context support/normalizer")
    pairs = []
    for context in previous.keys() & current.keys():
        n0, value0 = previous[context]
        n1, value1 = current[context]
        if n0 < 0 or n1 < 0 or not np.isfinite([n0,n1,value0,value1]).all():
            raise ValueError("invalid context statistics")
        if min(n0, n1) >= min_context_count:
            pairs.append((min(n0,n1), float(value0), float(value1)))
    if not pairs:
        return None
    mass = sum(v[0] for v in pairs)
    old = sum(n*a for n,a,b in pairs)/mass
    new = sum(n*b for n,a,b in pairs)/mass
    return float(min(1, max(0, old-new)/(abs(old)+epsilon)))


def persistent_loss(reference, gross_service, scale=1.0, min_slots=2):
    """Aggregate issued references and outcomes over the identical valid slots.

    The caller must issue reference predictions before each slot's service;
    this numerical helper cannot verify the provenance of supplied arrays.
    """
    ref, actual = np.asarray(reference,float), np.asarray(gross_service,float)
    if ref.ndim != 1 or ref.shape != actual.shape:
        raise ValueError("reference and service must describe the same slot set")
    if not np.isfinite(scale) or scale <= 0 or min_slots < 1:
        raise ValueError("positive scale and min_slots required")
    valid = np.isfinite(ref) & np.isfinite(actual)
    count = int(valid.sum())
    if count < min_slots:
        return None, {"valid_slots":count,"target_available":False}
    baseline, observed = float(ref[valid].mean()),float(actual[valid].mean())
    y=float(np.clip((baseline-observed)/scale,0,1))
    return y,{"valid_slots":count,"target_available":True,
              "reference_mean":baseline,"service_mean":observed}


def block_bootstrap_variance(samples, *, rng, replicates=200,
                             block_length=None, fallback_variance=0.25):
    """Variance of a chronological nan-aware mean, preserving within-block gaps.

    This is an empirical quality statistic, not an independent-sample error
    bound. Entirely unavailable replicates are skipped, not assigned mean zero.
    """
    values=np.asarray(samples,float)
    if values.ndim != 1 or np.isinf(values).any():
        raise ValueError("chronological finite/NaN samples required")
    if not isinstance(rng,np.random.Generator):
        raise TypeError("supply a recorded NumPy Generator")
    if replicates < 2 or not np.isfinite(fallback_variance) or fallback_variance < 0:
        raise ValueError("at least two replicates and nonnegative fallback required")
    size=len(values)
    if size < 2 or np.isfinite(values).sum()<2:
        return float(fallback_variance),{"valid_replicates":0,"fallback":True,"block_length":0}
    length=min(size,max(2,int(np.ceil(size**(1/3))))) if block_length is None else block_length
    if not isinstance(length,(int,np.integer)) or not 1<=length<=size:
        raise ValueError("block_length must lie between 1 and the record length")
    means=[]
    for _ in range(replicates):
        starts=rng.integers(0,size-length+1,size=int(np.ceil(size/length)))
        indices=(starts[:,None]+np.arange(length)[None,:]).ravel()[:size]
        draw=values[indices]
        valid=np.isfinite(draw)
        if valid.any():
            means.append(float(draw[valid].mean()))
    fallback=len(means)<2
    variance=fallback_variance if fallback else np.var(means,ddof=1)
    return float(variance),{"valid_replicates":len(means),"fallback":fallback,
                            "block_length":int(length)}
