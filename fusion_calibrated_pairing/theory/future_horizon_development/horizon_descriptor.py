"""UNFROZEN future descriptor prototype; no law fitting or admission.

Only source probabilities, current covariates/model versions and masks
are read. Labels, future covariates and complete-lease outcomes are absent.
This file does not change the delivered CJRT-P numerical method.
"""
from collections import deque
import numpy as np

VERSION = "future-horizon-descriptor-development-v0"
LAGS = 3


def softmax(logits):
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    out = np.exp(z)
    return out / out.sum(axis=-1, keepdims=True)


def _probabilities(value, shape, name):
    p = np.asarray(value, dtype=float)
    if p.shape != shape or not np.all(np.isfinite(p)):
        raise ValueError(f"Invalid {name} probability shape/value")
    if np.any(p < -1e-10) or np.max(abs(p.sum(axis=-1) - 1.)) > 1e-8:
        raise ValueError(f"Invalid normalized {name} probabilities")
    return p


def issued_snapshot(event):
    """Read exactly one current event, without consulting event['y'].

    Historical snapshots are never recomputed with subsequently trained
    model versions. Source blocks are missing when their issued mask is
    false, even if an offline replay stores an unavailable forecast.
    """
    x = np.asarray(event["x"], dtype=float)
    p = np.asarray(event["p"], dtype=float)
    if x.ndim != 2 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Current covariates must be nonempty and finite")
    if p.ndim != 3 or p.shape[1] != len(x) or p.shape[2] < 2:
        raise ValueError("Expected issued [sources,requests,classes] probabilities")
    m, n, classes = p.shape
    p = _probabilities(p, (m, n, classes), "source")
    mask = np.asarray(event["mask"], dtype=bool)
    if mask.shape != (m,):
        raise ValueError("Unaligned issued source mask")
    pc = softmax(x @ np.asarray(event["candidate"], dtype=float))
    pr = softmax(x @ np.asarray(event["reference"], dtype=float))
    pc = _probabilities(pc, (n, classes), "candidate")
    pr = _probabilities(pr, (n, classes), "reference")
    fused = _probabilities(event["external_probability"], (n, classes), "common fused")
    soft_contrast = pc - pr
    hard_contrast = np.eye(classes)[pc.argmax(1)] - np.eye(classes)[pr.argmax(1)]
    hard_disagreement = float(np.mean(np.any(hard_contrast != 0., axis=1)))
    soft_disagreement = float(np.mean(.5 * abs(soft_contrast).sum(axis=1)))
    hard_anchor = float(np.mean(np.einsum("ic,ic->i", fused, hard_contrast)))
    soft_anchor = float(np.mean(np.einsum("ic,ic->i", fused, soft_contrast)))
    hard = np.mean(np.einsum("sic,ic->si", p, hard_contrast), axis=1)
    soft = np.mean(np.einsum("sic,ic->si", p, soft_contrast), axis=1)
    entropy = -np.mean(np.sum(p * np.log(np.maximum(p, 1e-12)), axis=2), axis=1)
    source_top = np.sort(p, axis=2)
    source_margin = np.mean(source_top[:, :, -1] - source_top[:, :, -2], axis=1)
    # Each source's complete probability profile and two directional
    # evidences remain in ONE conditional block. No forecast head is added.
    blocks = np.column_stack((p.mean(axis=1), hard-hard_anchor,
                              soft-soft_anchor, entropy, source_margin))
    blocks = np.where(mask[:, None], blocks, 0.)
    pc_top, pr_top = np.sort(pc, axis=1), np.sort(pr, axis=1)
    common = np.r_[pc.mean(axis=0), pr.mean(axis=0), fused.mean(axis=0),
                   np.mean(pc_top[:, -1]-pc_top[:, -2]),
                   np.mean(pr_top[:, -1]-pr_top[:, -2]),
                   hard_disagreement, soft_disagreement, np.mean(mask)]
    return dict(blocks=blocks.copy(), source_mask=mask.copy(), common=common.copy(),
                hard_anchor=hard_anchor, soft_anchor=soft_anchor,
                hard_disagreement=hard_disagreement,
                soft_disagreement=soft_disagreement,
                state_weight_support=max(hard_disagreement, soft_disagreement),
                sources=m, classes=classes, requests=n,
                block_width=classes+4)


class HorizonDescriptor:
    """One immutable path descriptor from the current and preceding issues.

    Call reset on physical-session/recording boundaries. A changed
    recording identifier automatically resets the path. Absent startup
    lags and masked source/lags have false coordinate masks, not observed
    imputed values. No source inference or model update is performed here.
    """
    def __init__(self, lags=LAGS):
        if lags != LAGS:
            raise ValueError("This development prototype has fixed three issue windows")
        self.history = deque(maxlen=lags)
        self.recording = None
        self.schema = None

    def reset(self, recording=None):
        self.history.clear()
        self.recording = recording
        self.schema = None

    def push(self, event, recording):
        if recording != self.recording:
            self.reset(recording)
        snapshot = issued_snapshot(event)
        schema = (snapshot["sources"], snapshot["classes"])
        if self.schema is not None and schema != self.schema:
            raise ValueError("Source/class schema changed inside a physical session")
        self.schema = schema
        self.history.append(snapshot)
        m, width = snapshot["sources"], snapshot["block_width"]
        blocks = np.zeros((m, LAGS, width))
        observed = np.zeros((m, LAGS, width), dtype=bool)
        for lag, old in enumerate(reversed(self.history)):
            blocks[:, lag] = old["blocks"]
            observed[:, lag] = old["source_mask"][:, None]
        common = np.r_[snapshot["common"], len(self.history)/LAGS]
        return dict(version=VERSION, recording=recording,
                    source_blocks=blocks.reshape(m, LAGS*width).copy(),
                    source_coordinate_mask=observed.reshape(m, LAGS*width).copy(),
                    common_descriptor=common,
                    hard_anchor=snapshot["hard_anchor"],
                    soft_anchor=snapshot["soft_anchor"],
                    hard_disagreement=snapshot["hard_disagreement"],
                    soft_disagreement=snapshot["soft_disagreement"],
                    state_weight_support=snapshot["state_weight_support"],
                    active_sources=np.flatnonzero(snapshot["source_mask"]),
                    history_frames=len(self.history),
                    source_block_width=LAGS*width,
                    schema=dict(sources=m, classes=snapshot["classes"], lags=LAGS))
