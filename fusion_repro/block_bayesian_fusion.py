"""Analytical conditional Bayesian bootstrap for paired action-error blocks.

Each completed audit window is one block. A block supplies its mean action
error and mean per-request outer product. Positive Dirichlet concentrations
specify posterior mass over block distributions. This conditional bootstrap
does not assert iid observations, calibrated coverage, or new performance.

NumPy is the only dependency. Production moments are analytical: Dirichlet
sampling is provided only for numerical audits, not for the update gate.
"""
from __future__ import annotations

import numpy as np


def _inputs(concentrations, block_means, block_raw_seconds, tolerance):
    a = np.asarray(concentrations, dtype=float)
    means = np.asarray(block_means, dtype=float)
    raw = np.asarray(block_raw_seconds, dtype=float)
    if a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError("concentrations must be a nonempty vector of finite positive values")
    if means.ndim != 2 or means.shape[0] != len(a) or means.shape[1] == 0:
        raise ValueError("block_means must have shape (number_of_blocks, sources)")
    if raw.shape != (len(a), means.shape[1], means.shape[1]):
        raise ValueError("block_raw_seconds must have shape (blocks, sources, sources)")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(raw)):
        raise ValueError("block moments must be finite")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    scale = max(1., float(np.max(np.abs(raw))))
    if float(np.max(np.abs(raw - raw.swapaxes(1, 2)))) > tolerance * scale:
        raise ValueError("block raw second moments must be symmetric")
    raw = (raw + raw.swapaxes(1, 2)) / 2
    centered = raw - np.einsum("gi,gj->gij", means, means)
    if float(np.linalg.eigvalsh(centered).min()) < -tolerance * scale:
        raise ValueError("each block raw second moment must dominate its mean outer product")
    total = float(a.sum())
    if not np.isfinite(total):
        raise ValueError("total concentration must be finite")
    return a, means, raw, centered, total


def block_bayesian_moments(concentrations, block_means, block_raw_seconds,
                           *, tolerance=1e-10):
    """Return exact predictive and parameter moments of the block bootstrap.

    P ~ Dirichlet(a), F_P = sum_g P_g F_g. For Z ~ F_P the predictive
    covariance is E_g[Q_g] - mu mu'. Cov_P(E[Z|P]) instead equals the
    between-block mean covariance divided by sum(a)+1. Within-block
    variation must not be substituted into that latter expression.

    Input concentrations need not be integer counts. When they use context,
    forgetting, or adaptive remapping, this is a conditional generalized
    bootstrap specification, not an iid generative posterior.
    """
    a, means, raw, centered, total = _inputs(
        concentrations, block_means, block_raw_seconds, tolerance)
    mass = a / total
    mean = mass @ means
    second = np.einsum("g,gij->ij", mass, raw)
    within = np.einsum("g,gij->ij", mass, centered)
    between = np.einsum("g,gi,gj->ij", mass, means, means) - np.outer(mean, mean)
    predictive = second - np.outer(mean, mean)
    parameter = between / (total + 1.)
    conditional = predictive - parameter

    # Symmetrization corrects floating-point asymmetry only; no statistical
    # eigenvalue clipping is performed, preserving the exact decompositions.
    def symmetric(value):
        return (value + value.T) / 2

    return dict(total_concentration=total, mean_block_mass=mass,
                predictive_mean=mean,
                predictive_raw_second=symmetric(second),
                predictive_covariance=symmetric(predictive),
                within_block_covariance=symmetric(within),
                between_block_covariance=symmetric(between),
                posterior_mean_covariance=symmetric(parameter),
                expected_conditional_covariance=symmetric(conditional))


def draw_block_posteriors(concentrations, block_means, block_raw_seconds,
                          *, draws=20000, seed=0, tolerance=1e-10):
    """Sample conditional block mass and its induced moments for audits only."""
    if not isinstance(draws, (int, np.integer)) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    a, means, raw, _, _ = _inputs(
        concentrations, block_means, block_raw_seconds, tolerance)
    rng = np.random.default_rng(seed)
    mass = rng.dirichlet(a, size=int(draws))
    mean = mass @ means
    second = np.einsum("dg,gij->dij", mass, raw)
    covariance = second - np.einsum("di,dj->dij", mean, mean)
    return dict(block_mass=mass, mean=mean, raw_second=second,
                covariance=(covariance + covariance.swapaxes(1, 2)) / 2)
