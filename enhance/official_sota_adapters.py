"""Faithful QMF/PDF CORE kernels and explicitly adapted scalar branch models.

These are NOT reproductions of original image/text benchmark results. Real,
trained classifier logits are required; normalised scalar probabilities cannot
reconstruct QMF energy because the common logit offset has been lost.

Sources: QingyangZhang/QMF/text-image-classification/src/models/late_fusion.py;
Yinan-Xia/PDF/src/models/latefusion_pdf.py and train_pdf.py. Original QMF kernel
uses unnormalised logsumexp(logits)/10 coefficients. PDF train omits relative
calibration, which is applied at inference; confidence-feature/weight gradients
are detached in training, matching the checked official implementation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _check_logits(logits):
    if logits.ndim != 3 or logits.shape[-1] < 2:
        raise ValueError("Require actual [batch, modalities, classes>=2] classifier logits")


def qmf_core(logits, divisor=10.0, detach_weights=True):
    """Official text-image QMF fusion coefficient; deliberately NO simplex cap."""
    _check_logits(logits)
    coefficients = torch.logsumexp(logits, dim=-1) / float(divisor)
    applied = coefficients.detach() if detach_weights else coefficients
    return (logits * applied.unsqueeze(-1)).sum(dim=1), coefficients


def pdf_core(logits, mono_confidence, calibrate=True, detach_weights=True, epsilon=1e-8):
    """Official 2-source PDF algebra; M>2 follows paper Appendix A.5.

    Uniform-logit denominator guards are the disclosed numerical extension;
    well-defined two-source inputs match the official forward kernel.
    """
    _check_logits(logits)
    if mono_confidence.shape != logits.shape[:2]:
        raise ValueError("mono_confidence must have [batch, modalities] shape")
    modalities = logits.shape[1]
    if modalities < 2:
        raise ValueError("PDF core requires at least two modalities")
    mono = mono_confidence.clamp(epsilon, 1.0 - epsilon)
    log_mono = mono.log()
    sum_log = log_mono.sum(dim=1, keepdim=True)
    holo = (sum_log - log_mono) / (sum_log + epsilon)
    co_belief = mono + holo
    if calibrate:
        probabilities = logits.softmax(dim=-1)
        uniformity = (probabilities - 1.0 / logits.shape[-1]).abs().mean(dim=-1)
        other_uniformity = (uniformity.sum(dim=1, keepdim=True) - uniformity) / (modalities - 1)
        ratio = uniformity / other_uniformity.clamp_min(epsilon)
        # If every source is uniform, there is no relative preference.
        ratio = torch.where((uniformity <= epsilon) & (other_uniformity <= epsilon),
                            torch.ones_like(ratio), ratio)
        calibration = ratio.clamp(max=1.0)
        co_belief = co_belief * calibration
    else:
        calibration = torch.ones_like(mono)
    weights = co_belief.softmax(dim=1)
    applied = weights.detach() if detach_weights else weights
    return (logits * applied.unsqueeze(-1)).sum(dim=1), weights, {
        "mono": mono, "holo": holo, "co_belief": co_belief,
        "relative_calibration": calibration,
    }


def qmf_ranking_loss(confidence, cumulative_losses):
    """Official history-normalised adjacent-pair margin ranking loss.

    Full batches give deterministic adjacent sample pairs. ``cumulative_losses``
    is [training_samples, modalities] and must contain ONLY training history.
    Degenerate zero-range history receives zero margin rather than division by 0.
    """
    losses = []
    for source in range(confidence.shape[1]):
        history = cumulative_losses[:, source]
        scale = history.max() - history.min()
        normalised = (history - history.min()) / scale.clamp_min(1e-12)
        other = normalised.roll(-1, dims=0)
        sign = torch.sign(normalised - other)
        margin = (normalised - other).abs()
        current, paired = confidence[:, source], confidence[:, source].roll(-1, dims=0)
        # Same margin-ranking expression after substituting the author's target.
        losses.append(F.relu(sign * (current - paired) - margin).mean())
    return torch.stack(losses).sum()


class ScalarBranchFusion(nn.Module):
    """Three-source tabular adaptation, not the authors' BERT/ResNet model.

    Each source gets its own small encoder/classifier. PDF also receives a TCP
    head from detached hidden features. Batch order, task labels, chronology,
    sample count and predeclared features belong in the experiment manifest.
    Missing-modality handling is deliberately excluded from this first adapter.
    """
    def __init__(self, method, modalities=3, features=1, hidden=16, classes=2):
        super().__init__()
        self.method = str(method).upper()
        if self.method not in ("QMF_SCALAR", "PDF_SCALAR", "EF_CLASSIFIER"):
            raise ValueError("Choose QMF_SCALAR, PDF_SCALAR, or EF_CLASSIFIER")
        self.modalities = int(modalities)
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(features, hidden), nn.ReLU())
                                       for _ in range(modalities)])
        self.classifiers = nn.ModuleList([nn.Linear(hidden, classes) for _ in range(modalities)])
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden * 2), nn.Linear(hidden * 2, hidden),
                          nn.Linear(hidden, 1), nn.Sigmoid()) for _ in range(modalities)])

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1] != self.modalities:
            raise ValueError("inputs must be [batch, modalities, features]")
        hidden = [encoder(inputs[:, index]) for index, encoder in enumerate(self.encoders)]
        logits = torch.stack([classifier(value) for classifier, value in zip(self.classifiers, hidden)], dim=1)
        if self.method == "QMF_SCALAR":
            fused, confidence = qmf_core(logits)
            diagnostics = {"coefficients": confidence}
        elif self.method == "PDF_SCALAR":
            confidence = torch.cat([head(value.detach()) for head, value in zip(self.confidence_heads, hidden)], dim=1)
            fused, weights, diagnostics = pdf_core(logits, confidence, calibrate=not self.training)
            diagnostics["weights"] = weights
        else:
            fused = logits.mean(dim=1)
            confidence = torch.full(logits.shape[:2], 1.0 / self.modalities, device=logits.device)
            diagnostics = {"weights": confidence}
        return fused, logits, confidence, diagnostics

    def training_loss(self, inputs, labels, cumulative_losses=None):
        fused, branches, confidence, _ = self(inputs)
        branch_loss = torch.stack([F.cross_entropy(branches[:, source], labels, reduction="none")
                                   for source in range(self.modalities)], dim=1)
        classification = F.cross_entropy(fused, labels) + branch_loss.mean(dim=0).sum()
        if self.method == "PDF_SCALAR":
            probabilities = branches.softmax(dim=-1)
            true_class = probabilities.gather(-1, labels[:, None, None].expand(-1, self.modalities, 1)).squeeze(-1)
            auxiliary = (confidence - true_class.detach()).abs().mean(dim=0).sum()
        elif self.method == "QMF_SCALAR":
            if cumulative_losses is None:
                raise ValueError("QMF requires per-sample training-loss history")
            auxiliary = qmf_ranking_loss(confidence, cumulative_losses + branch_loss.detach())
        else:
            auxiliary = classification.new_tensor(0.0)
        return classification + auxiliary, branch_loss.detach()
