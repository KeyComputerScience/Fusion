"""Independent paired-rollout calibration for DA-RF candidate values.

The caller supplies actually measured, full-horizon paired gross utilities.
This module cannot infer unobserved counterfactual labels from factual logs.
Held-out maximum residuals are empirical allowances, never confidence bounds.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class PairedRecoverySample:
    candidate_name: str
    inference_name: str
    update: bool
    context: np.ndarray
    loss_state: float
    no_update_gross: float
    update_gross: float
    rollout_id: str

    def __post_init__(self):
        for name in ('candidate_name', 'inference_name', 'rollout_id'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'{name} must be a nonempty string')
        if not isinstance(self.update, (bool, np.bool_)):
            raise ValueError('update must be boolean')
        context = np.array(self.context, float, copy=True)
        if context.ndim != 1 or not np.isfinite(context).all():
            raise ValueError('context must be a finite vector')
        context.setflags(write=False)
        object.__setattr__(self, 'context', context)
        for name in ('loss_state', 'no_update_gross', 'update_gross'):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
            object.__setattr__(self, name, value)
        if not 0 <= self.loss_state <= 1:
            raise ValueError('loss_state must lie in [0, 1]')
        if not self.update and self.update_gross != self.no_update_gross:
            raise ValueError('a no-update sample must have exactly zero paired gain')


@dataclass(frozen=True)
class RecoveryEstimate:
    baseline: float | None
    gain: float | None
    error_allowance: float | None
    calibration_state: str
    support_count: int
    support_distance: float | None
    validation_count: int
    certificate_valid: bool = False


class PairedRecoveryCalibrator:
    """Frozen ridge response models, shared W by inference profile.

    Features are [1, z, y, y*z], so local coefficients c(z)+b(z)*y can be
    obtained before the current fused prediction exists. Standardize z on the
    independent prefix before passing it in; never refit scale on test outcomes.
    Split by rollout_id rather than by rows of the same simulator fork.
    """
    def __init__(self, context_dim: int, ridge=1e-3, support_radius=2.0,
                 min_support=5):
        if not isinstance(context_dim, int) or context_dim < 0:
            raise ValueError('context_dim must be a nonnegative integer')
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError('ridge must be finite and positive')
        if not np.isfinite(support_radius) or support_radius <= 0:
            raise ValueError('support_radius must be finite and positive')
        if not isinstance(min_support, int) or min_support < 1:
            raise ValueError('min_support must be a positive integer')
        self.context_dim, self.ridge = context_dim, float(ridge)
        self.support_radius, self.min_support = float(support_radius), min_support
        self._models = {}
        self._baselines = {}
        self._fitted = False

    def _context(self, values):
        z = np.asarray(values, float)
        if z.shape != (self.context_dim,) or not np.isfinite(z).all():
            raise ValueError('context must match context_dim and be finite')
        return z

    def _features(self, context, loss):
        z = self._context(context)
        if not np.isfinite(loss) or not 0 <= loss <= 1:
            raise ValueError('loss must be finite and lie in [0,1]')
        return np.r_[1.0, z, float(loss), float(loss)*z]

    def _ridge(self, features, labels):
        x, y = np.asarray(features, float), np.asarray(labels, float)
        with np.errstate(over='raise',invalid='raise'):
            result=np.linalg.solve(x.T @ x+self.ridge*np.eye(x.shape[1]), x.T @ y)
        if not np.isfinite(result).all():
            raise ValueError('calibration response could not be fitted finitely')
        return result

    def fit(self, calibration_rows: Sequence[PairedRecoverySample],
            validation_rows: Sequence[PairedRecoverySample]):
        if self._fitted:
            raise ValueError('calibration is frozen after fitting; use a new version to refit')
        train, held = tuple(calibration_rows), tuple(validation_rows)
        if not train:
            raise ValueError('calibration rows are required')
        for row in train+held:
            if not isinstance(row, PairedRecoverySample):
                raise TypeError('rows must be PairedRecoverySample instances')
            self._context(row.context)
        if {r.rollout_id for r in train} & {r.rollout_id for r in held}:
            raise ValueError('calibration and validation rollout groups must be disjoint')
        # Repeated baseline labels for different update arms of the same fork
        # must agree; deduplicate them so the baseline is not multiply weighted.
        def baseline_rows(rows):
            result = {}
            seen_candidates = set()
            for row in rows:
                arm_key = row.rollout_id, row.candidate_name
                if arm_key in seen_candidates:
                    raise ValueError('duplicate candidate arm within one rollout group')
                seen_candidates.add(arm_key)
                key = row.rollout_id, row.inference_name
                if key in result:
                    old = result[key]
                    if (old.no_update_gross != row.no_update_gross
                        or old.loss_state != row.loss_state
                        or not np.array_equal(old.context,row.context)):
                        raise ValueError('matched baseline labels/context differ within a shared fork')
                else:
                    result[key] = row
            return tuple(result.values())
        unique_baselines = baseline_rows(train)
        baseline_rows(held)
        baselines = {}
        for inference in sorted({r.inference_name for r in train}):
            group = [r for r in unique_baselines if r.inference_name == inference]
            baselines[inference] = self._ridge(
                [self._features(r.context,r.loss_state) for r in group],
                [r.no_update_gross for r in group])
        models = {}
        for candidate in sorted({r.candidate_name for r in train}):
            group = [r for r in train if r.candidate_name == candidate]
            inference_set = {r.inference_name for r in group}
            update_set = {r.update for r in group}
            if len(inference_set) != 1 or len(update_set) != 1:
                raise ValueError('a candidate must identify one inference/update configuration')
            inference, update = group[0].inference_name, bool(group[0].update)
            gain_theta = (self._ridge(
                [self._features(r.context,r.loss_state) for r in group],
                [r.update_gross-r.no_update_gross for r in group]) if update
                else np.zeros(2+2*self.context_dim))
            validation = [r for r in held if r.candidate_name == candidate]
            if any(r.inference_name != inference or bool(r.update) != update for r in validation):
                raise ValueError('held-out candidate configuration differs from calibration')
            total_theta = baselines[inference]+gain_theta
            errors = [abs(float(self._features(r.context,r.loss_state) @ total_theta)
                          - r.update_gross) for r in validation]
            models[candidate] = {
                'inference':inference,'update':update,'gain_theta':gain_theta,
                'contexts':np.stack([r.context for r in group]),
                'losses':np.array([r.loss_state for r in group]),
                'error_allowance':float(max(errors)) if errors else None,
                'validation_count':len(validation),
                'rollout_ids':sorted({r.rollout_id for r in group}),
            }
        if any(r.candidate_name not in models for r in held):
            raise ValueError('validation candidate has no calibrated model')
        self._baselines, self._models, self._fitted = baselines, models, True
        return self

    def _support(self, model, context, loss=None):
        z = self._context(context)
        distances = np.linalg.norm(model['contexts']-z,axis=1)
        if loss is not None:
            distances = np.sqrt(distances**2+(model['losses']-loss)**2)
        count = int(np.sum(distances <= self.support_radius))
        nearest = float(distances.min())
        state = ('supported' if count >= self.min_support and model['validation_count'] > 0
                 else 'extrapolated' if nearest <= 2*self.support_radius else 'unsupported')
        return state,count,nearest

    def local_coefficients(self, candidate_name, context):
        if not self._fitted:
            raise ValueError('fit independent calibration data first')
        if candidate_name not in self._models:
            raise KeyError('candidate has no measured calibration data')
        model = self._models[candidate_name]
        z = self._context(context)
        state,count,distance = self._support(model,z)
        if state != 'supported':
            raise ValueError('local response has no supported calibrated slope')
        nearby=np.linalg.norm(model['contexts']-z,axis=1)<=self.support_radius
        loss_span=float(np.ptp(model['losses'][nearby]))
        if loss_span < 1e-6:
            raise ValueError('local data have no loss variation to support a response slope')
        theta = self._baselines[model['inference']]+model['gain_theta']
        d = self.context_dim
        c = float(theta[0]+theta[1:1+d] @ z)
        b = float(theta[1+d]+theta[2+d:] @ z)
        return c,b,{'calibration_state':state,'support_count':count,
                    'support_distance':distance,'local_loss_span':loss_span,
                    'certificate_valid':False}

    def slopes(self, candidate_names, context):
        if len(candidate_names)==0:
            raise ValueError('at least one candidate is required')
        return np.array([self.local_coefficients(n,context)[1] for n in candidate_names])

    def predict(self, candidate_name, context, loss_state):
        phi = self._features(context,loss_state)
        if not self._fitted:
            raise ValueError('fit independent calibration data first')
        if candidate_name not in self._models:
            return RecoveryEstimate(None,None,None,'unsupported',0,None,0)
        model = self._models[candidate_name]
        state,count,distance = self._support(model,context,loss_state)
        baseline = float(phi @ self._baselines[model['inference']])
        gain = float(phi @ model['gain_theta'])
        return RecoveryEstimate(baseline,gain,model['error_allowance'],state,count,
                                distance,model['validation_count'])

    def audit_dict(self):
        return {'fitted':self._fitted,'context_dim':self.context_dim,'ridge':self.ridge,
                'support_radius':self.support_radius,'min_support':self.min_support,
                'allowance_type':'held-out maximum residual; empirical only',
                'support_count_type':'observed rows, not independent effective samples',
                'certificate_valid':False,
                'candidates':{name:{'inference':m['inference'],
                    'update':m['update'],'calibration_rows':len(m['losses']),
                    'validation_rows':m['validation_count'],'error_allowance':m['error_allowance']}
                    for name,m in self._models.items()}}


__all__=['PairedRecoverySample','RecoveryEstimate','PairedRecoveryCalibrator']
