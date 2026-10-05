"""Fail-closed adapter for the frozen masked-precision forecast.

This adapter is separate from the scientific replay runner. It consumes only
current forecasts, model versions, arrived state, matured calibration and
current reserve/ledger information. It never uses offline lease outcomes.
Durable queues, atomic settlement and the underlying service contract remain
the caller's responsibility; this function does not implement either.
"""
from __future__ import annotations
import math
import numpy as np


def _finite_array(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(name + ': invalid shape or nonfinite value')
    return a.copy()


def _psd(value, m, name):
    a = _finite_array(value, (m, m), name)
    scale = max(1.0, float(np.max(np.abs(a))))
    if np.max(np.abs(a-a.T)) > 1e-10*scale:
        raise ValueError(name + ': nonsymmetric matrix')
    if np.linalg.eigvalsh(a).min() < -1e-10*scale:
        raise ValueError(name + ': indefinite matrix')
    return a


def _number(value, name, minimum=0.0, positive=False):
    x = float(value)
    if not math.isfinite(x) or x < minimum or (positive and x <= minimum):
        raise ValueError(name + ': invalid value')
    return x


def _retain(reason, detail=None):
    return dict(action=False, reason=reason, detail=detail, weights=[],
                gain=None, predictive_scale=None, score=None, certificate=None)


def decide(decision, pre, cfg, q, *, service_feasible, ledger, core=None):
    """Validate and issue the primary precision-joint action, or retain reference.

    `ledger` contains current nonnegative C, Q, B and uniform reserve M >= 130.
    This interface enforces Fixed130; tighter certified decision reserves use
    the separately verified reserve service, not an unchecked input here.
    An admitted M must be atomically reserved by the caller before service.
    `decision` accepts current ids/h/N/models and temporal mu/R/P/eligible;
    extra fields, including offline outcomes, are never forwarded or read.
    """
    if core is None:
        import temporal_fusion as core
    try:
        raw_ids = np.asarray(decision['ids'])
        if raw_ids.ndim != 1:
            raise ValueError('source IDs must be a vector')
        if len(raw_ids) == 0:
            return _retain('empty_source_mask')
        if not np.issubdtype(raw_ids.dtype, np.integer):
            raise ValueError('source IDs must be integers')
        ids = raw_ids.astype(int, copy=True)
        m = len(ids)
        quality = np.asarray(pre['q'], dtype=float)
        if quality.ndim != 1 or not len(quality) or not np.all(np.isfinite(quality)) or np.min(quality) <= 0:
            raise ValueError('quality must be finite, positive and one-dimensional')
        if len(set(ids.tolist())) != m or np.min(ids) < 0 or np.max(ids) >= len(quality):
            raise ValueError('invalid or duplicated source IDs')
        h = _finite_array(decision['h'], (m,), 'current contrast forecasts')
        if np.max(np.abs(h)) > 1.0+1e-10:
            raise ValueError('contrast forecasts outside probability contrast range')
        N = _number(decision['N'], 'N', positive=True)
        horizon = int(cfg['horizon']); window = int(cfg['window'])
        if horizon != 4 or window != 32 or N > horizon*window or N != int(N) or int(N) % horizon:
            raise ValueError('N or horizon violates the frozen paid-lease contract')
        if cfg['deploy_drop'] != 2 or cfg['deploy_fee'] != 2 or cfg['probe_drop'] < 1:
            raise ValueError('fee/interruption contract requires a revised reserve')
        models = [np.asarray(decision[n], dtype=float) for n in ('candidate', 'reference')]
        if any(a.ndim != 2 or not np.all(np.isfinite(a)) for a in models) or models[0].shape != models[1].shape or min(models[0].shape) < 1 or models[0].shape[1] < 2:
            raise ValueError('candidate/reference model dimensions or values invalid')
        if int(pre['classes']) != models[0].shape[1]:
            raise ValueError('model class count inconsistent with source forecasts')
        if 'source_probabilities' in decision:
            probabilities = np.asarray(decision['source_probabilities'], dtype=float)
            if probabilities.ndim != 3 or probabilities.shape[0] != m or probabilities.shape[2] != pre['classes'] or not np.all(np.isfinite(probabilities)) or np.min(probabilities) < 0 or np.max(probabilities) > 1 or np.max(np.abs(probabilities.sum(2)-1)) > 1e-6:
                raise ValueError('invalid issued source probabilities')
        mm = decision['temporal']
        mu = _finite_array(mm['mu'], (m,), 'correction mean')
        R = _psd(mm['R'], m, 'predictive covariance')
        P = _psd(mm['P'], m, 'correction uncertainty')
        k = int(decision['k'])
        eligible = []
        for o in mm['eligible']:
            origin, maturity = int(o['origin']), int(o['maturity'])
            weight = _number(o['weight'], 'arrived block weight', positive=True)
            observed = int(o['observed_components'])
            if origin > maturity or maturity > k or observed < 1 or observed > len(quality):
                raise ValueError('invalid source block or unmatured historical information')
            eligible.append(dict(origin=origin, maturity=maturity, weight=weight, observed_components=observed))
        complete_blocks = int(mm['complete_blocks'])
        if complete_blocks < 0 or complete_blocks > len(eligible):
            raise ValueError('invalid complete-block count')
        q = _number(q, 'matured calibration threshold')
        if q < _number(cfg.get('q_floor', 0), 'calibration floor')-1e-12:
            raise ValueError('matured calibration threshold below selected floor')
        _number(cfg['tau'], 'entropy coefficient', positive=True)
        _number(cfg['norm_floor'], 'norm floor', positive=True)
        cap = _number(cfg['cap'], 'source cap', positive=True)
        if cap > 1:
            raise ValueError('source cap exceeds one')
        _number(cfg['quality_power'], 'quality power', positive=True)
        _number(cfg['threshold'], 'admission margin')
        C, Q, B, M = [_number(ledger[n], 'ledger '+n) for n in ('C', 'Q', 'B', 'M')]
        if M < 130.0:
            raise ValueError('this interface requires the certified uniform reserve of at least 130')
        if C+Q > B+1e-12:
            raise ValueError('invalid existing ledger invariant')
        if type(service_feasible) is not bool:
            raise ValueError('service feasibility must be explicit Boolean')
        if not service_feasible:
            return _retain('service_infeasible')
        if C+Q+M > B:
            return _retain('loss_budget_insufficient')
        # Only these fields enter the frozen forecast. Its required offline
        # diagnostic target is replaced by a constant and discarded below.
        safe = dict(k=k, ids=ids, h=h, N=N, truegross=0.0,
                    temporal=dict(mu=mu, R=R, P=P, eligible=eligible,
                                  complete_blocks=complete_blocks))
        w, cert, gain, sd, _unused_offline_score, extra = core.forecast(
            safe, dict(q=quality.copy()), dict(cfg), 'precision_joint', q)
        w = _finite_array(w, (m,), 'solver influence')
        actual_cap = max(cap, 1/m)
        primal = max(abs(float(w.sum())-1), float(np.max(w-actual_cap)), float(-np.min(w)))
        if not cert.get('converged', False) or not math.isfinite(float(cert.get('kkt', math.inf))) or float(cert['kkt']) >= 1e-6 or not math.isfinite(float(cert.get('primal', math.inf))) or float(cert['primal']) >= 1e-8 or primal >= 1e-8:
            return _retain('uncertified_solver', dict(cert))
        gain = _number(gain, 'conservative forecast', minimum=-math.inf)
        sd = _number(sd, 'predictive scale', positive=True)
        score = gain-q*sd-5.0-float(cfg['threshold'])
        return dict(action=bool(score > 0), reason='paid_value_admission' if score > 0 else 'paid_value_retention',
                    weights=w.tolist(), gain=gain, predictive_scale=sd, score=score,
                    certificate=dict(cert), information_ready=bool(extra['information_ready']))
    except (KeyError, TypeError, ValueError, IndexError, ArithmeticError, np.linalg.LinAlgError) as error:
        return _retain('invalid_input_or_numerical_failure', str(error))
