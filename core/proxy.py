"""Causal delayed-recovery forecasts and their conditional tangent bounds."""
from __future__ import annotations

import math
from dataclasses import dataclass


def geometric_sum(retention: float, terms: int) -> float:
    """1 + rho + ... + rho**(terms-1), with stable rho=0/1 limits."""
    if not 0 <= retention <= 1 or not isinstance(terms, int) or isinstance(terms, bool) or terms < 0:
        raise ValueError("Retention must lie in [0,1]; terms must be nonnegative")
    if terms == 0:
        return 0.0
    if retention == 1:
        return float(terms)
    if retention == 0:
        return 1.0
    logarithm = math.log(retention)
    return math.expm1(terms * logarithm) / math.expm1(logarithm)


@dataclass
class RecoveryForecast:
    """baseline[h] = recovery at t+h before the currently considered update."""
    baseline: list[float]
    retentions: list[float]
    attenuation: float
    constant_retention: float | None = None

    @classmethod
    def construct(cls, recovery: float, retention: float, horizon: int, attenuation: float,
                  future_retentions: list[float] | None = None,
                  pending_gains: dict[int, float] | None = None):
        if (not math.isfinite(recovery) or recovery < 0 or not isinstance(horizon, int)
                or isinstance(horizon, bool) or horizon < 0):
            raise ValueError("Recovery and horizon must be nonnegative")
        if not 0 <= attenuation <= 1 or not 0 <= retention <= 1:
            raise ValueError("Retention and attenuation must lie in [0,1]")
        retentions = [retention] * horizon if future_retentions is None else list(future_retentions[:horizon])
        if len(retentions) != horizon or any(not 0 <= rho <= 1 for rho in retentions):
            raise ValueError("A causal forecast requires one valid retention per lookahead slot")
        pending = pending_gains or {}
        if any(not isinstance(lag, int) or lag < 1 or not math.isfinite(gain) or gain < 0
               for lag, gain in pending.items()):
            raise ValueError("Pending deployments must have positive integer lags and nonnegative gains")
        baseline = [recovery]
        for h, rho in enumerate(retentions, 1):
            baseline.append(rho * baseline[-1] + pending.get(h, 0.0))
        constant = retentions[0] if retentions and all(rho == retentions[0] for rho in retentions) else None
        return cls(baseline, retentions, attenuation, constant)

    def factors(self, deployment_delay: int):
        if not isinstance(deployment_delay, int) or isinstance(deployment_delay, bool) or deployment_delay < 1:
            raise ValueError("Updates affect only future recovery states")
        factor = 1.0
        for h in range(deployment_delay, len(self.baseline)):
            if h > deployment_delay:
                factor *= self.retentions[h - 1]
            yield h, factor

    def coefficient(self, deployment_delay: int, mode: str = "tangent") -> float:
        if not isinstance(deployment_delay, int) or isinstance(deployment_delay, bool) or deployment_delay < 1:
            raise ValueError("Deployment delay must be a positive integer")
        if mode == "tangent":
            return self.attenuation * sum(math.exp(-self.baseline[h]) * factor
                                          for h, factor in self.factors(deployment_delay))
        if mode == "geometric_upper":
            if self.constant_retention is not None:
                terms = max(0, len(self.retentions) - deployment_delay + 1)
                return self.attenuation * geometric_sum(self.constant_retention, terms)
            return self.attenuation * sum(factor for _, factor in self.factors(deployment_delay))
        raise ValueError("Proxy mode must be tangent or geometric_upper")

    def exact_increment(self, gain: float, deployment_delay: int) -> float:
        """Exact incremental quality within this forecast, not actual DRL return."""
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("Gain must be finite and nonnegative")
        return self.attenuation * sum(math.exp(-self.baseline[h]) * (-math.expm1(-factor * gain))
                                     for h, factor in self.factors(deployment_delay))
