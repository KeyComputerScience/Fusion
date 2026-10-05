"""Reporting-only sequential risk monitor; never alters fusion decisions.

Use one causal physical path, not pooled repetitions of the same trace.
The theorem targets average latent pre-lease conditional risk in the enlarged
physical filtration. It is not a posterior credibility or future coverage bound.
Release observations only in a fully matured origin-ordered prefix.
"""
import math


def upper_conditional_average(observed, delta=0.05):
    assert 0 < delta < 1
    values = [float(x) for x in observed]
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in values)
    n = len(values)
    if n == 0:
        return dict(count=0, mean=None, upper=None, delta=delta)
    width = math.sqrt(math.log(math.pi**2 * n**2 / (6 * delta)) / (2 * n))
    average = sum(values) / n
    return dict(count=n, mean=average, width=width,
                upper=min(1.0, average + width), delta=delta)


def matured_prefix(rows, now, delta=0.05, maximum_capacity=128):
    admitted = sorted((r for r in rows if r['action']), key=lambda r: r['k'])
    ready = []
    for row in admitted:
        if row['maturity'] > now:
            break
        ready.append(row)
    violations = [int(not r['lower_covered']) for r in ready]
    harmful = [int(r['local_net'] < 0) for r in ready]
    excess = [max(0., r['gate_score'] - (r['truegross'] - 5.)) /
              (2 * maximum_capacity) for r in ready]
    assert all(x <= 1 + 1e-10 for x in excess)
    return dict(now=now, observed_origins=[r['k'] for r in ready],
                pending_or_blocked=len(admitted)-len(ready),
                violation=upper_conditional_average(violations, delta),
                harmful=upper_conditional_average(harmful, delta),
                normalized_excess=upper_conditional_average(excess, delta))


def checks():
    # Late origin 0 blocks origin 4 even when the latter label arrives first.
    rows = [dict(k=0, maturity=9, action=True, lower_covered=True,
                 local_net=3., gate_score=1., truegross=8.),
            dict(k=4, maturity=8, action=True, lower_covered=False,
                 local_net=-2., gate_score=4., truegross=3.)]
    assert matured_prefix(rows, 8)['observed_origins'] == []
    assert matured_prefix(rows, 9)['observed_origins'] == [0, 4]
    poisoned = [dict(r, local_net=1e30, gate_score=-1e30) for r in rows]
    assert matured_prefix(poisoned, 8)['observed_origins'] == []
    assert upper_conditional_average([])['upper'] is None
    assert upper_conditional_average([0.]*10000)['upper'] < 0.04
    return dict(delayed_prefix='pass', empty='pass', bounded='pass',
                role='risk reporting; no fusion/admission changes')


if __name__ == '__main__':
    import json
    print(json.dumps(checks(), indent=2))
