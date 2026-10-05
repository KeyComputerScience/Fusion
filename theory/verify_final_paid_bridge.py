"""Exact finite checks of the paid bridge; no source fitting or trace reads."""
from collections import defaultdict
from fractions import Fraction as Q
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent


def finite_bridge():
    patterns = ((12, 12, 12), (12, -12, -12), (-12, 12, -12), (-12, -12, 12))
    probability = Q(1, 2*4*129)
    reduced, retained = defaultdict(list), defaultdict(list)
    for sign in (1, -1):
        for residual in patterns:
            for gross in range(-64, 65):
                h = tuple(gross+sign*z for z in residual)
                paid = gross-5
                reduced[h].append(paid)
                retained[sign, h].append(paid)
    masses = {key: probability*len(ds) for key, ds in retained.items()}
    means = {key: Q(sum(ds), len(ds)) for key, ds in retained.items()}
    reports, sufficient_cases = [], 0
    for feasibility in ('all', 'fixed_reduced_mask'):
        G = lambda h: 1 if feasibility == 'all' else int((sum(h)//3) % 5 != 0)
        reduced_value = sum((probability*max(sum(ds), 0)*G(h) for h, ds in reduced.items()), Q(0))
        retained_value = sum((masses[key]*max(mean, 0)*G(key[1]) for key, mean in means.items()), Q(0))
        gap = retained_value-reduced_value
        plusminus = defaultdict(lambda: [Q(0), Q(0)])
        for key, mean in means.items():
            h = key[1]
            plusminus[h][0] += masses[key]*max(mean, 0)*G(h)
            plusminus[h][1] += masses[key]*max(-mean, 0)*G(h)
        assert gap == sum((min(v) for v in plusminus.values()), Q(0)) >= 0
        if feasibility == 'all':
            assert gap == Q(6, 43)
        cases = []
        for bias in (Q(0), Q(-20), Q(20), Q(1, 1000), Q(1, 2)):
            for penalty in (Q(0), Q(1, 1000), Q(1, 2), Q(5), Q(20)):
                value = sum((masses[key]*m*G(key[1]) for key, m in means.items() if m+bias-penalty > 0), Q(0))
                exact_regret = sum((masses[key]*abs(m)*G(key[1]) for key, m in means.items()
                                    if (m+bias-penalty > 0) != (m > 0)), Q(0))
                error = sum((masses[key]*abs(bias)*G(key[1]) for key in means), Q(0))
                rho = sum((masses[key]*penalty*G(key[1]) for key in means), Q(0))
                assert retained_value-value == exact_regret >= 0
                assert exact_regret <= error+rho
                assert value-reduced_value >= gap-error-rho
                # Every reduced policy, including arbitrary randomized ones,
                # is bounded by the reduced oracle under the same G.
                for rule in ('reject', 'admit', 'reduced_oracle', 'randomized'):
                    def action(h, ds):
                        if rule == 'reject':
                            return Q(0)
                        if rule == 'admit':
                            return Q(G(h))
                        if rule == 'reduced_oracle':
                            return Q(G(h)*int(sum(ds) > 0))
                        return Q(G(h)*((sum(h)+1000) % 11), 10)
                    reduced_policy_value = sum((probability*sum(ds)*action(h, ds) for h, ds in reduced.items()), Q(0))
                    assert reduced_policy_value <= reduced_value
                    assert value-reduced_policy_value >= gap-error-rho
                    if gap > error+rho:
                        assert value > reduced_policy_value
                        sufficient_cases += 1
                cases.append(dict(bias=str(bias), penalty=str(penalty), exact_regret=str(exact_regret),
                                  error_plus_penalty=str(error+rho), retained_minus_reduced_oracle=str(value-reduced_value)))
        reports.append(dict(common_feasibility=feasibility, exact_gap=str(gap), cases=cases))
    admission_examples = []
    for means2, weights in (((1, -100), (Q(99, 100), Q(1, 100))),
                           ((100, -1), (Q(1, 100), Q(99, 100)))):
        reduced_mean = sum((p*m for p, m in zip(weights, means2)), Q(0))
        rich_value = sum((p*max(m, 0) for p, m in zip(weights, means2)), Q(0))
        gap = rich_value-max(reduced_mean, 0)
        assert gap == Q(99, 100)
        admission_examples.append(dict(means=means2, probabilities=[str(x) for x in weights],
            retained_admission=str(sum((p for p, m in zip(weights, means2) if m > 0), Q(0))),
            reduced_admission=int(reduced_mean > 0), exact_value_gap=str(gap)))
    return dict(passed=True, finite_outcomes=2*4*129, fitted_cases=50, reduced_policies_per_case=4,
                sufficient_benefit_checks=sufficient_cases, feasibility_reports=reports,
                acceptance_can_increase_or_decrease=admission_examples,
                scope='exact rational enumeration under a coherent finite paired-error generative construction; no empirical physical estimation-error guarantee')


if __name__ == '__main__':
    result = finite_bridge()
    path = HERE/'final_paid_bridge_report.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'feasibility_reports'}, indent=2))
