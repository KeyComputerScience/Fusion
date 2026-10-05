"""Independently serve saved supplementary actions; never infer gross from local_net."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', type=Path, default=ROOT / 'bayes_closed_loop_repro')
    p.add_argument('--results', type=Path, default=ROOT / 'performance_gain_evidence/results.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    package = args.package.resolve()
    sys.path.insert(0, str(package))
    import cached_runner
    engine, adapter = cached_runner.install_cached_loaders()
    from audit_canary_execution import load_study, explicit_service, explicit_fork
    studies = json.loads(args.results.read_text())
    checks = []
    for name, task in studies.items():
        folder = package / ('new_bayes_fusion' if name.startswith('occupancy') else 'new_bayes_extension')
        data, pre, saved = load_study(package / 'independent_data', folder, name)
        if data['hashes'] != task['data_hashes'] or pre['split'] != task['split']:
            raise AssertionError('Data hash or split differs: ' + name)
        for trial in task['trials']:
            events = engine.world(pre['test_x'], pre['test_y'], pre['test_timestamp'], pre, trial['seed'], engine.BASE)
            for mode, result in trial['results'].items():
                cfg = task['selected'][mode]
                physical = explicit_service(events, pre, result['rows'], cfg)
                differences = {k: abs(float(physical[k]) - float(result[k]))
                               for k in ('gross', 'fees', 'net', 'drops', 'accuracy', 'deployments')}
                fork_error = target_error = covered_lower_error = 0.
                for row in result['rows']:
                    fork = explicit_fork(events, row, cfg)
                    fork_error = max(fork_error, abs(fork['actual'] - row['local_net']), abs(fork['gross'] - row['truegross']))
                    target_error = max(target_error, fork['identity_error'], abs(fork['standardized'] - row['standardized_score']))
                    covered_lower_error = max(covered_lower_error, fork['covered_actual_lower_error'])
                    if not -1e-8 <= fork['slack'] <= 2 + 1e-8:
                        raise AssertionError('Invalid interruption slack')
                for callback in result['q_updates']:
                    if not callback['issued_index'] < callback['maturity'] <= callback['update_index']:
                        raise AssertionError('Premature label callback')
                    violation = int(callback['standardized_score'] > callback['issued_q'])
                    expected = max(0., callback['q_before'] + .05 * (violation - .1))
                    if violation != callback['violation'] or abs(expected - callback['q_after']) > 1e-12:
                        raise AssertionError('Issued-threshold or projection mismatch')
                identity = sum(v['violation'] for v in result['q_updates']) - (
                    .1 * len(result['q_updates']) + (
                        result['q_final'] - result['q_initial'] - sum(v['regulator'] for v in result['q_updates'])) / .05)
                passed = max(differences.values()) < 1e-8 and fork_error < 1e-8 and target_error < 1e-8 and covered_lower_error < 1e-8 and abs(identity) < 1e-8
                checks.append(dict(dataset=name, seed=trial['seed'], mode=mode, passed=bool(passed),
                    service_errors=differences, fork_error=fork_error, target_identity_error=target_error,
                    covered_lower_error=covered_lower_error, calibration_identity_error=identity,
                    forks=len(result['rows']), callbacks=len(result['q_updates'])))
                if not passed:
                    raise AssertionError('Independent reconstruction differs: ' + name + '/' + mode)
    out = dict(kind='Independent request-level reconstruction of retained supplementary actions',
        passed=all(c['passed'] for c in checks), policy_trajectories=len(checks),
        method_fork_checks=sum(c['forks'] for c in checks),
        max_service_error=max(max(c['service_errors'].values()) for c in checks),
        max_fork_error=max(c['fork_error'] for c in checks),
        max_calibration_identity_error=max(abs(c['calibration_identity_error']) for c in checks),
        input_sha256=hashlib.sha256(args.results.read_bytes()).hexdigest(), checks=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
