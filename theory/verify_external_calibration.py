"""Read-only validation of completed calibration artifacts; no held-out data."""
from pathlib import Path
import datetime
import gzip
import hashlib
import importlib.util
import json
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1] / 'physical'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt') as source:
            return json.load(source)
    return json.loads(Path(path).read_text())


def verify():
    spec = importlib.util.spec_from_file_location('independent_calibration_audit', ROOT / 'run_external.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    protocol = runner.verify()
    runner.p.verify()
    selection = load(ROOT / 'external/selection.json')
    lock = load(ROOT / 'external/selection_freeze.json')
    checks = {
        'selection_sha256': ROOT / 'external/selection.json',
        'protocol_sha256': ROOT / 'external/protocol.json',
        'cache_sha256': ROOT / 'data/cached_dataset.npz',
        'algorithm_freeze_sha256': ROOT / 'algorithm_freeze.json',
        'cache_metadata_sha256': ROOT / 'data/cache_metadata.json',
    }
    for name, path in checks.items():
        assert lock[name] == sha(path), name
    assert selection['data_hashes'] == load(ROOT / 'data/cache_metadata.json')['consumed_raw_hashes']
    expected = {(rule, rate, margin) for rule in runner.RULES for rate in runner.RATES for margin in runner.MARGINS}
    actual = [(g['rule'], g['lr'], g['margin']) for g in selection['grid']]
    assert set(actual) == expected and len(actual) == len(expected) == 18
    assert len(selection['models']) == len(selection['training']) == 6
    for item in selection['training']:
        assert item['steps'] == 500
        assert item['objective_strength'] == (1. if item['rule'] == 'pdf' else .1)
    for item in selection['grid']:
        scores = item['fit_scores']
        initial = max(0., float(np.quantile(scores, .9, method='higher'))) if scores else 0.
        assert initial == item['q0']
    for rule in runner.RULES:
        winner = max((g for g in selection['grid'] if g['rule'] == rule),
                     key=lambda g: (g['guarded_net'], -g['margin'], -g['lr']))
        assert winner == selection['selected'][rule]

    trials = load(ROOT / 'external/calibration_trials.json.gz')
    assert len(trials) == 18
    counters = dict(trajectories=0, issued_rows=0, callbacks=0, rejected_callbacks=0)
    maxima = dict(q_update=0., q_identity=0., service=0., target_score=0.)
    common_chains = None
    for trial in trials:
        config = trial['configuration']
        assert config in selection['grid']
        chain_keys = [(c['recording'], c['person'], c['seed']) for c in trial['chains']]
        assert len(chain_keys) == len(set(chain_keys)) == 134 * len(runner.p.CAL_SEEDS)
        if common_chains is None:
            common_chains = chain_keys
        assert chain_keys == common_chains
        assert {c['person'] for c in trial['chains']} == {'S016', 'S017', 'S018', 'S019'}
        net = sum(c['guarded']['increment'] for c in trial['chains']) / len(runner.p.CAL_SEEDS)
        assert abs(net - config['guarded_net']) < 1e-10
        for chain in trial['chains']:
            result, guarded = chain['result'], chain['guarded']
            rows = {r['k']: r for r in result['rows']}
            executed = {r['k']: r for r in guarded['rows']}
            assert rows.keys() == executed.keys()
            counters['trajectories'] += 1
            counters['issued_rows'] += len(rows)
            q = result['q_initial']
            assert q == config['q0']
            violation_sum = regulator_sum = 0.
            for update in result['q_updates']:
                origin = rows[update['issued_index']]
                assert origin['k'] < update['maturity'] <= update['update_index']
                assert update['maturity'] == origin['maturity']
                assert update['issued_q'] == origin['q_issued']
                assert update['issued_std'] == origin['posterior_or_block_sd']
                assert update['standardized_score'] == origin['standardized_score']
                assert update['violation'] == int(origin['standardized_score'] > origin['q_issued'])
                proposal = q + .05 * (update['violation'] - .1)
                after = max(0., proposal)
                error = max(abs(q - update['q_before']), abs(after - update['q_after']),
                            abs(after - proposal - update['regulator']))
                maxima['q_update'] = max(maxima['q_update'], error)
                assert error < 1e-12
                q = after
                violation_sum += update['violation']
                regulator_sum += update['regulator']
                counters['callbacks'] += 1
                counters['rejected_callbacks'] += not executed[origin['k']]['action']
            assert q == result['q_final'] and len(result['q_updates']) == result['calibration_updates']
            identity = violation_sum - (.1 * len(result['q_updates']) +
                                        (q - result['q_initial'] - regulator_sum) / .05)
            maxima['q_identity'] = max(maxima['q_identity'], abs(identity))
            assert abs(identity) < 1e-10
            assert abs(identity - result['calibration_identity_error']) < 1e-12
            for row in rows.values():
                target_score = (row['gain'] - row['truegross']) / row['posterior_or_block_sd']
                maxima['target_score'] = max(maxima['target_score'], abs(target_score - row['standardized_score']))
                assert row['lower_covered'] == (row['standardized_score'] <= row['q_issued'])
            actions = [r for r in executed.values() if r['action']]
            assert len(actions) == guarded['admissions']
            assert sum(r['local_net'] > 0 for r in actions) == guarded['beneficial']
            assert sum(r['local_net'] < 0 for r in actions) == guarded['harmful']
            assert sum(r['local_net'] == 0 for r in actions) == guarded['zero']
            assert abs(sum(r['local_net'] for r in actions) - guarded['increment']) < 1e-8
            assert guarded['loss'] <= 130. + 1e-8 and guarded['minimum_prefix'] >= -130. - 1e-8
            assert all(r['spent_loss'] + r['reserved'] <= 130. + 1e-8 for r in guarded['ledger'])
            maxima['service'] = max(maxima['service'], guarded['service_error'])
    assert maxima['target_score'] < 1e-10 and maxima['service'] < 1e-8
    cache = load(ROOT / 'execution_cache_calibration_audit.json')
    assert cache['status'] == 'completed' and cache['before']['optimization_only']
    assert cache['before']['algorithm_freeze_sha256'] == sha(ROOT / 'algorithm_freeze.json')
    assert cache['before']['wrapper_sha256'] == sha(ROOT / 'execution_cache.py')
    report = dict(status='passed', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  frozen_sources_unchanged=True, source_count=28, selections_locked=True,
                  configurations=18, fitted_models=6, chains_per_configuration=len(common_chains),
                  counts=counters, maximum_errors=maxima, selected=selection['selected'],
                  cache_counts={k: cache['cache'][k] for k in ('calls', 'hits', 'misses', 'bypasses', 'evictions')},
                  calibration_callback_scope=protocol['calibration'],
                  held_out_data_or_results_read=False, input_sha256={str(p): sha(p) for p in checks.values()})
    (ROOT / 'external_calibration_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'configurations', 'chains_per_configuration',
                                           'counts', 'maximum_errors', 'cache_counts')}, indent=2))


if __name__ == '__main__':
    verify()
