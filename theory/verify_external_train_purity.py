"""Constructed external-training purity checks; no HARTH inputs or outcomes."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SOURCE = PROJECT/'work/fusion_strengthening_20261003/external/neural_external.py'


def same_models(a, b):
    return len(a) == len(b) and all(list(x) == list(y) and
        all(np.array_equal(x[key], y[key]) for key in x) for x, y in zip(a, b))


def same_rng(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def main():
    spec = importlib.util.spec_from_file_location('independent_external_train_purity', SOURCE)
    nn = importlib.util.module_from_spec(spec); spec.loader.exec_module(nn)
    rng = np.random.default_rng(202610041)
    xs = [np.c_[rng.normal(size=(17, 3)), np.ones(17)] for _ in range(2)]
    y = np.arange(17, dtype=np.int64) % 3
    xs0, y0 = [x.copy() for x in xs], y.copy()
    for x in xs:
        x.flags.writeable = False
    y.flags.writeable = False
    base, _ = nn.train(xs, y, 'qmf', .02, 1, 3)
    cfg = nn.helper.CFG
    checks = 0
    for rule in ('pdf', 'qmf'):
        for provided in (False, True):
            models = copy.deepcopy(base) if provided else None
            models0 = copy.deepcopy(models)
            if models is not None:
                for model in models:
                    for value in model.values():
                        value.flags.writeable = False
            global_before = np.random.get_state()
            first, first_work = nn.train(xs, y, rule, .08, 20, 3, models)
            second, second_work = nn.train(xs, y, rule, .08, 20, 3, models)
            assert same_models(first, second) and first_work == second_work
            assert same_rng(global_before, np.random.get_state())
            assert all(np.array_equal(x, old) for x, old in zip(xs, xs0))
            assert np.array_equal(y, y0)
            if provided:
                assert same_models(models, models0)
                assert all(not np.shares_memory(first[s][key], models[s][key])
                           for s in range(len(models)) for key in models[s])
            nn.helper.CFG = dict(cfg, lr=-999., l2=1e4, train_buffer=1, probe_steps=1)
            changed_cfg, changed_work = nn.train(xs, y, rule, .08, 20, 3, models)
            nn.helper.CFG = cfg
            assert same_models(second, changed_cfg) and second_work == changed_work
            first[0]['W'][0, 0] = 999.
            first_work['first_loss'] = -999.
            assert second[0]['W'][0, 0] != 999. and second_work['first_loss'] != -999.
            checks += 1
    report = dict(passed=True, scope='constructed arrays only; no HARTH data/outcomes opened',
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), cases=checks,
        rules=['pdf', 'qmf'], steps=20, supplied_models_and_seeded_initialization_checked=True,
        exact_repeated_numeric_and_work_outputs=True, global_numpy_rng_unchanged=True,
        read_only_inputs_labels_and_supplied_models_unmodified=True, no_return_model_aliasing_to_supplied_models=True,
        helper_CFG_changes_do_not_change_explicit_train_result=True,
        caveats=['key full ordered inputs, labels and model arrays; QMF ranking depends on row order',
            'key dtype, shape and memory layout as well as rule/rate/steps/classes and None-model state',
            'deep-copy model arrays and work on every cache return',
            'keep frozen function identities/runtime/BLAS settings; warnings/wall time are outside numeric purity'])
    path = HERE/'external_train_purity_report.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
