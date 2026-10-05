"""Read-only source closure and constructed-state audit of final original CJ.

Never opens a physical archive, processed dataset, calibration result or test
result. Its only writes are this audit's report in the theory directory.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import types
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
PHYSICAL = HERE.parent / 'physical'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def closure(objects):
    queue, seen, files = list(objects), set(), set()
    while queue:
        obj = queue.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, types.ModuleType):
            file = getattr(obj, '__file__', None)
            if not file:
                continue
            path = Path(file).resolve()
            if not path.is_relative_to(PROJECT):
                continue
            files.add(path)
            for value in vars(obj).values():
                if isinstance(value, (types.ModuleType, types.FunctionType)):
                    queue.append(value)
        elif isinstance(obj, types.FunctionType):
            path = Path(obj.__code__.co_filename).resolve()
            if path.is_relative_to(PROJECT):
                files.add(path)
                for value in obj.__globals__.values():
                    if isinstance(value, types.ModuleType):
                        queue.append(value)
    return files


def states(r):
    api = r.r
    _, _, _, _, original_cfg, _, _, _ = api.binding()
    cfg = dict(r.base_cfg(original_cfg), slice_bandwidth=.5, q_floor=.64)
    pre = dict(m=3, classes=2, q=np.array([1., 1.2, .8]))
    x = np.tile([1., 0.], (8, 1))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    future = [dict(x=x.copy(), y=y.copy(), keep=np.ones(8, dtype=bool)) for _ in range(4)]
    residuals = np.array([[.14, -.12, .06], [-.11, .1, -.13], [.17, .08, .12],
                          [-.15, -.09, .01], [.07, .18, -.16], [.12, -.13, -.07]])
    masks = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=bool)
    records = []
    for j, z in enumerate(residuals):
        age = 4 * (j + 1)
        probabilities = np.repeat(np.stack(((1+z)/2, (1-z)/2), axis=1)[:, None, :], 8, axis=1)
        records.append(dict(k=100-age, maturity=104-age, context=np.zeros(1),
                            x=x.copy(), p=probabilities, mask=masks[j], future=future, N=32))
    issues, checked, exact_diag_error, poison_error, marginal_error = [], 0, 0., 0., 0.
    for ids in (np.array([0]), np.array([0, 2]), np.array([0, 1, 2])):
        current = dict(k=100, context=np.zeros(1), candidate=np.eye(2),
                       reference=np.array([[0., 1.], [0., 1.]]), ids=ids,
                       h=np.array([.13, -.02, .07])[ids])
        for kind, archive in (('empty', []), ('complete', [dict(z, mask=np.ones(3, bool)) for z in records]), ('mixed', records)):
            mm = api.core.state(archive, current, pre, cfg)
            missing = sorted(set(api.MATCHED_ARMS)-set(mm['conditional']))
            assert not missing, (kind, ids.tolist(), missing)
            alpha, loc, V, _, _, _ = api.c.law(mm, pre, ids, cfg)
            diagonal_V = V-mm['P']+np.diag(np.diag(mm['P']))
            expected = api.c.joint_posterior(current['h'], alpha, loc, diagonal_V)
            exact_diag_error = max(exact_diag_error, *(abs(expected[key]-mm['conditional']['conditional_joint_diagP'][key]) for key in ('mean', 'variance')))
            marginal_error = max(marginal_error, float(np.max(abs(np.diagonal(V, axis1=1, axis2=2)-np.diagonal(diagonal_V, axis1=1, axis2=2)))))
            d = dict(temporal=mm, ids=ids, h=current['h'], N=128, truegross=7.)
            for arm in api.ARMS:
                answer = api.forecast(d, pre, cfg, arm, 1.2)
                poisoned = api.forecast(dict(d, truegross=-1e50), pre, cfg, arm, 1.2)
                assert answer[1]['converged'], (arm, answer[1])
                assert np.isfinite(answer[2:5]).all()
                poison_error = max(poison_error, abs(answer[2]-poisoned[2]), abs(answer[3]-poisoned[3]))
                assert answer[5]['information_ready'] == bool(mm['eligible'])
                checked += 1
    assert exact_diag_error < 1e-13 and marginal_error < 1e-13 and poison_error == 0
    return dict(passed=True, constructed_states=9, active_dimensions=[1, 2, 3],
                archive_cases=['empty', 'complete', 'mixed'], checked_arm_calls=checked,
                all_seven_matched_keys_present=True, all_nine_arm_forecasts_callable=True,
                old_diagP_formula_error=exact_diag_error, exact_marginal_variance_error=marginal_error,
                future_truth_F_S_poison_error=poison_error, all_ready_flags_match_mature_eligibility=True)


def main():
    r = module('final_original_independent_physical', PHYSICAL/'run_physical.py')
    frozen = r.verify()
    api_checks = states(r)
    ext = module('final_original_independent_external', PHYSICAL/'run_external.py')
    external = ext.verify()
    frozen_files = {Path(z).resolve() for z in frozen['source_hashes']}
    reachable = closure([ext, *ext.p.r.binding()])
    uncovered = sorted(reachable-frozen_files)
    inactive_adapter = PROJECT/'outputs/Fusion_Recovery_Repro/extension/new_data_adapter.py'
    assert set(uncovered) <= {inactive_adapter}, uncovered
    assert frozen['primary'] == r.PRIMARY == 'conditional_joint'
    assert frozen['arms'] == list(r.ARMS) and len(r.ARMS) == 9 and len(r.r.MATCHED_ARMS) == 7
    assert r.r.NEW_ARMS == ()
    protocol = r.load(PHYSICAL/'protocol.json')
    assert protocol['status'] == 'frozen_before_acquisition'
    report = dict(passed=True, scope='metadata/source imports and constructed states only; no physical data or outcomes opened',
        algorithm_freeze_sha256=sha(PHYSICAL/'algorithm_freeze.json'), method_sha256=sha(r.METHOD_PATH),
        primary=r.PRIMARY, arms=list(r.ARMS), frozen_source_count=len(frozen_files),
        all_frozen_hashes_and_acquisition_authorization_verified=True,
        external_source_hashes_verified=True, external_rules=external['rules'], api=api_checks,
        imported_local_python_count=len(reachable),
        uncovered_imported_local_sources=[str(z.relative_to(PROJECT)) for z in uncovered],
        uncovered_scope='old load_dataset/make_prefix adapter imported by generic helpers; final HARTH runner uses its separately frozen physical adapter, so old adapter is inactive for data preparation',
        scientific_limit='source consistency and finite API checks do not imply posterior calibration or superior physical return')
    path = HERE/'final_original_api_report.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
