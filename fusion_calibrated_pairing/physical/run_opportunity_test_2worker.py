"""Resource-only test executor; all frozen V2 numerical functions unchanged.

Two fork workers share read-only prepared arrays through copy-on-write.
Every world is independent by protocol. Results merge in the original
recording/seed/arm order and a serial-versus-worker first-world check is exact.
"""
from pathlib import Path
import concurrent.futures, datetime, hashlib, importlib.util, json, multiprocessing, os, sys, time

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sp = importlib.util.spec_from_file_location('unchanged_opportunity_v2', ROOT / 'run_opportunity_v2.py')
r = importlib.util.module_from_spec(sp); sp.loader.exec_module(r)
GLOBAL = None

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False,
        default=lambda x: x.item() if isinstance(x, r.np.generic) else x.tolist()).encode()

def solve(task):
    index, recording_index, seed = task
    engine, guard, cfg, service, fork, pre, meta, cache, models, selection = GLOBAL
    record = pre['recording_streams'][recording_index]
    selected = selection['selected']; library = selection['prior_library']; pools = selection['score_pools']
    original, base, pdf, qmf = r.world(record, seed, pre, cfg, engine, models)
    streams, raw = r.raw_pipelines(record, base, pdf, qmf, pre, cfg, library)
    availability = dict(recording=record['recording'], person=record['person'], seed=seed,
        empty_forecast_windows=sum(event['empty_forecast_mask'] for event in pdf),
        physical_all_absent_windows=sum(event['physical_all_absent'] for event in pdf),
        physically_absent_by_source=[sum(not event['physical_available'][s] for event in pdf) for s in range(pre['m'])])
    issued = []; guarded = []; budget = []; checks = []; raw_log = []
    for arm in r.ARMS:
        result = r.cal.calibrated_run(streams[arm], raw[arm]['rows'], selected[arm]['config'], pools[arm], False)
        g = r.api.db.guarded(original, pre, cfg, result, guard, service, 130.)
        identity = dict(arm=arm, recording=record['recording'], session=record['session'], person=record['person'], seed=seed)
        issued.append(dict(**identity, **result)); guarded.append(dict(**identity, **g)); raw_log.append(dict(**identity, **raw[arm]))
        checks.append(dict(**identity, **r.target_check(original, result['rows'], cfg, fork)))
        for B in r.BUDGETS:
            budget.append(dict(**identity, budget=B, **(g if B == 130. else r.api.db.guarded(original, pre, cfg, result, guard, service, B))))
    value = dict(issued=issued, guarded=guarded, budget_rows=budget, raw=raw_log,
                 availability=[availability], checks=checks)
    return index, value

def main():
    global GLOBAL
    for name in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
        assert os.environ.get(name) == '1', (name, os.environ.get(name))
    frozen = r.verify(); assert not (r.OUT / 'results.json.gz').exists()
    selection = r.load('opportunity/selection.json')
    assert r.api.sha(r.OUT / 'selection.json') == r.load('opportunity/selection_lock.json')['selection_sha256']
    manifest = dict(utc=r.now(), category='Post-acquisition resource-only execution supplement, no scientific method change',
        executor_sha256=r.api.sha(Path(__file__)), unchanged_runner_sha256=r.api.sha(ROOT / 'run_opportunity_v2.py'),
        original_source_count=frozen['source_count'], original_source_hashes=frozen['algorithm_source_hashes'],
        freeze_sha256=r.api.sha(ROOT / 'opportunity_freeze.json'), repair_sha256=r.api.sha(ROOT / 'opportunity_metadata_repair_freeze.json'),
        selection_sha256=r.api.sha(r.OUT / 'selection.json'), workers=2, start_method='fork', blas_threads=1,
        merge_order='Original recording order, then five declared seeds, then original nine-arm order',
        unchanged_functions=['prepared', 'world', 'raw_pipelines', 'calibrated_run', 'guarded', 'target_check'],
        all_sessions=12, all_seeds=list(r.TEST_SEEDS), all_arms=list(r.ARMS), all_budgets=list(r.BUDGETS),
        independent_worlds=True, test_based_selection_or_method_change=False)
    r.dump('opportunity/test_execution_2worker_preflight.json', manifest)
    engine, guard, cfg, service, fork, pre, meta, cache, models, training = r.prepared(selection)
    GLOBAL = (engine, guard, cfg, service, fork, pre, meta, cache, models, selection)
    tasks = [(n, i, seed) for n, (i, seed) in enumerate((i, seed)
        for i, record in enumerate(pre['recording_streams']) if record['partition'] == 'test' for seed in r.TEST_SEEDS)]
    assert len(tasks) == 60
    chunks = r.OUT / 'test_worlds_2worker'; chunks.mkdir(exist_ok=False)
    start = time.monotonic(); serial_index, reference = solve(tasks[0])
    reference_hash = hashlib.sha256(canonical(reference)).hexdigest()
    r.dump('opportunity/test_serial_reference.json.gz', reference)
    print('SERIAL_PREFLIGHT', tasks[0], round(time.monotonic() - start, 3), reference_hash, flush=True)
    context = multiprocessing.get_context('fork')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        worker_index, first = pool.submit(solve, tasks[0]).result()
        worker_hash = hashlib.sha256(canonical(first)).hexdigest()
        assert worker_index == serial_index and worker_hash == reference_hash, 'Numerical parallel invariance failed'
        r.verify()
        r.dump('opportunity/test_execution_2worker_invariance.json', dict(utc=r.now(), serial_sha256=reference_hash,
            worker_sha256=worker_hash, all_issue_score_action_target_budget_fields_bit_identical=True,
            scientific_sources_unchanged=True, world_recording=first['availability'][0]['recording'], seed=tasks[0][2]))
        r.api.dump(chunks / '000.json.gz', first)
        print('PARALLEL_INVARIANCE_PASS', flush=True)
        futures = {pool.submit(solve, task): task for task in tasks[1:]}
        for future in concurrent.futures.as_completed(futures):
            index, value = future.result()
            r.api.dump(chunks / f'{index:03d}.json.gz', value)
            print('TEST', value['availability'][0]['recording'], value['availability'][0]['seed'], 'index', index, flush=True)
    output = {key: [] for key in ('issued', 'guarded', 'budget_rows', 'raw', 'availability')}; checks = []
    for index, _, _ in tasks:
        value = r.api.load(chunks / f'{index:03d}.json.gz')
        for key in output: output[key].extend(value[key])
        checks.extend(value['checks'])
    output['selected'] = selection['selected']
    r.dump('opportunity/results.json.gz', output)
    r.dump('opportunity/test_checks.json', dict(target_checks=checks,
        max_service_error=max(g['service_error'] for g in output['budget_rows']),
        cache=dict(scope='Resource-only two fork workers; exact first-world score/action/budget equality verified',
                   parent_preparation_cache=cache.report())))
    r.verify()
    manifest.update(completed_utc=r.now(), elapsed_seconds=time.monotonic() - start,
        numerical_first_world_bit_identical=True, source_closure_unchanged_after=True,
        results_sha256=r.api.sha(r.OUT / 'results.json.gz'), worlds=60,
        policy_trajectories=len(output['guarded']), budget_paths=len(output['budget_rows']))
    r.dump('opportunity/test_execution_2worker_completed.json', manifest)
    print('COMPLETE', len(output['guarded']), 'policies', len(output['budget_rows']), 'ledger paths', flush=True)

if __name__ == '__main__':
    main()
