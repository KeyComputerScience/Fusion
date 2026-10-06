"""Read-only request-level reconstruction of the saved 68→132 chain."""
from pathlib import Path
import importlib.util, json
import numpy as np

ROOT = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location('unchanged_opportunity_reconstruction', ROOT / 'run_opportunity_v2.py')
r = importlib.util.module_from_spec(sp); sp.loader.exec_module(r)

def main():
    r.verify(); selection = r.load('opportunity/selection.json')
    assert r.api.sha(r.OUT / 'selection.json') == r.load('opportunity/selection_lock.json')['selection_sha256']
    engine, guard, cfg, service, fork, pre, metadata, cache, models, training = r.prepared(selection)
    record = next(x for x in pre['recording_streams'] if x['recording'] == 'S4-Drill')
    events = engine.world(record['x'], record['y'], record['timestamp'], pre, 162003, cfg)
    chunk = r.api.load(r.OUT / 'test_worlds_2worker/032.json.gz')
    rows = {arm: next(g for g in chunk['guarded'] if g['arm'] == arm)['rows']
            for arm in ('paired', 'conditional_moment')}
    output = []
    for origin in (68, 132):
        saved = next(x for x in rows['paired'] if x['k'] == origin)
        candidate = events[origin]['candidate']; reference = events[origin]['reference']
        requests = []
        for k in range(origin, origin + cfg['horizon']):
            event = events[k]
            assert np.array_equal(event['reference'], reference)
            pc = np.argmax(event['x'] @ candidate, axis=1)
            pr = np.argmax(event['x'] @ reference, axis=1)
            common = (cfg['probe_drop'] if event['probe'] else 0) + (2 if event['refresh'] else 0)
            for j in range(len(pr)):
                requests.append(dict(window=k, window_position=j,
                    acquisition_origin=record['timestamp'][k * cfg['window'] + j],
                    truth_class=int(event['y'][j]), candidate_class=int(pc[j]), reference_class=int(pr[j]),
                    candidate_correct=bool(pc[j] == event['y'][j]), reference_correct=bool(pr[j] == event['y'][j]),
                    common_served=j >= common,
                    extra_installation_interruption=bool(k == origin and common <= j < common + cfg['deploy_drop'])))
        keep = [x for x in requests if x['common_served']]
        candidate_correct = sum(x['candidate_correct'] for x in keep)
        reference_correct = sum(x['reference_correct'] for x in keep)
        extra = sum(x['candidate_correct'] for x in keep if x['extra_installation_interruption'])
        gross = candidate_correct - reference_correct
        served_contrast = candidate_correct - extra - reference_correct
        paid = served_contrast - 3.
        assert gross == saved['truegross'] and paid == saved['local_net']
        output.append(dict(origin=origin, total_requests=len(requests), common_served_requests=len(keep),
            candidate_correct_before_installation=candidate_correct, reference_correct=reference_correct,
            candidate_correct_lost_to_installation=extra, served_correctness_contrast=served_contrast,
            gross_correctness_contrast=gross, installation_fee=2., restoration_fee=1., complete_paid_increment=paid,
            saved_issued_target_matches=True, request_records=requests))
    value = dict(recording='S4-Drill', seed=162003, source_closure_unchanged=r.verify()['source_count'],
        cache_metadata_sha256=r.api.sha(ROOT / 'opportunity_data/cache_metadata.json'),
        selection_sha256=r.api.sha(r.OUT / 'selection.json'),
        results_sha256=r.api.sha(r.OUT / 'results.json.gz'), forks=output,
        scope='Independent raw-feature/label and frozen service-model reconstruction; no fusion scores refit and no additional policy experiment.')
    r.dump('opportunity_mechanism_request_reconstruction.json', value)
    for row in output: print({k: v for k, v in row.items() if k != 'request_records'}, flush=True)

if __name__ == '__main__':
    main()
