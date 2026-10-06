"""Known-selfBACK V2 calibration repair on immutable saved fusion scores.

The paired density, targets, strong views, and all V1 results are untouched.
No source neural training or density fitting is performed here.  The
unchanged shared linear service world is regenerated for request accounting.
This is development, not another untouched physical confirmation.
"""
from pathlib import Path
import argparse
import datetime
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / 'design'
RISK = ROOT.parent / 'risk'
sys.path.insert(0, str(RISK))
import immutable_conditional_calibration_v2 as cal


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


v1 = module('known_selfback_v1_development', V1 / 'run_selfback_development.py')
old, api, np = v1.old, v1.api, v1.np
ARMS = v1.ARMS
CAL_SEEDS, DEV_SEEDS = v1.CAL_SEEDS, v1.DEV_SEEDS


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(name, value):
    api.dump(ROOT / name, value)


def load(name):
    return api.load(ROOT / name)


def raw_rows(chain):
    """Recover the original issued raw score, not the V1 calibrated score."""
    out = []
    for row in chain['rows']:
        r = {key:value for key,value in row.items()
             if not key.startswith('calibration_') and key != 'risk_certificate'}
        score = float(row['raw_gate_score'])
        r.update(gate_score=score, raw_gate_score=score, gain=score+5.,
                 q_issued=0., posterior_or_block_sd=1.,
                 lower_covered=bool(row['truegross']-5. >= score),
                 standardized_score=score-(row['truegross']-5.))
        r['action'] = bool(r['information_ready'] and score > 0)
        out.append(r)
    return out


def stream_for(rows, windows, recording):
    return dict(windows=windows, recording=recording, decisions=[
        dict(k=r['k'], maturity=r['maturity'], N=r['N'],
             cal_context=r['cal_context']) for r in rows])


def prepared():
    engine, guard, cfg, service, fork, pre, meta, cache = old.prepare()
    selection = api.load(V1 / 'selection.json')
    pre = dict(pre, q=np.asarray(selection['quality']))
    return engine, guard, cfg, service, fork, pre, meta, cache


def worlds(pre, engine, cfg, ids, seeds):
    records = [r for r in pre['recording_streams'] if r['person'] in ids]
    return {(r['recording'], seed): (r, engine.world(r['x'], r['y'],
                r['timestamp'], pre, seed, cfg)) for r in records for seed in seeds}


def freeze():
    assert not (ROOT/'protocol.json').exists()
    v1.verify()
    sources = {Path(__file__), Path(cal.__file__)}
    for mod in list(sys.modules.values()):
        filename = getattr(mod, '__file__', None)
        if filename and str(filename).startswith(str(v1.PROJECT)) and str(filename).endswith('.py'):
            sources.add(Path(filename))
    inputs = [V1/'protocol.json', V1/'selection.json', V1/'calibration_trials.json.gz',
              V1/'development_results.json.gz', V1/'development_summary.json']
    dump('protocol.json', dict(utc=now(), version=2,
         status='KNOWN selfBACK development calibration repair; former physical outcomes already inspected; no independent confirmation',
         source_hashes={str(p): api.sha(p) for p in sorted(sources)},
         immutable_input_hashes={str(p): api.sha(p) for p in inputs},
         raw_scores='exact V1 raw_gate_score; no paired-law, conditional-moment, forecaster, quality, or target refitting',
         service='unchanged exogenous linear candidate/reference world regenerated solely for independent request-level service reconstruction; no neural training',
         calibration='V2 same previsible quiet/informative stratum; persistent whole canonical score-fit pool; global latest48 mature callbacks then stratum filtering with true global callback age; empty stratum physical minimum',
         grid={arm: cal.configurations() for arm in ARMS}, configurations_per_arm=9,
         selection='same dedicated two selection participants and three delays; maximize complete guarded net; ties higher quantile probability then larger bandwidth',
         partitions=api.load(V1/'selection.json')['partitions'],
         calibration_seeds=CAL_SEEDS, development_seeds=DEV_SEEDS,
         budgets=(0.,110.,130.,260.), primary_budget=130., reserve=130.,
         causal_scope='models use individually arrived labels; complete aggregate calibration/state callbacks require full maturity. Target-poison checks fix issued forecaster events, not labels legitimately already received by the shared predictor.'))
    dump('pre_run_freeze.json', dict(utc=now(), protocol_sha256=api.sha(ROOT/'protocol.json')))
    print('FROZEN_V2', api.sha(ROOT/'protocol.json'), flush=True)


def verify():
    p = load('protocol.json')
    assert api.sha(ROOT/'protocol.json') == load('pre_run_freeze.json')['protocol_sha256']
    for key in ('source_hashes', 'immutable_input_hashes'):
        for path, want in p[key].items():
            assert api.sha(path) == want, path
    return p


def calibrate():
    verify()
    assert not (ROOT/'selection.json').exists()
    selection = api.load(V1/'selection.json')
    pools = selection['score_pools']
    all_trials = api.load(V1/'calibration_trials.json.gz')
    # Each old trial shares identical raw laws. Keep exactly one copy of
    # every origin/arm/recording/seed; a tuning trial is not a new sample.
    stored = {}
    for trial in all_trials:
        for chain in trial['issued']:
            key = (chain['arm'], chain['recording'], chain['seed'])
            rows = raw_rows(chain)
            if key in stored:
                assert stored[key] == rows
            else:
                stored[key] = rows
    engine, guard, cfg, service, fork, pre, meta, cache = prepared()
    evs = worlds(pre, engine, cfg, selection['partitions']['selection'], CAL_SEEDS)
    trials = []
    grid = []
    poison_checks = []
    for arm in ARMS:
        for conf in cal.configurations():
            issued, guarded = [], []
            for recording, seed in sorted(evs):
                rec, events = evs[recording, seed]
                rows = stored[arm, recording, seed]
                stream = stream_for(rows, len(events), recording)
                r = cal.calibrated_run(stream, rows, conf, pools[arm], True)
                g = api.db.guarded(events, pre, cfg, r, guard, service, 130.)
                identity = dict(arm=arm, recording=recording, person=rec['person'], seed=seed)
                issued.append(dict(**identity, **r))
                guarded.append(dict(**identity, **g))
                cutoff = min(row['maturity'] for row in rows)-1
                poisoned = cal.calibrated_run(stream, rows, conf, pools[arm], True, poison_after=cutoff)
                before = [(x['k'], x['gate_score'], x['action']) for x in r['rows'] if x['k']<=cutoff]
                after = [(x['k'], x['gate_score'], x['action']) for x in poisoned['rows'] if x['k']<=cutoff]
                assert before == after
                poison_checks.append(dict(**identity, config=conf, cutoff=cutoff,
                                          unchanged_issued=len(before)))
            entry = dict(arm=arm, config=conf, net=sum(g['increment'] for g in guarded)/3)
            grid.append(entry)
            trials.append(dict(configuration=entry, issued=issued, guarded=guarded))
            print('CAL_V2', arm, conf, entry['net'], flush=True)
    selected = {arm:max((x for x in grid if x['arm']==arm),
                        key=lambda x:(x['net'], x['config']['probability'],x['config']['bandwidth'])) for arm in ARMS}
    dump('calibration_trials.json.gz', trials)
    dump('selection.json', dict(selected=selected, grid=grid, score_pools=pools,
         partitions=selection['partitions'], quality=selection['quality'],
         previous_linear_quality=selection['previous_linear_quality'],metadata=selection['metadata']))
    dump('selection_lock.json', dict(utc=now(),selection_sha256=api.sha(ROOT/'selection.json'),results_exist=False))
    dump('selection_poison_checks.json', poison_checks)
    print('SELECT_V2', selected, flush=True)


def development():
    verify()
    assert not (ROOT/'development_results.json.gz').exists()
    assert api.sha(ROOT/'selection.json') == load('selection_lock.json')['selection_sha256']
    selection = load('selection.json')
    v1_payload = api.load(V1/'development_results.json.gz')
    stored = {(c['arm'],c['recording'],c['seed']):c['rows'] for c in v1_payload['raw']}
    engine, guard, cfg, service, fork, pre, meta, cache = prepared()
    evs = worlds(pre, engine, cfg, selection['partitions']['development'], DEV_SEEDS)
    issued, guarded, budget, checks, raws = [], [], [], [], []
    for recording, seed in sorted(evs):
        rec, events = evs[recording,seed]
        for arm in ARMS:
            rows = stored[arm,recording,seed]
            stream = stream_for(rows,len(events),recording)
            r = cal.calibrated_run(stream,rows,selection['selected'][arm]['config'],selection['score_pools'][arm])
            identity = dict(arm=arm,recording=recording,person=rec['person'],seed=seed)
            g = api.db.guarded(events,pre,cfg,r,guard,service,130.)
            issued.append(dict(**identity,**r));guarded.append(dict(**identity,**g))
            raws.append(dict(**identity,rows=rows))
            checks.append(dict(**identity,**v1.target_check(events,r['rows'],cfg,fork)))
            for cap in (0.,110.,130.,260.):
                budget.append(dict(**identity,budget=cap,**(g if cap==130. else api.db.guarded(events,pre,cfg,r,guard,service,cap))))
        print('DEV_V2',recording,seed,flush=True)
    summaries = {}
    for arm in ARMS:
        gg=[g for g in guarded if g['arm']==arm]
        report=cal.actual_admission_report([r for g in gg for r in g['rows']],[r for g in gg for r in g['ledger']])
        increments=[sum(g['increment'] for g in gg if g['seed']==seed) for seed in DEV_SEEDS]
        summaries[arm]=dict(report,mean_increment=float(np.mean(increments)),seed_increments=increments)
    dump('development_results.json.gz',dict(issued=issued,guarded=guarded,budget_rows=budget,raw=raws,selected=selection['selected']))
    dump('development_summary.json',dict(summary=summaries,selected=selection['selected'],target_checks=checks,
         max_service_error=max(g['service_error'] for g in budget),cache=cache.report()))
    print('SUMMARY_V2',summaries,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['freeze','verify','calibrate','development'])
    args=parser.parse_args()
    result=globals()[args.phase]()
    if args.phase=='verify':print('VERIFIED_V2')
