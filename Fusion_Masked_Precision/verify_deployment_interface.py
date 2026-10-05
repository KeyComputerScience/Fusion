"""Independent live-interface and real fixed-state mechanism validation."""
import copy, hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import temporal_fusion as core
from deployment_interface import decide

ROOT = Path(__file__).resolve().parent
SHA = '811f98531ad3e135ceafbc64152b41ffbf380878cb45a1f7106eb54364157ff0'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(ROOT/'temporal_fusion.py') == SHA
engine, recovery, guard, source, cfg0, study, service, fork = core.support.setup(core.support.DEFAULT_PACKAGE, core.support.DEFAULT_GUARD)
sys.path.insert(0, str(core.PROJECT/'work/fusion_focus_20261003/fresh'))
from new_data_adapter import load_dataset, make_prefix
report = dict(core_sha256=SHA, interface_sha256=sha(ROOT/'deployment_interface.py'),
              validation_sha256=sha(__file__), nonempty_saved_comparisons=0,
              future_truth_checks=0, malformed_checks=[], maximum_forecast_difference=0.0,
              scope='separate fail-closed deployment adapter; frozen primary algorithm is unchanged',
              package_sources_rebound_only=False)
interventions = json.loads((ROOT/'fixed_state_interventions.json').read_text())
case_lookup = {(c['task'],c['seed'],c['origin'],c['variant']): c for c in interventions['cases']}
verified_cases = []; fixtures = []
all_results = json.loads((ROOT/'final_development/results.json').read_text())
all_results.update(json.loads((ROOT/'rss_validation/results.json').read_text()))
ledger = dict(C=0.0, Q=0.0, B=1000000.0, M=130.0)
for name in ('gas224', 'rss348'):
    folder = ROOT/('rss_validation' if name=='rss348' else 'final_development')
    cfgbase = json.loads((folder/'protocol.json').read_text())['cfg']
    if name == 'rss348':
        data, derivation = load_dataset(ROOT/'physical/data', name)
        pre = make_prefix(data, derivation, cfgbase, engine)
    else:
        data, pre = study(name, source['studies'][name])
    rr = all_results[name]
    chosen = rr['selected']['precision_joint']
    cfg = dict(cfgbase, prior_sd=chosen['prior_sd'], q_floor=chosen['q_floor'], threshold=chosen['margin'])
    for trial in rr['trials']:
        events = engine.world(pre['test_x'], pre['test_y'], pre['test_timestamp'], pre, trial['seed'], cfg)
        stream = core.augment(events, pre, engine.precompute(events, pre, cfg), cfg)
        reconstructed = service(events, pre, trial['results']['precision_joint']['rows'], cfg)
        assert abs(reconstructed['net']-trial['results']['precision_joint']['net']) < 1e-8
        for row, d in zip(trial['results']['precision_joint']['rows'], stream['decisions']):
            q = row['q_issued']; full = core.forecast(d, pre, cfg, 'precision_joint', q)
            live = dict(d, source_probabilities=events[d['k']]['p'][d['ids']].copy())
            issued = decide(live, pre, cfg, q, service_feasible=True, ledger=ledger, core=core)
            assert issued['reason'] != 'invalid_input_or_numerical_failure', issued
            assert issued['reason'] != 'uncertified_solver', issued
            error = max(float(np.max(np.abs(np.asarray(issued['weights'])-full[0]))),
                        abs(issued['gain']-full[2]), abs(issued['predictive_scale']-full[3]),
                        abs(issued['score']-row['gate_score']))
            assert error < 1e-8 and issued['action'] == row['action']
            report['maximum_forecast_difference'] = max(report['maximum_forecast_difference'], error)
            report['nonempty_saved_comparisons'] += 1
            poisoned = dict(live, truegross=object(), localnet=object(), future=object(), future_labels=object())
            assert decide(poisoned, pre, cfg, q, service_feasible=True, ledger=ledger, core=core) == issued
            report['future_truth_checks'] += 1
            if len(fixtures) < 2 and len(d['ids']) > 1:
                fixtures.append((live, pre, cfg, q, issued))
            variants = dict(scalar_mass=core.forecast(d, pre, cfg, 'scalar_mass', q),
                            complete_only=core.forecast(d, pre, cfg, 'complete_only', q),
                            no_posterior=core.forecast(d, pre, cfg, 'no_posterior', q),
                            decoupled=core.forecast(d, pre, cfg, 'precision_joint', 0.0))
            diagonal = dict(d, temporal=dict(d['temporal'], P=np.diag(np.diag(d['temporal']['P']))))
            variants['P_diagonal_only'] = core.forecast(diagonal, pre, cfg, 'precision_joint', q)
            for variant, altered in variants.items():
                score = altered[2]-q*altered[3]-5.0
                action = bool(score > 0)
                key = (name, trial['seed'], row['k'], variant)
                if action != row['action']:
                    saved = case_lookup[key]
                    actual = fork(events, row, cfg)
                    assert abs(actual['actual']-saved['actual_lease_increment']) < 1e-8
                    assert abs(score-saved['variant_score']) < 1e-8
                    assert abs(issued['score']-saved['full_score']) < 1e-8
                    assert np.max(np.abs(full[0]-saved['full_weights'])) < 1e-8
                    assert np.max(np.abs(altered[0]-saved['variant_weights'])) < 1e-8
                    verified_cases.append(dict(task=name, seed=trial['seed'], origin=row['k'], variant=variant,
                                               full_action=issued['action'], variant_action=action,
                                               full_score=issued['score'], variant_score=score,
                                               actual_lease_increment=actual['actual'],
                                               one_step_full_gain=(int(row['action'])-int(action))*actual['actual'],
                                               full_weights=full[0].tolist(), variant_weights=altered[0].tolist(),
                                               issued_q=q, non_scalar=d['temporal']['non_scalar']))
                else:
                    assert key not in case_lookup

live, pre, cfg, q, baseline = fixtures[0]
def check(label, decision=live, prefix=pre, config=cfg, threshold=q, feasible=True, state=ledger, dependency=core):
    r = decide(decision, prefix, config, threshold, service_feasible=feasible, ledger=state, core=dependency)
    assert not r['action'], (label,r)
    report['malformed_checks'].append(dict(name=label, reason=r['reason']))
    return r

empty = dict(live, ids=np.array([],dtype=int))
assert check('empty_forecast_mask', decision=empty)['reason'] == 'empty_source_mask'
one = copy.deepcopy(live); one['ids'] = one['ids'][:1]; one['h'] = one['h'][:1]
one['source_probabilities'] = one['source_probabilities'][:1]
one['temporal']['mu'] = one['temporal']['mu'][:1]
one['temporal']['R'] = one['temporal']['R'][:1,:1]
one['temporal']['P'] = one['temporal']['P'][:1,:1]
singleton = decide(one, pre, cfg, q, service_feasible=True, ledger=ledger, core=core)
assert singleton['weights'] == [1.0]
report['singleton'] = singleton
for label, mutate in [
    ('nonfinite_h',lambda d: d['h'].__setitem__(0,np.nan)),
    ('forecast_dimension',lambda d: d.update(h=np.r_[d['h'],0.])),
    ('duplicate_source_id',lambda d: d['ids'].__setitem__(1,d['ids'][0])),
    ('invalid_source_id',lambda d: d['ids'].__setitem__(0,-1)),
    ('indefinite_R',lambda d: d['temporal']['R'].__setitem__((0,0),-1.)),
    ('nonsymmetric_P',lambda d: d['temporal']['P'].__setitem__((0,1),10.)),
    ('nonfinite_P',lambda d: d['temporal']['P'].__setitem__((0,0),np.inf)),
    ('model_dimension',lambda d: d.update(reference=d['reference'][:-1])),
    ('nonfinite_model',lambda d: d['candidate'].__setitem__((0,0),np.nan)),
    ('future_maturity',lambda d: d['temporal']['eligible'].append(dict(origin=0,maturity=d['k']+1,weight=.1,observed_components=1))),
    ('invalid_N',lambda d: d.update(N=129)),
    ('invalid_probabilities',lambda d: d['source_probabilities'].__setitem__((0,0,0),2.))]:
    changed = copy.deepcopy(live); mutate(changed)
    assert check(label,decision=changed)['reason'] == 'invalid_input_or_numerical_failure'
bad_pre = dict(pre,q=pre['q'].copy()); bad_pre['q'][0]=0.
check('zero_quality',prefix=bad_pre)
check('negative_threshold',threshold=-1)
check('nonfinite_threshold',threshold=np.nan)
check('service_infeasible',feasible=False)
check('reserve_exceeds_capacity',state=dict(C=0.,Q=0.,M=130.,B=110.))
check('invalid_ledger',state=dict(C=2.,Q=0.,M=130.,B=1.))
check('uncertified_small_reserve',state=dict(C=0.,Q=0.,M=0.,B=130.))
check('nonfinite_contract_dimension',config=dict(cfg,horizon=float('inf')))
check('revised_fee_contract',config=dict(cfg,deploy_fee=3.))
check('invalid_entropy_coefficient',config=dict(cfg,tau=0.))
def fail_solver(*args):
    out=list(core.forecast(*args));out[1]=dict(out[1],converged=False);return tuple(out)
assert check('injected_solver_failure',dependency=SimpleNamespace(forecast=fail_solver))['reason'] == 'uncertified_solver'

recomputed = {}
for name in ('gas224','rss348'):
    recomputed[name] = {}
    for variant in ('P_diagonal_only','scalar_mass','complete_only','no_posterior','decoupled'):
        c=[x for x in verified_cases if x['task']==name and x['variant']==variant]
        recomputed[name][variant]=dict(changed_actions=len(c),one_step_gain=sum(x['one_step_full_gain'] for x in c),
                                     positive_changes=sum(x['one_step_full_gain']>0 for x in c),negative_changes=sum(x['one_step_full_gain']<0 for x in c))
assert recomputed == interventions['summaries']
report['fixed_state_scope'] = interventions['scope']
report['all_fixed_state_summaries'] = recomputed
report['fixed_state_cases_verified'] = len(verified_cases)
report['selected_real_case'] = next(x for x in verified_cases if x['task']=='rss348' and x['seed']==88003 and x['origin']==164 and x['variant']=='scalar_mass')
assert sha(ROOT/'temporal_fusion.py') == SHA
report['frozen_core_unchanged'] = True
report['passed'] = True
(ROOT/'deployment_interface_checks.json').write_text(json.dumps(report,indent=2,allow_nan=False))
(ROOT/'independent_real_case_verification.json').write_text(json.dumps(dict(scope=interventions['scope'],summaries=recomputed,verified_cases=verified_cases,selected=report['selected_real_case']),indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in report.items() if k not in ('malformed_checks','all_fixed_state_summaries','singleton')},indent=2))
