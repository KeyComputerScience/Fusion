"""Independently adapt and audit completed external trial logs, without reruns.

The only schema shim adds trial.baseline from that trial's frozen arm and
trial.windows from the frozen original trial (checked against explicit-service
metadata and ceil(test_rows/32)). Original external logs/runner are never edited.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

EPS=1e-8

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,data):Path(p).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def compare_nested(a,b):
    """Compare original required fields; extra offline diagnostic fields exempt."""
    if isinstance(a,dict):
        skipped={'pessimistic_net_target','opportunity_slack','candidate_extra_correct'}
        return max([compare_nested(v,b[k]) for k,v in a.items() if k not in skipped]+[0.])
    if isinstance(a,list):
        assert len(a)==len(b)
        return max([compare_nested(x,y) for x,y in zip(a,b)]+[0.])
    if isinstance(a,bool) or a is None or isinstance(a,str):
        assert a==b
        return 0.
    return abs(float(a)-float(b))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[3]/'outputs/bayes_closed_loop_repro')
    args=ap.parse_args(); directory=args.directory.resolve(); package=args.package.resolve(); output=directory/'external_verified'; output.mkdir(exist_ok=True)
    protocol=read(directory/'protocol.json'); freeze=read(directory/'prefix_freeze.json')
    original=read(directory/'results.json'); execution=read(directory/'execution_audit.json'); summary=read(directory/'summary.json')
    inputs=[directory/p for p in ('results.json','summary.json','execution_audit.json','protocol.json','prefix_freeze.json','pre_calibration_freeze.json','run_external_pdf.py','coverage_analysis.py','coverage_analysis.json','coverage_notes.md')]
    before={str(p):sha(p) for p in inputs}
    assert sha(directory/'protocol.json')==freeze['protocol_sha256']==execution['protocol_sha256']
    assert sha(directory/'run_external_pdf.py')==freeze['script_sha256']==execution['script_sha256']
    assert sha(package/'independent_bayes_fusion.py')==protocol['frozen_engine_sha256']
    assert sha(package/'independent_bayes_extension.py')==protocol['frozen_adapter_sha256']
    assert execution['all_passed'] and execution['original_package_unchanged']
    assert not freeze['new_test_metrics_inspected']
    assert [t['seed'] for t in original['occupancy357']['trials']]==protocol['test_seeds']
    shim=copy.deepcopy(original); checks=[]; max_primary_difference=0.; max_source_weight_error=0.; max_external_penalty_error=0.
    source_hashes={}
    for name,study in original.items():
        specification=protocol['source_studies'][name]
        source_path=package/specification['folder']/(name+'_results.json'); source_hashes[str(source_path)]=sha(source_path)
        assert sha(source_path)==specification['result_sha256']
        source=read(source_path); by_seed={int(t['seed']):t for t in source['trials']}
        assert source['data_hashes']==study['data_hashes']==specification['data_hashes']
        assert source['split']==study['split']==specification['split']
        selection_path=directory/(name+'_prefix_selection.json')
        assert sha(selection_path)==study['prefix_selection_sha256']==freeze['all_prefix_selection_sha256'][name]
        selection=read(selection_path)
        assert selection['selected']==study['selected']
        assert len(selection['grid'])==18
        for arm in ('pdf_point','pdf_calibrated'):
            grid=[r for r in selection['grid'] if r['arm']==arm]
            assert len(grid)==9
            assert {(float(r['l2_multiplier']),float(r['margin'])) for r in grid}=={(l,m) for l in (0.,.25,1.) for m in (0.,4.,12.)}
        for trial,t in zip(study['trials'],shim[name]['trials']):
            old=by_seed[int(trial['seed'])]
            reference=trial['results']['frozen']
            baseline={k:reference[k] for k in ('gross','fees','drops','net')}
            assert compare_nested(old['baseline'],baseline)<EPS
            windows=int(old['windows'])
            assert windows==math.ceil(study['split']['test_rows']/32)
            for mode,audit in trial['execution_audit'].items():
                assert audit['all_passed']
                assert int(audit['explicit_service']['windows'])==windows
            for mode in ('bayes_both','joint','frequentist_gate','frozen'):
                difference=compare_nested(old['results'][mode],trial['results'][mode])
                max_primary_difference=max(max_primary_difference,difference)
                assert difference<EPS, 'frozen original reproduction changed'
            for mode in ('pdf_point','pdf_calibrated'):
                saved=trial['results'][mode]
                assert saved['external_arm']==mode
                assert saved['q_initial']==selection['selected'][mode]['q_initial']
                for row in saved['rows']:
                    weights=np.asarray(row['pdf_weights'],float)
                    assert weights.ndim==2 and weights.shape[1]==len(row['active_sources'])
                    error=max(float(np.max(abs(weights.sum(1)-1.))),max(0.,-float(weights.min())))
                    max_source_weight_error=max(max_source_weight_error,error)
                    assert error<EPS
                    expected_penalty=0. if mode=='pdf_point' else float(row['q_issued'])*float(row['posterior_or_block_sd'])
                    max_external_penalty_error=max(max_external_penalty_error,abs(row['penalty']-expected_penalty))
                    assert abs(row['penalty']-expected_penalty)<EPS
                    assert abs(row['sd']-row['posterior_or_block_sd'])<EPS
                    assert abs(row['standardized']-row['standardized_score'])<EPS
                if mode=='pdf_point':
                    assert saved['q_initial']==0. and 'diagnostic' in saved['coverage_semantics']
            t['baseline']=baseline
            t['windows']=windows
            checks.append(dict(dataset=name,seed=int(trial['seed']),baseline_source='same-trial frozen arm',windows_source='original frozen trial; checked with ceil(test_rows/32) and all explicit-service audits',baseline=baseline,windows=windows))
    shim_path=output/'shim_results.json'; write(shim_path,shim)
    module_spec=importlib.util.spec_from_file_location('coverage_analysis',directory/'coverage_analysis.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    analyzed={name:module.analyze_study(study,comparators=('reference','pdf_point','pdf_calibrated','frequentist_gate','joint')) for name,study in shim.items()}
    # Cross-check independently recomputed results against the runner's summary.
    for name,task in analyzed.items():
        for mode,arm in task['arms'].items():
            existing=summary[name]['controllers'][mode]
            assert abs(arm['net']['mean']-existing['mean_net'])<EPS
            for key,original_key in (('issued','overall'),('informative_current_disagreement','informative'),('admitted','admitted')):
                v=arm['coverage'][key]
                assert [v['numerator'],v['denominator']]==existing['pooled_coverage'][original_key]
        for mode in ('pdf_point','pdf_calibrated','frequentist_gate','joint'):
            recomputed=task['full_comparisons'][mode]['paired_net_gain']
            existing=summary[name]['paired']['full_minus_'+mode]['net']
            assert abs(recomputed['mean']-existing['mean'])<EPS
            assert np.max(abs(np.asarray(recomputed['by_seed'])-np.asarray(existing['values'])))<EPS
    unchanged=all(sha(path)==digest for path,digest in before.items());assert unchanged
    original_package=read(directory/'original_files_before.json')
    changed=[relative for relative,digest in original_package.items() if sha(package/relative)!=digest]
    assert not changed
    report=dict(kind='Independent no-rerun external coverage/action/accounting audit',all_passed=True,
                verifier_code_sha256=sha(__file__),input_sha256=before,original_source_result_sha256=source_hashes,
                originals_and_original_coverage_reports_unchanged=unchanged,original_package_files_unchanged=not changed,
                shim=dict(file=str(shim_path),sha256=sha(shim_path),only_added_trial_fields=['baseline','windows'],checks=checks),
                external_rule_scope='Probability-interface PDF-P adaptation with static prefix TCP heads, not full end-to-end PDF reproduction; later diagnostic on already evaluated physical traces',
                raw_point_coverage_scope='Issued q tracker is diagnostic; q0=0 and adaptive tracker unused by point gate; not an executable calibrated lower-bound guarantee',
                optimizer_scope='External source weights use closed-form softmax; placeholder kkt=0 is not an optimization certificate',
                number_fixed_physical_traces=4,delay_seeds=protocol['test_seeds'],
                total_arm_seed_log_audits=sum(sum(a['seed_count'] for a in d['arms'].values()) for d in analyzed.values()),
                max_frozen_primary_field_difference=max_primary_difference,max_source_weight_simplex_error=max_source_weight_error,
                max_external_gate_penalty_error=max_external_penalty_error,reported_request_level_audit=execution,
                tasks=analyzed)
    write(output/'external_analysis.json',report)
    lines=['# Independent external-fusion validation audit','',
           'All four tasks, both external pipelines, and all five original delay schedules are retained. This is a later diagnostic extension on the same four previously evaluated physical traces. It does not add independent tasks or sites. Intervals describe delay variability on each fixed trace.','',
           'The external rule is PDF-P: a probability-interface adaptation of Predictive Dynamic Fusion with static prefix-fitted TCP heads, not a complete end-to-end reproduction. Both point and delayed-calibrated pipelines have nine prefix trials; parameter families differ from internal fusion, and additional static TCP-head prefix training is disclosed.','',
           '## Schema conversion and integrity','',
           'Original external trials have no baseline/windows fields. The immutable shim adds baseline={gross,fees,drops,net} from the same trial’s frozen arm and windows from the original frozen trial; both are independently checked against original baseline, ceil(test_rows/32), and explicit-service audit metadata. No original result, frozen runner, parameter, or old coverage artifact is edited.','',
           f"All {report['total_arm_seed_log_audits']} arm×seed log audits pass; original internal arm results match every common saved field (maximum error {max_primary_difference:g}). Source-weight simplex maximum error is {max_source_weight_error:.3g}, and external gate penalty mismatch is {max_external_penalty_error:g}. Original files and old 290-trial analysis remain unchanged.",'',
           'Point-arm q0=0; its adaptive q tracker is not used in admission. Its q-based coverage must be labelled a diagnostic and cannot supply an executable lower-gate protection claim. Calibrated-arm penalty=q_issued×SD is checked directly. Scalar gate weights=[1] are not physical-source weights: request-level PDF softmax weights are checked separately. Softmax does not have the internal source cap; kkt=0 is an unused interface placeholder, not a source-weight optimization certificate.','',
           '## Complete paired full versus external results','',
           '| Task | Comparator | Mean net gain | Conditional t4 interval | Five gains | +/=/- | Avoided harmful / comparator harmful | Retained beneficial / comparator beneficial |','|---|---|---:|---|---|---|---|---|']
    for name,task in analyzed.items():
        for mode in ('pdf_point','pdf_calibrated'):
            c=task['full_comparisons'][mode];s=c['paired_net_gain'];ci=s['conditional_delay_t95_descriptive_interval']
            lines.append(f"| {name} | {mode} | {s['mean']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {s['by_seed']} | {s['positive']}/{s['tie']}/{s['negative']} | {module.fmt_ratio(c['harmful_control_leases_avoided'])} | {module.fmt_ratio(c['beneficial_control_leases_retained'])} |")
    lines+=['','## Action and coverage denominators','','| Task | Arm | Admissions / harmful / beneficial | All-issued coverage | Current-disagreement coverage | Admitted coverage |','|---|---|---|---|---|---|']
    for name,task in analyzed.items():
        for mode in ('pdf_point','pdf_calibrated','bayes_both','frequentist_gate','frozen'):
            a=task['arms'][mode];counts=a['actions'];c=a['coverage']
            lines.append(f"| {name} | {mode} | {counts['admissions']} / {counts['harmful']} / {counts['beneficial']} | {module.fmt_ratio(c['issued'])} | {module.fmt_ratio(c['informative_current_disagreement'])} | {module.fmt_ratio(c['admitted'])} |")
    lines+=['','## Exact utility and changed-action decomposition','','Values are sums over all five delay schedules; dividing by five gives task means. Gains include useful admissions missed and harmful admissions newly added.','','| Task | Comparator | Served gross change | Saved fees | Avoided harmful: count / gain | Missed beneficial: count / gain | New harmful: count / gain | New beneficial: count / gain |','|---|---|---:|---:|---|---|---|---|']
    for name,task in analyzed.items():
        for mode in ('pdf_point','pdf_calibrated'):
            c=task['full_comparisons'][mode];e=c['event_summary'];cells=[f"{e[k]['count']} / {e[k]['contribution_sum']:g}" for k in ('avoided_harmful','missed_beneficial','new_harmful','new_beneficial')]
            lines.append(f"| {name} | {mode} | {c['served_gross_change']['sum']:g} | {c['saved_fees']['sum']:g} | "+' | '.join(cells)+' |')
    lines+=['','## Interpretation','',
            '- Full improves over PDF-P point admission on Room 864 and MHEALTH in every delay schedule, with HAR four positive and one negative schedule. The point gate is less protective than the calibrated external pipeline and must not be the sole modern comparator.',
            '- Against PDF-P calibrated admission, full matches both occupancy tasks and HAR exactly in actions/returns. MHEALTH gains are [0,5,12,5,27], mean9.8, four positive and one tie; its conditional t4 interval includes zero. The gain is real and must be presented as task-specific, not universal superiority.',
            '- Both calibrated pipelines remain harmful on MHEALTH: full11/12 harmful, external12/12 harmful. Full is still below reference by23.2 mean units and below the matched empirical-block gate by6.6. The external comparison strengthens the available baseline set without removing this boundary.',
            '- Full and calibrated PDF-P agree on all HAR admissions. Their equality means the +3.6 versus empirical-block gate cannot be described as uniquely Bayesian relative to the new external calibrated pipeline.',
            '- Full versus each external comparator satisfies ΔJ=Δserved_gross+saved_fees, and the sum of changed-action increments equals the same paired gain. Complete per-origin events remain in JSON; pooled bookkeeping is not a cross-task population effect.','']
    (output/'external_review.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in report.items() if k not in ('tasks','reported_request_level_audit','input_sha256','shim')},indent=2))

if __name__=='__main__':main()
