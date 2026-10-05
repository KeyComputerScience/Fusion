"""Portable entry point for the frozen masked-precision reproduction bundle.

Only top-level dependency paths and declared diagnostic budget values are
rebound. Every scientific function AST is checked unchanged before execution.
All reruns write outside the delivered bundle and original dependencies.
"""
from __future__ import annotations
import argparse,ast,datetime,gzip,hashlib,importlib.util,json,os,shutil,sys,types
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
SCIENCE=ROOT/'work/fusion_temporal_20261003'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def dump(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False))
def function_hashes(tree):
    return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
            for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}

def bind_module(name,path,values):
    tree=ast.parse(Path(path).read_text());before=function_hashes(tree);count=0
    for n in tree.body:
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name) and t.id in values:
                    value=values[t.id]
                    n.value=(ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(value))],keywords=[])
                             if isinstance(value,Path) else ast.parse(repr(value),mode='eval').body)
                    count+=1
    ast.fix_missing_locations(tree);assert function_hashes(tree)==before
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(tree,str(path),'exec'),module.__dict__)
    return module,dict(top_level_rebindings=count,function_ast_sha256=before)

def dependencies(parent):
    parent=Path(parent).resolve() if parent else ROOT.parent
    recovery=parent/'Fusion_Recovery_Repro';guard=parent/'Fusion_Loss_Budget_Extension_Repro'
    if not (recovery/'base/cached_runner.py').exists() or not (guard/'guard/run_loss_budget_guard.py').exists():
        raise SystemExit('Place both original dependency directories beside this bundle, or pass --dependency-dir.')
    return recovery,guard

def verify(deps,deep=True):
    manifest=json.loads((ROOT/'MANIFEST.json').read_text());checked=0
    for name,expected in manifest['files'].items():
        assert sha(ROOT/name)==expected,('Changed bundle file',name);checked+=1
    config=json.loads((ROOT/'DEPENDENCIES.json').read_text());depchecks={}
    for folder in deps:
        expected=config['dependencies'][folder.name]['manifest_sha256']
        assert sha(folder/'MANIFEST.json')==expected,('Changed dependency manifest',folder.name)
        files=json.loads((folder/'MANIFEST.json').read_text())['files'];n=0
        if deep:
            for name,h in files.items():assert sha(folder/name)==h,(folder.name,name);n+=1
        depchecks[folder.name]=dict(manifest_sha256=expected,files_checked=n,read_only_dependency=True)
    frozen=json.loads((SCIENCE/'final_algorithm_freeze.json').read_text())
    for name,h in frozen['algorithm_sha256'].items():assert sha(SCIENCE/name)==h
    return dict(passed=True,bundle_files_checked=checked,dependencies=depchecks)

def support_module(deps):
    return bind_module('run_strong_controls',ROOT/'work/fusion_focus_20261003/controls/run_strong_controls.py',
      {'DEFAULT_PACKAGE':deps[0],'DEFAULT_GUARD':deps[1]/'guard'})

def core_module(deps):
    support,identity=support_module(deps)
    spec=importlib.util.spec_from_file_location('temporal_fusion',SCIENCE/'temporal_fusion.py')
    mod=importlib.util.module_from_spec(spec);sys.modules['temporal_fusion']=mod;spec.loader.exec_module(mod)
    assert mod.support is support
    return mod,identity

def safe_output(path):
    if path is None:raise SystemExit('A new --output outside the bundle is required.')
    path=Path(path).resolve()
    if path==ROOT or ROOT in path.parents:raise SystemExit('Keep delivered bundle read-only; choose an external output directory.')
    for ancestor in path.parents:
        if ancestor.name in ('Fusion_Recovery_Repro','Fusion_Loss_Budget_Extension_Repro'):
            raise SystemExit('Do not write into an original dependency.')
    if path.exists() and any(path.iterdir()):raise SystemExit('Use a new empty output directory.')
    path.mkdir(parents=True,exist_ok=True);return path

def invoke_main(mod,arguments):
    previous=sys.argv[:];sys.argv=[mod.__file__]+list(map(str,arguments))
    try:mod.main()
    finally:sys.argv=previous

def reproduce_rss(deps,out):
    mod,identity=core_module(deps)
    invoke_main(mod,['--output',out,'--tasks','rss348','--phase','calibrate'])
    freeze=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(SCIENCE/'temporal_fusion.py'),
                selection_sha256=sha(out/'rss348_selection.json'),adapter_sha256=sha(SCIENCE/'physical/new_data_adapter.py'),
                cache_sha256=sha(SCIENCE/'physical/data/rss348/cached_dataset.npz'),scope='Known-result reproduction; no new validation')
    dump(out/'rerun_pre_test_freeze.json',freeze)
    invoke_main(mod,['--output',out,'--tasks','rss348','--phase','test'])
    assert sha(out/'rss348_selection.json')==freeze['selection_sha256']
    original=json.loads((SCIENCE/'rss_validation/rss348_results.json').read_text())
    rerun=json.loads((out/'rss348_results.json').read_text())
    osa=json.loads((SCIENCE/'rss_validation/rss348_selection.json').read_text());osb=json.loads((out/'rss348_selection.json').read_text())
    checks=dict(exact_full_calibration_grid_and_selected_parameters=osa==osb,
                exact_trial_rows_actions_scores_callbacks_net=original==rerun,
                independent_execution_checks_passed=all(c['passed'] for t in rerun['trials'] for c in t['checks'].values()))
    assert all(checks.values()),checks
    report=dict(passed=True,checks=checks,top_level_path_binding=identity,
                source_sha256=freeze['source_sha256'],calibration_trials_per_arm=27,arms=5,test_scenarios=5,
                scope='Actual relocated known-result reproduction, not additional physical evidence')
    dump(out/'relocated_core_reproduction_report.json',report);return report

def run_budget(deps,out):
    _,identity=support_module(deps)
    mod,binding=bind_module('portable_budget',SCIENCE/'analysis_budget_three_grid.py',{'BUDGETS':(0.,110.,130.,260.)})
    invoke_main(mod,['--inputs',SCIENCE/'final_development',SCIENCE/'rss_validation','--output',out])
    actual=json.loads((out/'summary.json').read_text());expected=json.loads((SCIENCE/'final_budget_four_grid/summary.json').read_text())
    assert actual==expected,'Guard summary differs'
    check=json.loads((out/'verification.json').read_text());assert check['passed']
    trajectories=json.loads((out/'results.json').read_text())
    assert trajectories==json.loads((SCIENCE/'final_budget_four_grid/results.json').read_text())
    original_check=json.loads((SCIENCE/'final_budget_four_grid/verification.json').read_text())
    assert check['uncompressed_jsonl_sha256']==original_check['uncompressed_jsonl_sha256']
    report=dict(passed=True,exact_budget_summary_match=True,exact_all_guarded_trajectories_match=True,
                exact_uncompressed_rows_and_ledgers_match=True,verification=check,dependency_binding=identity,
                budget_binding=binding,evidence_scope='Known frozen-proposal guard reproduction')
    dump(out/'budget_reproduction_report.json',report);return report

def run_solver(deps,out):
    support_module(deps)
    for n in ('temporal_fusion.py','verify_convex_solver.py'):shutil.copyfile(SCIENCE/n,out/n)
    sys.path.insert(0,str(out))
    spec=importlib.util.spec_from_file_location('portable_solver_check',out/'verify_convex_solver.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    result=json.loads((out/'independent_solver_checks.json').read_text());assert result['passed']
    return dict(passed=True,source_sha256=result['source_sha256_at_finish'])

def run_interface(deps,out):
    core_module(deps)
    for name in ('temporal_fusion.py','deployment_interface.py','fixed_state_interventions.json',
                 'final_development/results.json','final_development/protocol.json',
                 'rss_validation/results.json','rss_validation/protocol.json',
                 'physical/data/rss348/cache_metadata.json',
                 'physical/data/rss348/cached_dataset.npz'):
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(SCIENCE/name,target)
        assert sha(target)==sha(SCIENCE/name)
    sys.path.insert(0,str(SCIENCE))
    _,binding=bind_module('portable_interface_check',SCIENCE/'verify_deployment_interface.py',{'ROOT':out})
    result=json.loads((out/'deployment_interface_checks.json').read_text());assert result['passed']
    return dict(passed=True,interface_checks=result,diagnostic_path_binding=binding,
                frozen_core_unchanged=result['frozen_core_unchanged'],
                scope='Known-state interface verification, not new physical performance')

def inspect_records():
    report=json.loads((SCIENCE/'analysis_summary.json').read_text())
    return dict(evidence_scope='7 known development tasks;1 new physical provided-group/trajectory testRSS348',
       execution=report['execution'],core_summary={n:r['summary']['precision_joint'] for n,r in report['tasks'].items()},
       external=json.loads((SCIENCE/'rss_external/summary.json').read_text()),
       budget_checks=json.loads((SCIENCE/'final_budget_four_grid/verification.json').read_text()),
       pamap_status=json.loads((SCIENCE/'physical/PAMAP_ACQUISITION_STATUS.json').read_text()))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['verify','inspect','rss','budget','solver','interface'])
    p.add_argument('--dependency-dir',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    deps=dependencies(a.dependency_dir);check=verify(deps,deep=True)
    if a.phase=='verify':print(json.dumps(check,indent=2));return
    if a.phase=='inspect':print(json.dumps(inspect_records(),indent=2));return
    out=safe_output(a.output)
    if a.phase=='rss':result=reproduce_rss(deps,out)
    elif a.phase=='budget':result=run_budget(deps,out)
    elif a.phase=='interface':result=run_interface(deps,out)
    else:result=run_solver(deps,out)
    dump(out/'bundle_verification_before_rerun.json',check);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
