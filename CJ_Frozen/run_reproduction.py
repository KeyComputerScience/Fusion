"""Portable launch of unchanged final CJ/HARTH scientific source bytes.

Rebinding is restricted to filesystem globals and import locations. Archived
freeze bytes stay unchanged. Fresh replay timestamps are not preregistration.
"""
from pathlib import Path
import argparse,ast,datetime,gzip,hashlib,importlib.abc,importlib.machinery,importlib.util,json,os,shutil,sys
sys.dont_write_bytecode=True
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parent
PHYSICAL_REL=Path('work/fusion_cj_next_20261004/physical')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt') as f:return json.load(f)
    return json.loads(Path(path).read_text())
def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def verify():
    manifest=load(ROOT/'MANIFEST.json');bad=[]
    for name,expected in manifest['files'].items():
        if not (ROOT/name).is_file() or sha(ROOT/name)!=expected:bad.append(name)
    assert not bad,('package hashes',bad)
    identities=load(ROOT/'SOURCE_IDENTITIES.json');old=Path(identities['original_project'])
    frozen=load(ROOT/'project'/PHYSICAL_REL/'algorithm_freeze.json')
    assert len(frozen['source_hashes'])==identities['source_count']
    for original,expected in frozen['source_hashes'].items():
        path=ROOT/'project'/Path(original).relative_to(old)
        assert sha(path)==expected,original
        assert identities['sources'][str(path.relative_to(ROOT))]['sha256']==expected
    physical=ROOT/'project'/PHYSICAL_REL
    auth=load(physical/'acquisition_authorization.json')
    assert sha(physical/'algorithm_freeze.json')==auth['algorithm_freeze_sha256']
    assert sha(physical/'protocol.json')==auth['physical_protocol_sha256']
    assert sha(physical/'adapter.py')==auth['adapter_sha256']
    preserved=0
    for snapshot in physical.glob('freeze_*/snapshot_manifest.json'):
        for entry in load(snapshot)['files'].values():
            assert sha(snapshot.parent/entry['snapshot'])==entry['sha256'],entry['snapshot']
            preserved+=1
    cache=physical/'data/cached_dataset.npz';has_cache=cache.exists()
    if has_cache:
        meta=load(physical/'data/cache_metadata.json')
        assert sha(cache)==meta['cache_sha256']
        assert meta['adapter_sha256']==sha(physical/'adapter.py')
        assert meta['physical_protocol_sha256']==sha(physical/'protocol.json')
        assert meta['algorithm_freeze_sha256']==sha(physical/'algorithm_freeze.json')
    return dict(passed=True,package_files=len(manifest['files']),frozen_scientific_sources=len(frozen['source_hashes']),
        all_scientific_bytes_unchanged=True,processed_cache_present=has_cache,raw_archive_bundled=False,
        numerical_source_rewriting=False,original_project_access_required=False,preserved_freeze_snapshot_files=preserved)

def install_paths(project):
    """Rebase Path arguments and only literal top-level filesystem PROJECT.

    Imports use the rebased project as their only local source location.
    The transformer never descends into a numerical function/class body.
    """
    project=Path(project).resolve();old=Path(load(ROOT/'SOURCE_IDENTITIES.json')['original_project'])
    def resolve(path):
        path=Path(path)
        if path.is_absolute() and path.is_relative_to(project):return path
        try:return project/path.relative_to(old)
        except ValueError:return path
    def no_original_reads(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).absolute()
            if path.is_relative_to(old) and not path.is_relative_to(project) and not path.is_relative_to(ROOT):
                raise RuntimeError('Portable replay attempted original-workspace access: '+str(path))
    sys.addaudithook(no_original_reads)
    class FilesystemPath(type(Path())):
        def __new__(cls,*args,**kwargs):
            if args:
                args=(str(resolve(args[0])),)+args[1:]
            return super().__new__(cls,*args,**kwargs)
        def __init__(self,*args,**kwargs):
            # Python3.12 moved path argument storage from __new__ to __init__.
            if sys.version_info>=(3,12):
                if args:args=(str(resolve(args[0])),)+args[1:]
                super().__init__(*args,**kwargs)
    class FilesystemLoader(importlib.abc.Loader):
        def __init__(self,path):self.path=Path(path)
        def create_module(self,spec):return None
        def exec_module(self,m):
            tree=ast.parse(self.path.read_bytes(),filename=str(self.path))
            bodies=[ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
            for statement in tree.body:
                if isinstance(statement,ast.Assign):
                    # Several old adapters hard-code PROJECT; preserve every
                    # other expression and all function/class definitions.
                    targets=[x.id for x in statement.targets if isinstance(x,ast.Name)]
                    if targets==['PROJECT'] and isinstance(statement.value,ast.Call):
                        call=statement.value
                        if isinstance(call.func,ast.Name) and call.func.id=='Path' and call.args and isinstance(call.args[0],ast.Constant) and isinstance(call.args[0].value,str):
                            original=Path(call.args[0].value)
                            if original==old:call.args[0]=ast.Constant(str(project))
            ast.fix_missing_locations(tree)
            assert bodies==[ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
            m.__file__=str(self.path);exec(compile(tree,str(self.path),'exec'),m.__dict__)
            # Functions keep their unchanged code; their Path global now maps
            # historical absolute freeze keys to the copied filesystem only.
            if m.__dict__.get('Path') is Path:m.__dict__['Path']=FilesystemPath
    original_spec=importlib.util.spec_from_file_location
    def portable_spec(name,location,*args,**kwargs):
        location=resolve(location)
        if location.suffix=='.py' and location.resolve().is_relative_to(project):
            kwargs['loader']=FilesystemLoader(location)
        return original_spec(name,location,*args,**kwargs)
    importlib.util.spec_from_file_location=portable_spec
    class FilesystemFinder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
            if spec and spec.origin and str(spec.origin).endswith('.py'):
                origin=Path(spec.origin).resolve()
                if origin.is_relative_to(project):spec.loader=FilesystemLoader(origin)
            return spec
    sys.meta_path.insert(0,FilesystemFinder())
    return dict(project=str(project),original_project=str(old),scope='top-level literal PROJECT, module Path globals, and import paths only',
        original_workspace_reads_denied=True,function_and_class_bodies_identical=True)

def prepare(output,archive=False):
    output=Path(output).resolve()
    if output==ROOT or ROOT in output.parents:raise ValueError('Use a fresh output outside the delivered package.')
    output.mkdir(parents=True,exist_ok=True);project=output/'project'
    state=output/'REPLAY_STATE.json'
    if state.exists():
        metadata=load(state);assert metadata['package_manifest_sha256']==sha(ROOT/'MANIFEST.json')
        assert project.exists()
    else:
        assert not list(output.iterdir()),'Initial output directory must be empty.'
        project.mkdir();identities=load(ROOT/'SOURCE_IDENTITIES.json')
        # Every copied scientific file retains its original relative location.
        for rel in identities['sources']:
            source=ROOT/rel;target=output/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        original=ROOT/'project'/PHYSICAL_REL;target=project/PHYSICAL_REL
        static=('algorithm_freeze.json','acquisition_authorization.json','acquisition_record.json',
            'data/cache_metadata.json','data/cached_dataset.npz','cache_snapshot_before_calibration.json',
            'execution_cache_report.json','execution_cache_validation_attempt1.json',
            'EXECUTION_TEST_BINDING.json','external_calibration_validation.json')
        for rel in static:
            source=original/rel
            if source.exists():dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        # Additional independent theory/analysis code is byte-identical too.
        for rel in load(ROOT/'PACKAGE_CONTENTS.json')['extra_sources']:
            source=ROOT/rel;dest=output/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        dump(state,dict(package_manifest_sha256=sha(ROOT/'MANIFEST.json'),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            archived_freezes_unchanged=True,reproduction_is_new_registration=False))
    if archive:
        original=ROOT/'project'/PHYSICAL_REL;target=project/PHYSICAL_REL
        for rel in load(ROOT/'PACKAGE_CONTENTS.json')['physical_records']:
            source=original/rel;dest=target/rel
            if dest.exists():assert sha(dest)==sha(source),('existing replay differs from archive',rel)
            else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    immutable=list(load(ROOT/'SOURCE_IDENTITIES.json')['sources'])+load(ROOT/'PACKAGE_CONTENTS.json')['extra_sources']
    for rel in immutable:assert sha(output/rel)==sha(ROOT/rel),('replay source bytes changed',rel)
    install_paths(project)
    return output,project

def compare_phase(command,project):
    """Compare scientific records; path/time-only lock files are excluded."""
    if not load(ROOT/'PACKAGE_CONTENTS.json')['results_complete']:
        return dict(available=False,reason='Completed archived physical results are not bundled.')
    before=ROOT/'project'/PHYSICAL_REL;after=project/PHYSICAL_REL
    patterns={
        'physical-calibrate':('selection.json','calibration_trials.json.gz'),
        'external-calibrate':('external/selection.json','external/calibration_trials.json.gz'),
        'physical-test':('S*_results.json.gz','S*_guarded.json.gz','S*_fixed_state.json.gz','S*_verification.json','test_chain_summary.json'),
        'external-test':('external/S*_results.json.gz','external/guarded_all.json.gz','external/issued_all.json.gz','external/summary.json'),
        'analyze':('analysis.json','independent_outcome_audit.json',)}
    files=sorted({path for pattern in patterns.get(command,()) for path in before.glob(pattern) if path.is_file()})
    numeric=0;max_error=0.;exact=True
    def compare(a,b,path):
        nonlocal numeric,max_error,exact
        if isinstance(a,dict):
            assert isinstance(b,dict) and a.keys()==b.keys(),('record keys',path)
            for key in a:compare(a[key],b[key],path+'/'+str(key))
        elif isinstance(a,list):
            assert isinstance(b,list) and len(a)==len(b),('record length',path)
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
        elif isinstance(a,bool) or not isinstance(a,(int,float)):
            assert a==b,('discrete record',path,a,b)
        else:
            assert isinstance(b,(int,float)) and not isinstance(b,bool),('numeric type',path)
            error=abs(a-b);numeric+=1;max_error=max(max_error,error);exact=exact and a==b
            assert error<=1e-10+1e-12*abs(a),('numeric record',path,a,b)
    for source in files:compare(load(source),load(after/source.relative_to(before)),str(source.relative_to(before)))
    return dict(available=bool(files),passed=True,files=len(files),numeric_leaves=numeric,all_decoded_values_exact=exact,
        maximum_absolute_difference=max_error,absolute_tolerance=1e-10,relative_tolerance=1e-12,
        discrete_structure_and_actions_exact=True,locks_with_new_paths_and_timestamps_excluded=True)

def execute(command,output,cache_prefix_training=False):
    if cache_prefix_training and command not in ('external-calibrate','external-test'):
        raise ValueError('--cache-prefix-training applies only to external calibration/test.')
    output,project=prepare(output,archive=command=='analyze');physical=project/PHYSICAL_REL
    if command=='theory':
        target=project/'work/fusion_cj_next_20261004/theory/verify_final_original.py'
        check=module('portable_final_original_theory',target);check.main()
        report=load(target.parent/'final_original_api_report.json');assert report['passed']
        law_target=project/'work/fusion_conditional_20261004/theory/verify_conditional_theory.py'
        law_check=module('portable_original_conditional_theory',law_target);law_report=law_check.check()
        dump(law_target.parent/'conditional_theory_report.json',law_report)
        assert law_report['execution']['passed']
        paid_target=project/'work/fusion_cj_next_20261004/theory/verify_final_paid_bridge.py'
        paid_check=module('portable_final_paid_bridge',paid_target);paid_report=paid_check.finite_bridge()
        assert paid_report['passed']
        dump(paid_target.parent/'final_paid_bridge_report.json',paid_report)
        result=dict(passed=True,source_closure_and_constructed_states=True,no_physical_outcomes_read=True,
            report=report,conditional_information_mechanism=law_report,fitted_paid_benefit=paid_report)
    elif command=='analyze':
        audit_target=project/'work/fusion_cj_next_20261004/reproduction/audit_completed_outcomes.py'
        auditor=module('portable_completed_outcome_audit',audit_target);audit_report=auditor.audit(physical)
        assert audit_report['status']=='passed';dump(physical/'independent_outcome_audit.json',audit_report)
        target=physical/'analyze_physical.py';analysis=module('portable_harth_analysis',target);analysis.main()
        result=dict(passed=True,analysis=str(physical/'analysis.json'),archived_results_reanalyzed=True,
            independent_completed_outcome_audit=True,audit=str(physical/'independent_outcome_audit.json'))
    elif command in ('physical-calibrate','locktest','physical-test'):
        runner=module('portable_harth_physical',physical/'run_physical.py')
        getattr(runner,{'physical-calibrate':'calibrate','locktest':'locktest','physical-test':'test'}[command])()
        result=dict(passed=True,phase=command)
    else:
        runner=module('portable_harth_external',physical/'run_external.py')
        cache=None
        if cache_prefix_training:
            runner.verify();frozen=runner.p.verify()
            cache_module=module('portable_exact_prefix_train_cache',physical/'execution_cache.py')
            cache=cache_module.install(runner.nn,frozen['base_cfg']['train_buffer'])
            audit_path=output/(command+'_execution_cache_audit.json')
            assert not audit_path.exists(),'Preserve prior cache execution audits.'
            cache_before=dict(wrapper_sha256=sha(physical/'execution_cache.py'),
                runner_sha256=sha(physical/'run_external.py'),algorithm_freeze_sha256=sha(physical/'algorithm_freeze.json'),
                phase=command,optimization_only=True,numerical_method_changed=False)
            dump(audit_path,dict(before=cache_before,status='running',cache=cache.report()))
        status='failed'
        try:
            getattr(runner,{'external-calibrate':'calibrate','external-test':'test'}[command])();status='completed'
        finally:
            if cache is not None:dump(audit_path,dict(before=cache_before,status=status,cache=cache.report()))
        result=dict(passed=True,phase=command,exact_prefix_training_cache=cache is not None)
    result['archived_record_comparison']=compare_phase(command,project)
    dump(output/(command+'_reproduction_report.json'),dict(**result,scientific_bytes_unchanged=True,
        filesystem_rebinding_only=True,raw_acquisition_required=False,reproduction_is_new_registration=False))
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('verify','analyze','theory','physical-calibrate','external-calibrate','locktest','physical-test','external-test'))
    parser.add_argument('--output',type=Path);parser.add_argument('--cache-prefix-training',action='store_true')
    args=parser.parse_args()
    if args.cache_prefix_training and args.command not in ('external-calibrate','external-test'):
        raise ValueError('--cache-prefix-training applies only to external calibration/test.')
    report=verify()
    if args.command!='verify':
        if args.output is None:raise ValueError('--output is required for execution.')
        report=execute(args.command,args.output,args.cache_prefix_training)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
