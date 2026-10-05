"""Copy frozen HARTH science and optional completed records without edits.

Results are copied as opaque bytes only after both pipelines have finished.
No raw archive is included and this builder never evaluates physical labels.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,sys
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[2]
NEXT=PROJECT/'work/fusion_cj_next_20261004';PHYSICAL=NEXT/'physical'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def build(destination,include_results=False,sources_only=False,include_development=True):
    destination=Path(destination).resolve();assert not destination.exists(),'Use a new package directory; prior packages remain unchanged.'
    frozen=load(PHYSICAL/'algorithm_freeze.json');assert frozen['source_hashes']
    for original,want in frozen['source_hashes'].items():assert sha(original)==want,original
    if include_results:
        assert (PHYSICAL/'test_chain_summary.json').exists(),'Physical pipeline not complete.'
        assert (PHYSICAL/'external/summary.json').exists(),'External pipeline not complete.'
        assert not sources_only
    destination.mkdir(parents=True);source_records={};extra=[];physical_records=[];history=[];proof_records=[]
    def copy(path,relative=None):
        path=Path(path);relative=Path(relative) if relative is not None else Path('project')/path.relative_to(PROJECT)
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        assert sha(target)==sha(path);return str(relative)
    for original,want in frozen['source_hashes'].items():
        rel=copy(original);source_records[rel]=dict(original_path=original,sha256=want,byte_identical=True)
    extras=[PHYSICAL/'analyze_physical.py',PHYSICAL/'execution_cache.py',PHYSICAL/'verify_execution_cache.py',HERE/'audit_completed_outcomes.py',
        PROJECT/'work/fusion_conditional_20261004/theory/verify_conditional_theory.py']
    for path in extras:extra.append(copy(path))
    # Preserve every final-method proof and review record, including the paid
    # bridge and strict CSV-header amendment. These are opaque byte copies;
    # only Python files are prepared as additional executable replay sources.
    for path in sorted((NEXT/'theory').rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix not in ('.py','.json','.tex','.md'):continue
        rel=copy(path);proof_records.append(rel)
        if path.suffix=='.py':extra.append(rel)
    static=('algorithm_freeze.json','acquisition_authorization.json','acquisition_record.json',
        'acquisition_failure.json','cache_snapshot_before_calibration.json','PREPARATION_STATUS.json',
        'metadata_readiness.json','PREFREEZE_METADATA_AMENDMENT.json','METADATA_PREPARATION.md',
        'PREFREEZE_IMPLEMENTATION_AMENDMENT.json','SOURCE_CLOSURE_ADDENDUM.json',
        'SCHEMA_HEADER_AMENDMENT.json','official_csv_headers.json','ingest_schema_failure.json',
        'execution_cache_report.json','execution_cache_validation_attempt1.json',
        'ingest_authorization.json','ingestion_authorization.json','protocol_amendment.json',
        'acquisition_authorization_v3.json')
    for rel in static:
        if (PHYSICAL/rel).exists():copy(PHYSICAL/rel)
    for path in sorted(PHYSICAL.glob('*.json')):
        name=path.name.lower()
        if any(word in name for word in ('authorization','amendment','schema','closure')):copy(path)
    # Freeze-history snapshots include their old parser/science source bytes,
    # original acquisition record/auth and failed-ingestion provenance.
    for folder in sorted(PHYSICAL.glob('freeze_*')):
        if not folder.is_dir():continue
        for path in sorted(folder.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.zip','.npz'):
                copy(path);history.append(str(path.relative_to(PHYSICAL)))
    if not sources_only:
        for rel in ('data/cache_metadata.json','data/cached_dataset.npz'):
            assert (PHYSICAL/rel).exists(),f'{rel} not ready; use --sources-only for source audit.'
            copy(PHYSICAL/rel)
        metadata=load(PHYSICAL/'data/cache_metadata.json');assert sha(PHYSICAL/'data/cached_dataset.npz')==metadata['cache_sha256']
    if include_results:
        # Copy all matching records, never a favorable subset. The immutable
        # source files/protocols already copied above retain the same bytes.
        for path in sorted(PHYSICAL.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or 'raw' in path.relative_to(PHYSICAL).parts:continue
            if path.suffix not in ('.json','.gz','.csv','.log','.md'):continue
            copy(path);rel=str(path.relative_to(PHYSICAL))
            if rel not in static and not rel.startswith('data/') and rel not in ('protocol.json','external/protocol.json'):
                physical_records.append(rel)
    if include_development:
        for folder in ('design','design_transport','design_quantile','target_theory','quantile_theory'):
            for path in sorted((NEXT/folder).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.json','.gz','.csv','.md','.tex','.log'):
                    copy(path)
    for path in (PROJECT/'work/fusion_conditional_20261004/theory/conditional_theory_report.json',):
        if path.exists():copy(path)
    shutil.copy2(HERE/'run_reproduction.py',destination/'run_reproduction.py')
    shutil.copy2(HERE/'build_package.py',destination/'build_package.py')
    shutil.copy2(HERE/'README.md',destination/'README.md');shutil.copy2(HERE/'LICENSES.md',destination/'LICENSES.md')
    manuscript=PROJECT/'outputs/risk_rf_performance.tex'
    dump(destination/'MANUSCRIPT_POINTER.json',dict(project_relative_path=str(manuscript.relative_to(PROJECT)),
        sha256_at_package_build=sha(manuscript),included=False,
        note='Manuscript may still be edited by root; use the final separately supplied source/PDF. This package preserves frozen algorithms independently.'))
    dump(destination/'SOURCE_IDENTITIES.json',dict(original_project=str(PROJECT),source_count=len(source_records),sources=source_records,
        algorithm_freeze_sha256=sha(PHYSICAL/'algorithm_freeze.json'),scientific_bytes_preserved=True,
        active_method='original CJ original mean/sd/q gate; failed CJ-R/CJ-T/CJ-Q remain development archives'))
    dump(destination/'PACKAGE_CONTENTS.json',dict(extra_sources=extra,physical_records=sorted(set(physical_records)),freeze_history_records=history,proof_records=proof_records,
        results_complete=include_results,processed_cache_present=not sources_only,development_candidates_retained=include_development,
        raw_archive_included=False,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        launcher_source_sha256=sha(HERE/'run_reproduction.py'),builder_source_sha256=sha(__file__)))
    files={str(p.relative_to(destination)):sha(p) for p in sorted(destination.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'}
    dump(destination/'MANIFEST.json',dict(files=files,total_files=len(files),total_bytes=sum((destination/p).stat().st_size for p in files),
        scientific_source_count=len(source_records),numerical_source_rewriting=False,results_complete=include_results,raw_archive_included=False))
    return dict(package=str(destination),files=len(files),bytes=sum((destination/p).stat().st_size for p in files),
        manifest_sha256=sha(destination/'MANIFEST.json'),frozen_sources=len(source_records),results_complete=include_results)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--include-results',action='store_true')
    p.add_argument('--sources-only',action='store_true');p.add_argument('--no-development',action='store_true');a=p.parse_args()
    print(json.dumps(build(a.output,a.include_results,a.sources_only,not a.no_development),indent=2))
