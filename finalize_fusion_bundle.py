from pathlib import Path
import hashlib, json, shutil
R=Path('/Users/key/Documents/Codex/2026-10-02/jih')
B=R/'outputs/Fusion_External_Validation_Repro'
S=R/'work/external_fusion_validation_20261002'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2)+'\n')
# Preserve frozen runners, protocols and complete logs using the final source inventory.
source_inventory=json.loads((S/'copy_manifest.json').read_text())
common={'paper_provenance.md','external_references.bib','summary.md','combined_summary.json'}
analysis={'coverage_analysis.py','coverage_analysis.json','coverage_notes.md'}
copy_report=[]
for item in source_inventory['required_files']:
    rel=item['path']; source=S/rel
    assert sha(source)==item['sha256'], rel
    if rel=='consistency_review.md': dest=B/'verification/consistency_review.md'
    elif rel in common: dest=B/'external'/rel
    elif rel in analysis: dest=B/'analysis'/('original_'+rel if rel!='coverage_analysis.py' else rel)
    elif rel.startswith('qmf/portable_full_rerun/'): dest=B/'verification'/('qmf_source_'+Path(rel).name)
    elif rel.startswith('portable_full_rerun/'): dest=B/'verification'/('pdf_source_'+Path(rel).name)
    elif rel.startswith('qmf/'): dest=B/'external'/rel
    else: dest=B/'external/pdf'/rel
    dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,dest)
    assert sha(dest)==item['sha256'], rel
    copy_report.append(dict(source=rel,delivered=str(dest.relative_to(B)),sha256=item['sha256']))
shutil.copy2(S/'copy_manifest.json',B/'verification/external_source_copy_manifest.json')
# This unchanged helper is imported by the QMF runner from its parent directory.
shutil.copy2(S/'run_external_pdf.py',B/'external/run_external_pdf.py')
# Keep PDF's own provenance link usable without changing its source README.
shutil.copy2(S/'paper_provenance.md',B/'external/pdf/paper_provenance.md')
# Parent confirms these manuscript sources are final and compiled.
for name in ('risk_rf_performance.tex','risk_rf_revision.tex','risk_rf_references.bib'):
    if (R/'outputs'/name).exists(): shutil.copy2(R/'outputs'/name,B/'manuscript'/name)
# Keep fresh unified full-run proofs, not duplicate complete result files.
F=Path('/private/tmp/fusion_external_all_unified_20261003')
for p in F.iterdir():
    if p.is_file(): shutil.copy2(p,B/'verification'/('external_unified_'+p.name))
for method in ('pdf','qmf'):
    shutil.copy2(F/method/'portable_proof.json',B/'verification'/(method+'_unified_portable_proof.json'))
A=Path('/private/tmp/fusion_all_analysis_unified_20261003')
for p in (A/'analysis').iterdir():
    if p.is_file(): shutil.copy2(p,B/'analysis'/p.name)
for p in A.iterdir():
    if p.is_file(): shutil.copy2(p,B/'verification'/('all_cohorts_'+p.name))
# Last byte-for-byte protection of the original 67-file base.
orig=R/'outputs/bayes_closed_loop_repro'; supplied=B/'bayes_closed_loop_repro'
def inventory(root): return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
x,y=inventory(orig),inventory(supplied)
byte_check=dict(passed=x==y,original_files=len(x),delivered_files=len(y),changed=sorted(k for k in x.keys()&y.keys() if x[k]!=y[k]),added=sorted(y.keys()-x.keys()),missing=sorted(x.keys()-y.keys()))
assert byte_check['passed'] and len(x)==67
dump(B/'verification/frozen_base_byte_preservation.json',byte_check)
dump(B/'verification/external_source_copy_verification.json',dict(passed=True,files=copy_report,source_inventory_verified=True))
p=B/'PORTABILITY_VERIFICATION.json'; v=json.loads(p.read_text())
v['commands']=['verify','audit-primary','audit-supp','analyze','rerun-primary','rerun-supp','crossings','sensitivity','benchmark','external-pdf','external','analyze-external']
v['untested_in_this_pass']=['hardware/field deployment and persistent restart recovery']
v.update(external_qmf_full_prefix_and_policy_execution_fields_match=True,
    external_qmf_stage_trajectory_checks=120,new_qmf_policy_replays=40,
    external_qmf_internal_reference_reconstruction_checks=80,
    all_original_and_external_stage_checks=530,
    new_external_policy_replays_total=80,
    original_and_both_external_analysis_passed=True,
    unified_external_fresh_output='/private/tmp/fusion_external_all_unified_20261003',
    all_retained_analysis_output='/private/tmp/fusion_all_analysis_unified_20261003',
    stage_count_interpretation='290 original logs + PDF 120 + QMF 120 stage checks. Each external stage includes 40 new policy replays and 80 reconstructions of existing controls; these are repeated interventions on four fixed traces, not 530 independent experiments.')
v.pop('original_and_pdf_analysis_arm_seed_checks',None)
dump(p,v)
p=B/'BUNDLE_VERSION.json'; v=json.loads(p.read_text()); v['finalized_date']='2026-10-03';v['layout']['external']='Separate frozen PDF-P and QMF-E probability/energy-rule adaptations, with full prefix grids and test logs';v['source_hashes'].update(pdf_runner='50dbfaa37d89ff97975bc61cdd1ec4b6afc9de63cfea199c21ec02068dc3c517',qmf_runner='a0484d795adcbc0626c744a72d6b097b1713677c74aa076f3c2e2225c3812ba4');dump(p,v)
p=B/'README.md'; s=p.read_text()
s=s.replace('External comparator files, when added under `external/`, have their own declared protocol and results; they do not replace the original study.','The supplied PDF-P and QMF-E external adaptations under `external/` have separate declared protocols and full results.')
s=s.replace('# When the separately declared QMF-E package is present, run it similarly.','# Separately declared QMF-E energy-rule adaptation: full prefix fitting and test.')
s=s.replace('The unified PDF-P command also rebuilt prefix heads/grid and matched every retained policy execution field. Its 120 stage checks comprise 40 new PDF-P policy replays and 80 reconstructions of the existing internal controllers; these repeated controls are not additional independent tasks or primary trials.','The unified external command rebuilt both PDF-P and QMF-E prefix fits/grids and matched every retained policy execution field. Each adaptation has 120 stage checks: 40 new policy replays and 80 reconstructions of existing internal controllers. The original 290 logs plus these two sets give 530 stage checks, including 80 new external policy replays. Repeated controls and imposed delay schedules do not provide additional independent traces.')
s=s.replace('Candidate preparation/source training is shared across controllers.','Candidate preparation/source training is shared across controllers. The training algorithm and schedule are frozen, while common model parameters continue to update causally from arrived eligible labels. External PDF TCP heads are fitted on the prefix and held fixed during testing.')
s=s.replace('- `external/qmf/`: separately declared QMF-E adaptation when supplied, with its own protocol and results. `external` refuses to claim a completed method when its portable runner is absent.','- `external/qmf/`: separately declared QMF-E energy-rule adaptation, frozen protocol, source-logit checks, complete prefix selections and four-task results. Energy coefficients remain unnormalized. Its README and `external/paper_provenance.md` state the exact unclipped logit reconstruction conditions and omitted end-to-end training components.')
s=s.replace('- `analysis/`: portable read-only denominator and accounting analysis when supplied; external cohorts remain separate from original cohorts.','- `analysis/`: portable read-only denominator, action, coverage and accounting analysis of all supplied cohorts. `coverage_analysis.json` includes both external adaptations; original cohorts remain separate. The combined ten-delay summary is descriptive.')
s=s.replace('After adding final external deliverables, the packager can refresh the delivery inventory','After changing delivery files, the packager can refresh the delivery inventory')
p.write_text(s)
print(json.dumps(dict(copy_manifest_required=len(copy_report),base_files=len(x),unified_pdf_and_qmf_passed=True)))
