from pathlib import Path
import json, subprocess, sys, tempfile, zipfile
R=Path('/Users/key/Documents/Codex/2026-10-02/jih'); archive=R/'outputs/Fusion_External_Validation_Repro.zip'
folder=Path(tempfile.mkdtemp(prefix='fusion_final_zip_portable_',dir='/private/tmp'))
with zipfile.ZipFile(archive) as z: z.extractall(folder)
bundle=folder/'Fusion_External_Validation_Repro'; runs=folder/'fresh_analysis'
commands=[['verify'],['analyze-external','--output',str(runs)]]
checks=[]
for args in commands:
    p=subprocess.run([sys.executable,str(bundle/'run_reproduction.py'),*args],cwd='/private/tmp',text=True,capture_output=True)
    checks.append(dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    assert p.returncode==0,p.stdout+p.stderr
analysis=json.loads((runs/'analysis/coverage_analysis.json').read_text())
report_path=archive.with_suffix(archive.suffix+'.verification.json')
report=json.loads(report_path.read_text())
report['extracted_zip_relocation_verified']=True
report['extracted_bundle_path']=str(bundle)
report['relocated_entry_checks']=checks
report['relocated_analysis_original_arm_seed_logs']=analysis['original_arm_seed_log_audits']
report['relocated_analysis_external_methods']=list(analysis['external'])
report['note']='Relocation checks use the archive-extracted files without modifying them; full original/external computational reruns are separately retained in the delivered verification records.'
report_path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(passed=True,extracted_bundle=str(bundle),original_logs=analysis['original_arm_seed_log_audits'],external_methods=list(analysis['external']))))
