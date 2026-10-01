"""Hash and archive the supplied revision, excluding caches and generated ZIPs."""
from pathlib import Path
import hashlib,json,zipfile
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
SKIP={'__pycache__','.pytest_cache','.DS_Store'}
def files():return sorted(p for p in ROOT.rglob('*') if p.is_file() and not (set(p.parts)&SKIP) and p.name!='artifact_manifest.json' and p.suffix not in {'.pyc','.zip'})
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
a=files()
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in a],'scope':'All supplied artifacts except caches and this manifest. Synthetic executed evidence; no raw Alibaba trace.'}
(ROOT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
a.append(ROOT/'artifact_manifest.json')
for filename,select in [('fusion_revision_complete.zip',lambda p:True),('performance_source_and_tables.zip',lambda p:p.parent==ROOT and p.suffix in {'.tex','.md'} or 'tables' in p.relative_to(ROOT).parts or 'figures' in p.relative_to(ROOT).parts)]:
 out=ROOT.parent/filename
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in a:
   if select(p):z.write(p,Path('revision')/p.relative_to(ROOT))
 with zipfile.ZipFile(out) as z:
  bad=z.testzip()
  if bad:raise ValueError('CRC check failed: '+bad)
 print(filename,out.stat().st_size,'bytes',sha(out),flush=True)
print(len(a),'files',sum(p.stat().st_size for p in a),'uncompressed bytes')
