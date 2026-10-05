"""Portable paths for byte-identical scientific sources and read-only packages.

Only top-level filesystem bindings are replaced in memory using Python AST.
No update, fusion, optimization, gradient, score, action or accounting function
is transformed. Exact source bytes are verified before execution.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,os,sys,types
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
ORIGINAL=Path(json.loads((ROOT/'DEPENDENCIES.json').read_text())['original_project'])
FRESH=ROOT/'work/fusion_strengthening_20261003/fresh'
MATRIX=ROOT/'work/fusion_strengthening_20261003/matrix'
EXTERNAL=ROOT/'work/fusion_strengthening_20261003/external'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def resolve(p):
 p=Path(p)
 try:r=p.relative_to(ORIGINAL)
 except ValueError:return p
 if r.parts[0]=='outputs':return ROOT.parent/Path(*r.parts[1:])
 return ROOT/r

def dependencies():
 spec=load(ROOT/'DEPENDENCIES.json');report={}
 for name,entry in spec['packages'].items():
  base=ROOT.parent/name;manifest=base/'MANIFEST.json'
  if not manifest.exists():raise FileNotFoundError(f'Place the unchanged {name} directory next to this bundle.')
  assert sha(manifest)==entry['manifest_sha256'],f'Dependency manifest differs: {name}'
  m=load(manifest);bad=[p for p,h in m['files'].items() if not (base/p).exists() or sha(base/p)!=h]
  assert not bad,(name,bad);report[name]=dict(files=len(m['files']),unchanged=True)
 return report

def verify():
 manifest=load(ROOT/'MANIFEST.json');bad=[p for p,h in manifest['files'].items() if not (ROOT/p).exists() or sha(ROOT/p)!=h];assert not bad,bad
 dep=dependencies();core=ROOT/'work/fusion_temporal_20261003/temporal_fusion.py';assert sha(core)=='811f98531ad3e135ceafbc64152b41ffbf380878cb45a1f7106eb54364157ff0'
 matrix=load(MATRIX/'protocol.json')
 for p,h in matrix['frozen_input_sha256'].items():assert sha(resolve(p))==h,p
 factor=load(MATRIX/'factorized/protocol.json')
 for p,h in factor['source_hashes'].items():assert sha(resolve(p))==h,p
 for task in ('arem366','gashome362'):
  meta=load(FRESH/'data'/task/'cache_metadata.json');assert sha(FRESH/'data'/task/'cached_dataset.npz')==meta['cache_sha256']
  v=load(FRESH/'evaluation'/f'{task}_verification.json');assert v['passed'] and v['core_unchanged'] and v['max_accounting_error']<1e-8
 return dict(passed=True,own_files=len(manifest['files']),dependencies=dep,original_core_unchanged=True,scientific_source_rewriting=False,filesystem_ast_bindings_only=True)

class Bindings(ast.NodeTransformer):
 def __init__(self,bindings):self.bindings=bindings
 def visit_Assign(self,node):
  if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in self.bindings:
   name=node.targets[0].id;node.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(self.bindings[name]))],keywords=[])
  return node

def import_bound(name,path,bindings=None):
 tree=ast.parse(Path(path).read_text(),filename=str(path));tree=Bindings(bindings or {}).visit(tree);ast.fix_missing_locations(tree)
 m=types.ModuleType(name);m.__file__=str(path);m.__package__='';sys.modules[name]=m;exec(compile(tree,str(path),'exec'),m.__dict__);return m

def support():
 sys.path.insert(0,str(ROOT/'work/fusion_focus_20261003/controls'))
 import run_strong_controls as s
 s.DEFAULT_PACKAGE=ROOT.parent/'Fusion_Recovery_Repro';s.DEFAULT_GUARD=ROOT.parent/'Fusion_Loss_Budget_Extension_Repro/guard'
 return s

def output_dir(p):
 p=Path(p).resolve()
 if p==ROOT or ROOT in p.parents:raise ValueError('Use a new output directory outside the delivered bundle.')
 p.mkdir(parents=True,exist_ok=True);return p

def fresh_run(output,phase,tasks):
 support();spec=importlib.util.spec_from_file_location('portable_fresh_execution',FRESH/'run_fresh.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=output_dir(output)
 for step in (('calibrate','test','guard') if phase=='all' else (phase,)):
  sys.argv=[str(FRESH/'run_fresh.py'),step,'--tasks',*tasks];m.main()


def neural_module():
 support();helper=import_bound('run_trained_external',ROOT/'work/fusion_focus_20261003/external_author/run_trained_external.py',{'PROJECT':ROOT,'PACKAGE':ROOT.parent/'Fusion_Recovery_Repro'})
 nn=import_bound('neural_external',EXTERNAL/'neural_external.py',{'PROJECT':ROOT});return nn

def neural_run(output,phase,tasks):
 nn=neural_module();out=output_dir(output)
 if phase in ('all','check'):
  if not (out/'declaration.json').exists():sys.argv=[nn.__file__,'--phase','declare','--output',str(out)];nn.main()
  sys.argv=[nn.__file__,'--phase','check','--output',str(out)];nn.main()
  if phase=='check':return
 for task in tasks:
  adapter=FRESH/'new_data_adapter.py' if task!='rss348' else ROOT/'work/fusion_focus_20261003/fresh/new_data_adapter.py'
  data=FRESH/'data' if task!='rss348' else ROOT/'work/fusion_temporal_20261003/physical/data'
  protocol=EXTERNAL/f'{task}_protocol.json' if task!='rss348' else ROOT/'work/fusion_temporal_20261003/rss_validation/protocol.json'
  if task=='rss348':
   protocol=EXTERNAL/'rss348_protocol.json'
  for step in (('calibrate','test') if phase=='all' else (phase,)):
   sys.argv=[nn.__file__,'--phase',step,'--output',str(out),'--task',task,'--adapter',str(adapter),'--data',str(data),'--task-protocol',str(protocol)];nn.main()


class AuditPaths(ast.NodeTransformer):
 def __init__(self,out):self.out=out
 def visit_FunctionDef(self,node):
  if node.name=='sha':
   argument=node.args.args[0].arg
   node.body=[ast.Return(ast.Call(func=ast.Name(id='portable_sha',ctx=ast.Load()),args=[ast.Name(id=argument,ctx=ast.Load())],keywords=[]))]
  return node
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if isinstance(node.func,ast.Attribute) and node.func.attr=='write_text':
   value=node.func.value
   if isinstance(value,ast.BinOp) and isinstance(value.right,ast.Constant) and isinstance(value.right.value,str):
    node.func.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(self.out/value.right.value))],keywords=[])
  return node

def audit_matrix(output):
 out=output_dir(output)
 for source in [MATRIX/'verify_matrix_results.py',MATRIX/'factorized/verify_factorized_results.py']:
  tree=ast.parse(source.read_text(),filename=str(source));tree=AuditPaths(out).visit(tree);ast.fix_missing_locations(tree)
  namespace={'__file__':str(source),'__name__':'portable_matrix_audit','portable_sha':lambda p:sha(resolve(p))}
  exec(compile(tree,str(source),'exec'),namespace)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['verify','analyze','fresh','neural','audit-matrix']);ap.add_argument('--output',type=Path);ap.add_argument('--phase',default='all',choices=['all','calibrate','test','guard','check']);ap.add_argument('--tasks',nargs='+');args=ap.parse_args();assert sys.version_info>=(3,10)
 if args.command=='verify':print(json.dumps(verify(),indent=2));return
 if args.command=='analyze':
  print(json.dumps(load(FRESH/'fresh_validation_summary.json'),indent=2));return
 verify();os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
 if args.output is None:raise ValueError('--output is required for execution.')
 if args.command=='audit-matrix':audit_matrix(args.output)
 elif args.command=='fresh':fresh_run(args.output,args.phase,args.tasks or ['arem366','gashome362'])
 elif args.command=='neural':neural_run(args.output,args.phase,args.tasks or ['rss348','arem366','gashome362'])
if __name__=='__main__':main()
