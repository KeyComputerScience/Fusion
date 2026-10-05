"""Portable entry point for the separately frozen diagnostic replay runner.

The original run_extension.py is hash-checked and never modified. Runtime path
globals select the supplied unchanged package and declaration/output directory.
No model, split, controller configuration, seed, or calibration is changed.
"""
from __future__ import annotations
import argparse,datetime,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
FROZEN_RUNNER_SHA='07e27b7d0a394aebe0117e8a92bc722c9de8ed5eb78b386c3c8689a29b28583b'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    candidates=(HERE.parent/'bayes_closed_loop_repro',HERE.parent.parent/'outputs'/'bayes_closed_loop_repro')
    default=next((p for p in candidates if p.is_dir()),candidates[0])
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,default=default,help='Unchanged bayes_closed_loop_repro package');parser.add_argument('--output',type=Path,default=HERE,help='Replay output directory; declaration copied unchanged when new');args=parser.parse_args()
    package=args.package.resolve();out=args.output.resolve();runner=HERE/'run_extension.py'
    if digest(runner)!=FROZEN_RUNNER_SHA:raise SystemExit('Frozen original runner SHA mismatch')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('protocol.json','pre_replay_freeze.json','original_files_before.json'):
        source=HERE/name;destination=out/name
        if destination.exists():
            if digest(destination)!=digest(source):raise SystemExit('Different declaration already exists in --output: '+name)
        elif source.resolve()!=destination.resolve():shutil.copyfile(source,destination)
    sys.path.insert(0,str(package))
    # Prime the import namespace from the explicit package before loading the
    # byte-frozen runner whose original default path is workspace-relative.
    import cached_runner
    if Path(cached_runner.__file__).resolve()!=package/'cached_runner.py':raise SystemExit('Wrong cached_runner import path')
    cached_runner.install_cached_loaders()
    spec=importlib.util.spec_from_file_location('frozen_delay_replay_runner',runner);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=out;module.PACKAGE=package
    (out/'portable_launch_manifest.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),package=str(package),output=str(out),frozen_runner_sha256=FROZEN_RUNNER_SHA,launcher_sha256=digest(__file__),runtime_change='Paths and output location only; frozen execution source and protocol unchanged'),indent=2))
    module.main()

if __name__=='__main__':main()
