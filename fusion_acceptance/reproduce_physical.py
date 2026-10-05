"""Fresh-output gas replay; uses the original frozen project source closure."""
from pathlib import Path
import argparse, importlib.util, json, shutil, sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parent
ORIGINAL = SOURCE/'physical'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('init','verify','calibrate','test','compare'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output == SOURCE or SOURCE in output.parents:
        parser.error('Choose a fresh output directory outside the research archive.')
    spec = importlib.util.spec_from_file_location('fresh_flow308_runner', ORIGINAL/'run_physical.py')
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    api.verify()
    if args.phase == 'init':
        if output.exists():
            parser.error('The output directory must not exist.')
        output.mkdir(parents=True)
        for name in ('protocol.json','pre_acquisition_freeze.json','algorithm_choice.json'):
            shutil.copy2(ORIGINAL/name,output/name)
        shutil.copytree(ORIGINAL/'data',output/'data')
        print('Initialized fresh output; original frozen project bindings remain unchanged.')
        return
    if not (output/'pre_acquisition_freeze.json').exists():
        parser.error('Run init first.')
    # Installing the operator remains bound to its frozen original path;
    # only protocol/output/cache lookup moves to the new output directory.
    original_install = api.install
    def install():
        newroot = api.ROOT
        try:
            api.ROOT = ORIGINAL
            original_install()
        finally:
            api.ROOT = newroot
    api.install = install
    api.ROOT = output
    api.verify()
    if args.phase == 'verify':
        meta = api.load('data/cache_metadata.json')
        assert api.sha(output/'data/cached_dataset.npz') == meta['cache_sha256']
        print('Original source hashes, protocol and copied physical cache verified.')
    elif args.phase == 'calibrate':
        api.calibration()
    elif args.phase == 'test':
        api.test()
    else:
        selected = api.load('selection.json')['selected']
        original_selected = json.loads((ORIGINAL/'selection.json').read_text())['selected']
        assert selected == original_selected
        fresh = api.load('results.json.gz')
        reference = api.api.load(ORIGINAL/'results.json.gz')
        for key in ('issued','guarded','budget_rows'):
            assert fresh[key] == reference[key], key
        report = dict(selected_parameters='exact', issued_scores_and_callbacks='exact',
                      guarded_actions_and_service='exact', all_budget_paths='exact',
                      scope='same source/data closure; fresh output, calibration and test processes; no new efficacy observations')
        (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
