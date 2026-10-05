"""Portable runner supplying lossless cached engine-input datasets.

Frozen algorithm files are never edited. Runtime loader injection changes only
wearable input I/O; cached raw features/labels/order are identical to the original
adapters. Occupancy CSV/TXT inputs remain the supplied original files.
"""
from __future__ import annotations
import hashlib,importlib, json, os, runpy, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE_SHA='15e3fc99d3f0935d2e66853b9b1949ef7825869c2f0d56a8ba49513838376e57'
ADAPTER_SHA='c70141db94d36b06079e89bd69e7a84c240938d2e40977cb807078cdb6f0253f'


def cached_dataset(root,name=None,spec=None):
    root=Path(root);folder=root/name if name else root
    path=folder/'cached_dataset.npz';meta=json.loads((folder/'cache_metadata.json').read_text())
    if spec is not None and spec!=meta['derivation']:
        raise ValueError('Cached derivation differs from requested frozen dataset specification')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['cache_sha256']:
        raise ValueError('Cached dataset checksum failed: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        arrays={key:z[key].copy() for key in ('raw','y','timestamp','subject')}
    assert arrays['raw'].dtype==np.float64
    assert len(arrays['raw'])==len(arrays['y'])==len(arrays['timestamp'])==len(arrays['subject'])==meta['n']
    return dict(raw=arrays['raw'],y=arrays['y'],timestamp=arrays['timestamp'].tolist(),subject=arrays['subject'],
                hashes=meta['consumed_raw_hashes'],n=meta['n'],raw_rows=meta['raw_rows'],participants=meta['participants'])


def install_cached_loaders():
    # __file__ and both source SHA values remain those of the frozen originals.
    for filename,sha in (('independent_bayes_fusion.py',BASE_SHA),('independent_bayes_extension.py',ADAPTER_SHA)):
        if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen executable checksum failed: '+filename)
    engine=importlib.import_module('independent_bayes_fusion')
    adapter=importlib.import_module('independent_bayes_extension')
    raw_mhealth=adapter.load_mhealth;raw_har=adapter.load_har
    def load_mhealth(root,spec):
        return cached_dataset(root,spec=spec) if (Path(root)/'cached_dataset.npz').exists() else raw_mhealth(root,spec)
    def load_har(root,spec):
        return cached_dataset(root,spec=spec) if (Path(root)/'cached_dataset.npz').exists() else raw_har(root,spec)
    adapter.load_mhealth=load_mhealth;adapter.load_har=load_har
    return engine,adapter


def main():
    if len(sys.argv)<2:
        raise SystemExit('Usage: python cached_runner.py SCRIPT.py [script arguments]')
    target=Path(sys.argv[1]);args=sys.argv[2:]
    if target.is_absolute() or target.name!=str(target) or not (ROOT/target).is_file():
        raise SystemExit('SCRIPT.py must be a script filename in this package')
    os.chdir(ROOT);sys.path.insert(0,str(ROOT));engine,adapter=install_cached_loaders()
    sys.argv=[str(ROOT/target),*args]
    # Running these two modules again through runpy would redefine their loaders.
    # Calling the already-imported main preserves injection and frozen __file__.
    if target.name=='independent_bayes_extension.py':adapter.main()
    elif target.name=='independent_bayes_fusion.py':engine.main()
    else:runpy.run_path(str(ROOT/target),run_name='__main__')

if __name__=='__main__':main()
