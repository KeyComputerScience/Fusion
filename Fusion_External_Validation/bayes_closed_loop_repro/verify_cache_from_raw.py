"""Rebuild and verify exact engine-input caches from full official raw files.

Uses the byte-frozen adapters, with no cached-loader monkeypatch. Comparison is
array/dtype exact and includes all prefix/model fields. NPZ byte identity can also
be observed but is not required across compression-library versions.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import independent_bayes_fusion as engine
import independent_bayes_extension as adapter
from cached_runner import BASE_SHA,ADAPTER_SHA,cached_dataset

ROOT=Path(__file__).resolve().parent


def equal_recursive(a,b,path):
    if isinstance(a,np.ndarray):
        if not isinstance(b,np.ndarray) or a.dtype!=b.dtype or not np.array_equal(a,b):
            raise AssertionError('Array or dtype differs: '+path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('Metadata fields differ: '+path)
        for key in a:equal_recursive(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise AssertionError('Length differs: '+path)
        for i,(x,y) in enumerate(zip(a,b)):equal_recursive(x,y,path+f'[{i}]')
    elif a!=b:raise AssertionError('Value differs: '+path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw-data',type=Path,default=ROOT/'full_raw_data');parser.add_argument('--cache-data',type=Path,default=ROOT/'independent_data');parser.add_argument('--output',type=Path);parser.add_argument('--datasets',nargs='+',default=['mhealth319','har240'],choices=['mhealth319','har240']);parser.add_argument('--report',type=Path,default=ROOT/'cache_rebuild_verification.json');args=parser.parse_args()
    if hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest()!=BASE_SHA or hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest()!=ADAPTER_SHA:
        raise ValueError('Frozen scientific sources differ')
    protocol=json.loads((ROOT/'new_bayes_extension/protocol.json').read_text());checks=[]
    for name in args.datasets:
        spec=protocol['datasets'][name];raw_folder=args.raw_data/name
        if not raw_folder.is_dir():raise FileNotFoundError('Fetch/extract official raw data first: '+str(raw_folder))
        raw=adapter.load_mhealth(raw_folder,spec) if name=='mhealth319' else adapter.load_har(raw_folder,spec)
        reference=args.cache_data/name;metadata=json.loads((reference/'cache_metadata.json').read_text());cached=cached_dataset(reference,spec=spec)
        equal_recursive(raw,cached,'dataset.'+name)
        # Includes scaling, lag construction, fixed split, priors, qualities,
        # contexts, training buffers, and every initially fitted source/model.
        raw_prefix=engine.prefix(raw,spec,engine.BASE);cached_prefix=engine.prefix(cached,spec,engine.BASE)
        equal_recursive(raw_prefix,cached_prefix,'prefix.'+name)
        check=dict(dataset=name,consumed_raw_hashes_exact=True,all_array_dtypes_and_values_exact=True,all_prefix_fields_and_models_exact=True,max_float_difference=0.,reference_cache_sha256=metadata['cache_sha256'],retained_rows=raw['n'],original_raw_rows=raw['raw_rows'])
        if args.output is not None:
            target=args.output/name
            if target.resolve()==reference.resolve():raise ValueError('Choose a fresh output folder, or omit --output to verify only')
            target.mkdir(parents=True,exist_ok=True);npz=target/'cached_dataset.npz'
            np.savez_compressed(npz,raw=np.asarray(raw['raw'],dtype=np.float64),y=raw['y'],timestamp=np.asarray(raw['timestamp']),subject=raw['subject'])
            fresh=dict(metadata,cache_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),cache_bytes=npz.stat().st_size,reference_cache_sha256=metadata['cache_sha256'])
            (target/'cache_metadata.json').write_text(json.dumps(fresh,indent=2))
            equal_recursive(raw,cached_dataset(target,spec=spec),'fresh_cache.'+name)
            check.update(rebuilt_cache_sha256=fresh['cache_sha256'],rebuilt_cache_bytes=fresh['cache_bytes'],npz_byte_hash_equal=fresh['cache_sha256']==metadata['cache_sha256'])
        checks.append(check);print('RAW_CACHE_REBUILD_VERIFIED',name,flush=True)
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(dict(all_passed=True,base_sha=BASE_SHA,adapter_sha=ADAPTER_SHA,numpy_version=np.__version__,checks=checks),indent=2))

if __name__=='__main__':main()
