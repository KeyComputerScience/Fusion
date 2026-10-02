"""Package exact pre-prefix wearable engine inputs and verify prefix equality."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import independent_bayes_fusion as engine
import independent_bayes_extension as adapter

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'outputs'/'bayes_closed_loop_repro'
sys.path.insert(0,str(OUT))
from cached_runner import cached_dataset


def equal_recursive(a,b,path='prefix'):
    if isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal_recursive(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal_recursive(x,y,path+f'[{i}]')
    else:assert a==b,path


def main():
    protocol=json.loads((ROOT/'new_bayes_extension'/'protocol.json').read_text());records=[]
    for name,spec in protocol['datasets'].items():
        source=ROOT/'independent_data'/name
        data=adapter.load_mhealth(source,spec) if name=='mhealth319' else adapter.load_har(source,spec)
        folder=OUT/'independent_data'/name;folder.mkdir(parents=True,exist_ok=True);cache=folder/'cached_dataset.npz'
        np.savez_compressed(cache,raw=np.asarray(data['raw'],dtype=np.float64),y=data['y'],timestamp=np.asarray(data['timestamp']),subject=data['subject'])
        meta=dict(format_version=1,dataset=name,n=data['n'],raw_rows=data['raw_rows'],participants=data['participants'],consumed_raw_hashes=data['hashes'],cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),cache_bytes=cache.stat().st_size,raw_shape=list(data['raw'].shape),raw_dtype=str(data['raw'].dtype),label_dtype=str(data['y'].dtype),origin='Official UCI raw archive, SHA256 verified; frozen unchanged adapter derivation',derivation=spec,cache_scope='Lossless arrays returned by frozen loader BEFORE prefix scaling/lagging/model fitting. Full original sensor logs and HAR acquisition segments are not embedded; prepare_raw_data.py fetches them.',license='CC BY 4.0',doi='10.24432/C5TW22' if name=='mhealth319' else '10.24432/C54S4K',frozen_engine_sha=hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),frozen_adapter_sha=hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest())
        (folder/'cache_metadata.json').write_text(json.dumps(meta,indent=2))
        cached=cached_dataset(folder);equal_recursive(data,cached,'dataset')
        original_prefix=engine.prefix(data,spec,engine.BASE);cached_prefix=engine.prefix(cached,spec,engine.BASE);equal_recursive(original_prefix,cached_prefix)
        records.append(dict(dataset=name,dataset_arrays_and_metadata_exact=True,all_prefix_fields_exact=True,all_initial_models_exact=True,max_float_difference=0.,cache_sha256=meta['cache_sha256'],cache_bytes=meta['cache_bytes'],n=data['n'],raw_rows=data['raw_rows'],raw_shape=meta['raw_shape']))
        print('EXACT_CACHE',records[-1],flush=True)
    report=dict(all_passed=True,kind='exact cached loader and all prefix/model fields verification',datasets=records)
    (OUT/'cache_verification.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
