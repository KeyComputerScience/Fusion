"""Reconstruct descriptive matrices after frozen execution; changes no policy."""
from pathlib import Path
import json
import numpy as np
import independent_bayes_fusion as engine
import independent_bayes_extension as adapter
ROOT=Path(__file__).resolve().parent
protocol=json.loads((ROOT/'new_bayes_extension/protocol.json').read_text());results=json.loads((ROOT/'new_bayes_extension/results.json').read_text());out={}
for name,study in results.items():
    spec=protocol['datasets'][name];data=adapter.load_mhealth(ROOT/'independent_data'/name,spec) if name=='mhealth319' else adapter.load_har(ROOT/'independent_data'/name,spec);pre=engine.prefix(data,spec,engine.BASE);rows=[];maxerror=0.
    for t in study['trials']:
        stream=engine.precompute(engine.world(pre['test_x'],pre['test_y'],pre['test_timestamp'],pre,t['seed'],engine.BASE),pre,engine.BASE)
        for d in stream['decisions']:
            row=dict(seed=t['seed'],k=d['k'],ids=d['ids'].tolist(),Sigma_scaled=d['S'].tolist(),posterior_U=d['U'].tolist(),between_B=d['B'].tolist(),mu=d['mu'].tolist(),N=d['N'],mass=d['mass'],tangent_scale=d['scale'])
            for mode in ('joint','bayes_weights','bayes_both'):
                cfg=study['selected'][mode];weights,cert=engine.convex_fuse(pre['q'][d['ids']],d['S'],d['Us'],cfg,mode);saved=next(r for r in t['results'][mode]['rows'] if r['k']==d['k']);maxerror=max(maxerror,float(np.max(np.abs(weights-np.array(saved['weights'])))))
            assert np.allclose(d['B'],(d['mass']+1)*d['U'],rtol=1e-12,atol=1e-12)
            row['posterior_identity_error']=float(np.max(np.abs(d['B']-(d['mass']+1)*d['U'])));rows.append(row)
    out[name]=dict(weight_reconstruction_error=maxerror,rows=rows)
(ROOT/'new_bayes_extension/matrix_diagnostics.json').write_text(json.dumps(out,indent=2));print({k:v['weight_reconstruction_error'] for k,v in out.items()})
