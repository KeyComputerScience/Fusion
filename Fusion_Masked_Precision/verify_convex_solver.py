"""Independent mathematical checks of the current issued-value convex solver.

No dataset is loaded. Results are correctness checks, not physical utility.
"""
from pathlib import Path
import ast,hashlib,importlib.util,json
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'temporal_fusion.py'
source_text=SOURCE.read_text()
source_sha=hashlib.sha256(source_text.encode()).hexdigest()
tree=ast.parse(source_text)
function_hashes={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
                 for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('solve','norm_terms','capped')}
spec=importlib.util.spec_from_file_location('temporal_solver_under_check',SOURCE)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
rng=np.random.default_rng(410301)

def value_gradient_hessian(w,rate,R,P,q,quality,cfg):
    pi=quality**cfg['quality_power'];pi=pi/pi.sum()
    pn,pg,ph=mod.norm_terms(w,P,cfg['norm_floor'])
    rn,rg,rh=mod.norm_terms(w,R,cfg['norm_floor'])
    tau=cfg['tau'];kap=cfg['posterior_kappa']
    value=-w@rate+tau*np.sum(w*np.log(w/pi))+kap*pn+q*rn
    gradient=-rate+tau*(np.log(w/pi)+1)+kap*pg+q*rg
    hessian=np.diag(tau/w)+kap*ph+q*rh
    return float(value),gradient,hessian

def cfg(tau=.01,kap=1.):
    return dict(cap=.8,quality_power=1.,tau=tau,norm_floor=.01,posterior_kappa=kap)

def psd(m,scale):
    A=rng.normal(size=(m,m));return scale*(A@A.T/m+.02*np.eye(m))

finite=[];dense=[];multivariate=[];mean_only=[]
for i in range(32):
    m=2+i%4;R=psd(m,.02);P=psd(m,.003)
    rate=rng.normal(scale=.1,size=m);quality=np.exp(rng.normal(scale=.4,size=m))
    q=(0.,.1,1.,3.)[i%4];c=cfg(tau=(.01,.05)[i%2],kap=(0.,1.)[i%2])
    w=rng.uniform(.1,1,size=m);w/=w.sum()
    v,g,h=value_gradient_hessian(w,rate,R,P,q,quality,c)
    eps=1e-6;ng=[];nh=np.zeros((m,m))
    for j in range(m):
        d=np.zeros(m);d[j]=eps
        vp,gp,_=value_gradient_hessian(w+d,rate,R,P,q,quality,c)
        vm,gm,_=value_gradient_hessian(w-d,rate,R,P,q,quality,c)
        ng.append((vp-vm)/(2*eps));nh[:,j]=(gp-gm)/(2*eps)
    finite.append(dict(dimension=m,gradient_error=float(np.max(np.abs(g-np.asarray(ng)))),
                       hessian_error=float(np.max(np.abs(h-nh))),
                       minimum_hessian_eigenvalue=float(np.linalg.eigvalsh(h).min())))
    sol,cert,obj=mod.solve(rate,R,P,q,quality,c,1)
    independent=value_gradient_hessian(sol,rate,R,P,q,quality,c)[0]
    multivariate.append(dict(dimension=m,certificate=cert,
                             objective_error=abs(obj-independent)))

for i in range(32):
    R=psd(2,.02);P=psd(2,.003);quality=np.exp(rng.normal(scale=.7,size=2))
    rate=rng.normal(scale=.15,size=2);q=(0.,.1,1.,3.)[i%4]
    c=cfg(tau=(.01,.05)[i%2]);w,cert,obj=mod.solve(rate,R,P,q,quality,c,1)
    xs=np.linspace(.2,.8,12001);W=np.column_stack([xs,1-xs]);pi=quality/quality.sum()
    brute=-W@rate+c['tau']*np.sum(W*np.log(W/pi),axis=1)
    brute+=np.sqrt(np.einsum('ni,ij,nj->n',W,P,W)+c['norm_floor']**2)
    brute+=q*np.sqrt(np.einsum('ni,ij,nj->n',W,R,W)+c['norm_floor']**2)
    idx=int(np.argmin(brute))
    dense.append(dict(weights=w.tolist(),dense_weights=W[idx].tolist(),
                      dense_minus_solver=float(brute[idx]-obj),certificate=cert))

for i in range(16):
    m=2+i%4;R=psd(m,.02);P=psd(m,.003);quality=np.exp(rng.normal(scale=.4,size=m))
    rate=rng.normal(scale=.01,size=m);c=cfg(kap=0.)
    w,cert,obj=mod.solve(rate,R,P,0.,quality,c,1)
    pi=quality/quality.sum();log=np.log(pi)+rate/c['tau']
    expected=mod.capped(np.exp(log-log.max()),max(c['cap'],1/m))
    mean_only.append(dict(dimension=m,weight_error=float(np.max(np.abs(w-expected))),certificate=cert))

single=mod.solve(np.array([.1]),np.array([[.01]]),np.array([[.0025]]),.2,np.array([1.]),cfg(),1)

# Same means, quality, R and q; only posterior shape changes.
c=cfg(tau=.01);R=.01*np.eye(2);quality=np.ones(2);rate=np.zeros(2)
operational_q=1.2815515655446004
swap_a=mod.solve(rate,R,np.diag([.001,.005]),operational_q,quality,c,1)
swap_b=mod.solve(rate,R,np.diag([.005,.001]),operational_q,quality,c,1)
scalar=mod.solve(rate,R,.3*R,operational_q,quality,c,1)
R=.01*np.array([[1.,.8],[.8,1.]])
P=np.linalg.solve(100*np.eye(2)+np.linalg.solve(R,np.eye(2)),np.eye(2))
rate=np.array([.01,0.]);full=mod.solve(rate,R,P,operational_q,quality,c,1)
diag=mod.solve(rate,R,np.diag(np.diag(P)),operational_q,quality,c,1)
N=128.;fee=5.;shift=(full[2]+diag[2])/2+fee/N
full_gate=N*(shift-full[2])-fee;diag_gate=N*(shift-diag[2])-fee

cases=dict(mask_swap_a=dict(weights=swap_a[0].tolist(),certificate=swap_a[1]),
           mask_swap_b=dict(weights=swap_b[0].tolist(),certificate=swap_b[1]),
           scalar_same_mean_same_R_same_q=dict(weights=scalar[0].tolist(),certificate=scalar[1]),
           posterior_offdiagonal_only=dict(R=R.tolist(),P=P.tolist(),
               full_weights=full[0].tolist(),diagonal_P_weights=diag[0].tolist(),
               q=operational_q,tau=c['tau'],rate=rate.tolist(),common_mean_shift=shift,
               full_paid_gate=full_gate,diagonal_P_paid_gate=diag_gate,
               full_certificate=full[1],diagonal_certificate=diag[1]))

passed=(max(r['gradient_error'] for r in finite)<1e-7
        and max(r['hessian_error'] for r in finite)<3e-6
        and min(r['minimum_hessian_eigenvalue'] for r in finite)>0
        and all(r['certificate']['converged'] for r in multivariate+dense+mean_only)
        and max(r['objective_error'] for r in multivariate)<1e-12
        and min(r['dense_minus_solver'] for r in dense)>-1e-9
        and max(r['dense_minus_solver'] for r in dense)<2e-7
        and max(r['weight_error'] for r in mean_only)<2e-5
        and single[1]['converged']
        and abs(single[0][0]-1)<1e-14
        and swap_a[0][0]>.5 and swap_b[0][0]<.5
        and abs(scalar[0][0]-.5)<1e-12
        and full[0][0]>diag[0][0]>.5
        and full_gate<0<diag_gate)
report=dict(kind='INDEPENDENT_NUMERICAL_CORRECTNESS_AND_MATHEMATICAL_FIXTURES',passed=bool(passed),
    source_sha256_at_import=source_sha,function_ast_sha256=function_hashes,
    source_sha256_at_finish=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    deterministic_seed=410301,finite_difference=finite,multivariate_solvers=multivariate,
    dense_m2_comparisons=dense,mean_only_closed_form_checks=mean_only,
    singleton=dict(weights=single[0].tolist(),certificate=single[1]),literal_shape_interventions=cases,
    maxima=dict(gradient_error=max(r['gradient_error'] for r in finite),
                hessian_error=max(r['hessian_error'] for r in finite),
                dense_grid_objective_gap=max(r['dense_minus_solver'] for r in dense),
                kkt=max(r['certificate']['kkt'] for r in multivariate+dense+mean_only)))
(ROOT/'independent_solver_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(passed=passed,maxima=report['maxima'],shape=cases),indent=2))
assert passed
