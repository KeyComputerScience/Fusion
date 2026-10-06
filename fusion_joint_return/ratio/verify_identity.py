from pathlib import Path
import sys,math,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_ratio as r
np=r.np
r.U=np.linspace(-1.,1.,131073);u=r.U
t=np.sin(4*u)+.3*u*u;logbase=-.5*((u-.18)/.24)**2
z=.4;alpha=.23;eps=1e-5
log=logbase+z*t;w=np.exp(log-log.max());mass=np.trapezoid(w,u);p=w/mass
cdf=np.r_[0.,np.cumsum((p[:-1]+p[1:])/2*np.diff(u))];quantile=float(np.interp(alpha,cdf,u))
b=np.maximum(0.,quantile-u)
cov=float(np.trapezoid(b*t*p,u)-np.trapezoid(b*p,u)*np.trapezoid(t*p,u))
finite=(r.density_tail(logbase+(z+eps)*t,alpha)-r.density_tail(logbase+(z-eps)*t,alpha))/(2*eps)
error=abs(finite+cov/alpha)
assert error<2e-5,error
normal_checks=[]
for q in (0.,.25,1.,2.):
    a=r.alpha_from_q(q)
    x=-.5*((u-.1)/.08)**2
    got=r.density_tail(x,a);want=.1-q*.08
    normal_checks.append(dict(q=q,alpha=a,tail=got,unbounded_tail=want,error=abs(got-want)))
assert max(x['error'] for x in normal_checks)<1e-6
out=dict(derivative_error=error,finite_derivative=finite,theoretical_derivative=-cov/alpha,normal_mapping=normal_checks,grid=len(u))
(Path(__file__).resolve().parent/'identity_checks.json').write_text(json.dumps(out,indent=2))
print(out)
