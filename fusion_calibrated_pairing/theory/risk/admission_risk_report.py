"""Post-decision risk reports, with no change to scored admission actions."""
import math
LAMBDAS=(.25,.5,1.,2.,4.,8.)
ENDPOINTS=('violation','harmful','normalized_excess','normalized_negative_loss')
def upper_endpoint(total,n,error,lambdas=LAMBDAS):
    if n==0:return None
    if not 0<=total<=n or not 0<error<1:raise ValueError('Invalid bounded endpoint report')
    penalty=math.log(len(lambdas)/error)
    return min(1.,min((lam*total+penalty)/(-math.expm1(-lam)*n) for lam in lambdas))
def actual_prefix_risk(guarded_rows,now=None,global_error=.05,policies=5,reported_chains=60):
    """Familywise envelope on the longest mature admitted origin prefix.

    reported_chains counts ALL reported replay chains for error allocation;
    it is not a count of independent physical environments.
    """
    if policies<1 or reported_chains<1:raise ValueError('Declared report family required')
    acts=sorted((r for r in guarded_rows if r['action']),key=lambda r:r['k'])
    if now is None:now=max((r['maturity'] for r in acts),default=0)
    complete=[]
    for r in acts:
        if r['maturity']>now:break
        complete.append(r)
    n=len(complete);pending=len(acts)-n
    totals=dict(violation=0.,harmful=0.,normalized_excess=0.,normalized_negative_loss=0.)
    for r in complete:
        X=float(r['truegross']-5.);D=float(r['local_net']);L=float(r['gate_score'])
        if not 0<L<=123.+1e-10 or X< -133.-1e-10 or D< -130.-1e-10:raise ValueError('Admitted endpoint support outside declared contract')
        totals['violation']+=float(X<L);totals['harmful']+=float(D<0)
        totals['normalized_excess']+=max(0.,L-X)/256.
        totals['normalized_negative_loss']+=max(0.,-D)/130.
    error=global_error/(policies*reported_chains*len(ENDPOINTS))
    bounds={key:upper_endpoint(v,n,error) for key,v in totals.items()}
    all_bounds={key:min(1.,(n*b+pending)/len(acts)) if b is not None and acts else (1. if acts else None) for key,b in bounds.items()}
    return dict(admitted_origins=len(acts),complete_origin_prefix=n,pending_after_prefix=pending,totals=totals,average_conditional_risk_upper=bounds,issued_prefix_average_conditional_risk_upper=all_bounds,global_error=global_error,policies=policies,reported_replay_chains=reported_chains,error_per_endpoint=error,lambda_grid=LAMBDAS,nominal10percent_supported=bool(n and bounds['violation']<=.1),scope='Average conditional endpoint risk of observed predictable admission prefixes. No next-environment or individual conditional-coverage claim. Delay repeats do not add independent physical sites.')
