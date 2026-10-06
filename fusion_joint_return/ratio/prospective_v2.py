"""Single-posterior ratio family prepared before unused physical acquisition.

The old RSS/gas protocol and selections remain unchanged. Version 2 changes
only the predeclared calibration tie-break: retain the largest paired
information exponent when paid calibration return is tied. This preference
comes from known-data design, never a new physical test outcome.
"""
from pathlib import Path
import datetime,json,sys
sys.dont_write_bytecode=True
import run_ratio as r
api=r.api;np=r.np
ARMS=r.ARMS

def configurations():return r.configurations()

def attach_native(stream,native_result):
    """Attach original native-issued F,S,q to already-built decision states.

    native_result is the unchanged native run dictionary with rows. Its q
    callbacks must use only fully matured outcomes. This helper neither fits
    q nor reads truegross to form the law. All ratio arms share these rows.
    """
    rows=native_result['rows'] if isinstance(native_result,dict) else native_result
    assert len(stream['decisions'])==len(rows)
    for d,row in zip(stream['decisions'],rows):
        assert d['k']==row['k'];assert row['posterior_or_block_sd']>0
        d['native']=row
        d['loganchor']=-.5*((r.U-row['gain']/d['N'])/(row['posterior_or_block_sd']/d['N']))**2
        d['prob']=r.alpha_from_q(row['q_issued'])
        d['ratios']={}
    return stream

def decorate_ratios(stream,pre,cfg):
    """State must contain paired/factorized atoms, masked means and moments.

    Use the same cumulant_state as the old prototype. cfg carries the fixed
    native-calibrated prior_sd and the common historical/context settings.
    Every trial kernel is also used by its matched full-moment denominator.
    """
    for bandwidth in r.BANDS:
        cc=dict(cfg,slice_bandwidth=bandwidth)
        for d in stream['decisions']:d['ratios'][bandwidth]=r.density_logs(d,pre,cc)
    return stream

def issue(stream,arm,conf,selection=False):
    assert arm in ARMS
    assert conf in configurations()
    return r.issue(stream,arm,conf,selection)

def select(grid):
    """Uniform paid-return selection; the same tie preference in ALL arms."""
    for arm in ARMS:
        entries=[entry for entry in grid if entry['arm']==arm]
        assert len(entries)==9,(arm,len(entries))
        assert {tuple(sorted(x['config'].items())) for x in entries}=={tuple(sorted(x.items())) for x in configurations()}
    return {arm:max((x for x in grid if x['arm']==arm),key=lambda x:(x['net'],x['config']['zeta'],x['config']['bandwidth'])) for arm in ARMS}

def summary(guarded,seeds,arms=ARMS):
    """Coverage uses actual guarded admissions, not pre-guard proposals."""
    out={}
    for arm in arms:
        gs=[g for g in guarded if g['arm']==arm];rows=[row for g in gs for row in g['rows']]
        groups=dict(issued=rows,ready=[row for row in rows if row['information_ready']],informative=[row for row in rows if row['disagreement']>0],admitted=[row for row in rows if row['action']])
        acts=groups['admitted'];excess=[max(0.,row['gate_score']-(row['truegross']-5.)) for row in acts]
        increments=[sum(g['increment'] for g in gs if g['seed']==seed) for seed in seeds]
        out[arm]=dict(seed_increments=increments,mean_increment=float(np.mean(increments)),admissions=len(acts),beneficial=sum(row['local_net']>0 for row in acts),harmful=sum(row['local_net']<0 for row in acts),zero=sum(row['local_net']==0 for row in acts),negative_loss=sum(max(0.,-row['local_net']) for row in acts),coverage={key:[sum(row['lower_covered'] for row in values),len(values)] for key,values in groups.items()},admitted_max_excess=max(excess,default=0.),admitted_total_excess=sum(excess),admitted_mean_excess=float(np.mean(excess)) if acts else None,refused=sum(g['refused'] for g in gs))
    return out

def scientific_sources():
    sources={Path(__file__).resolve(),Path(r.__file__).resolve()}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename and str(filename).startswith(str(r.PROJECT)) and str(filename).endswith('.py'):sources.add(Path(filename).resolve())
    return sorted(sources)

def freeze_definition(path,unread_raw_paths=(),extra_sources=()):
    """Write before new raw acquisition; caller freezes physical adapter too.

    Existing raw paths cause rejection rather than a false prospective label.
    May be called with no paths to freeze a method definition only; acquisition
    status then remains unconfirmed until the runner's complete protocol.
    """
    path=Path(path);assert not path.exists()
    for raw in unread_raw_paths:assert not Path(raw).exists(),f'Raw path already exists: {raw}'
    sources=scientific_sources()+[Path(x).resolve() for x in extra_sources]
    protocol=dict(version=2,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='Known-data method design; numerical operator and largest-exponent tie rule frozen for subsequent physical validation. Acquisition status is established only by the complete runner protocol.',
        arms=ARMS,grids={arm:configurations() for arm in ARMS},additional_configs_per_arm=9,additional_calibration_schedules_per_config=3,
        native_stage='one strongest native factorized anchor selected only from its nine calibration configs and same three delays; this common first-stage search is charged equally to every ratio arm; all shared native candidate trial logs retained',
        selection='maximize independently reconstructed guarded secondhalf paid calibration return; ties largest exponent, then largest bandwidth; identical in every arm',
        density=r.load('protocol.json')['operator'],calibration='original native all-issued delayed q, including its causal q-dependent optimizer, is fixed as shared issuance information; the new ratio score is not automatically calibrated by the anchor',
        full_feedback='all complete mature potential lease outcomes shared across arms; actual admission strata additionally evaluated on guarded executed actions',
        quadrature='2049 fixed uniform nodes, trapezoidal density CDF/first moment and interpolated lower expected shortfall; no inferred statistical coverage radius',
        service='same paid score subtracts5, complete four-window leases, fixed130 reserve and B130 loss guard; all controls share guard',
        unread_raw_paths=[str(x) for x in unread_raw_paths],source_hashes={str(p):api.sha(p) for p in sorted(set(sources))},known_data_design='RSS/gas primary prototype selected exponent0; diagnostic all-exponent grid retained. V2 tie preference was chosen before the subsequent unused physical data, but after those known-data results.',
        coverage_outputs='all-issued/ready/informative/actual-admitted exact numerators/denominators; excess max/sum/mean, harmful lease count and negative loss, guard refusals; N/A when no actual admission')
    api.dump(path,protocol)
    return protocol
