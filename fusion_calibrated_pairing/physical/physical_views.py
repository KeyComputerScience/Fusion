"""Predictable physical availability and a nonempty-preserving NN wrapper.

All candidate/reference work and lease truth are computed by the common
service engine. Missing physical forecasts are applied only to the source
views. A source is absent if every raw coordinate is missing on every
request row of the current window. Partial missing inputs retain their
fit-median and missing-indicator features.
"""
import numpy as np

def forecast_events(events,record,cfg):
    available=np.asarray(record['physical_source_present'],bool)
    if available.shape!=(len(record['y']),len(events[0]['mask'])):
        raise ValueError('Unaligned physical availability')
    out=[]
    for k,event in enumerate(events):
        start=k*cfg['window'];stop=start+len(event['y'])
        physical=available[start:stop].any(axis=0)
        mask=np.asarray(event['mask'],bool)&physical
        out.append(dict(event,mask=mask,physical_available=physical,
            physical_all_absent=bool(not physical.any()),empty_forecast_mask=bool(not mask.any())))
    return out

def neural_world(nn,events,pre,rule,lr,initial):
    # The original learner uses masks only for fusion, not parameter
    # updates. A temporary nonempty mask avoids its zero-length max call;
    # all training and every originally nonempty forecast are unchanged.
    temporary=[dict(event,mask=np.ones(pre['m'],bool)) if not event['mask'].any() else event for event in events]
    learned=nn.world(temporary,pre,rule,lr,initial)
    out=[]
    for original,row in zip(events,learned):
        if original['mask'].any():
            out.append(dict(row,physical_available=original['physical_available'],
                physical_all_absent=original['physical_all_absent'],empty_forecast_mask=False))
        else:
            probability=nn.helper.sm(original['x']@original['reference'])
            out.append(dict(row,mask=original['mask'].copy(),external_probability=probability,
                external_weights=np.zeros((len(original['y']),0)),
                physical_available=original['physical_available'],
                physical_all_absent=original['physical_all_absent'],empty_forecast_mask=True))
    return out

def native_stream(cp,events,base,pre,recording=None):
    """Same original lease outcomes, native fused anchor and issue context."""
    decisions=[]
    for original in base['decisions']:
        event=events[original['k']];ids,h,anchor,support=cp.issued_descriptor(event,pre)
        decisions.append(dict(original,ids=ids,h=h,anchor=anchor,
            descriptor=h-anchor,disagreement=support,
            cal_context=np.asarray([anchor,support,len(ids)/pre['m']]),
            context=event['context']))
    return dict(base,decisions=decisions,recording=recording)

def dbf_world(op,pdf_events,pre):
    out=[];quality=np.asarray(pre['q'])/np.mean(pre['q'])
    for event in pdf_events:
        ids=np.flatnonzero(event['mask'])
        if len(ids):
            probability,state=op.fuse_evidence(op.evidence(event['p'][ids],quality[ids],pre['classes']),'dbf')
            weights=state['discount'].T
        else:
            probability=event['external_probability'];weights=np.zeros((len(event['y']),0))
        out.append(dict(event,external_probability=probability,external_weights=weights))
    return out

def native_raw(stream):
    rows=[]
    for d in stream['decisions']:
        score=float(d['N']*d['anchor']-5.);ready=bool(len(d['ids']))
        rows.append(dict(k=d['k'],action=bool(ready and score>0),
            maturity=d['maturity'],local_net=d['localnet'],truegross=d['truegross'],
            gate_score=score,raw_gate_score=score,gain=score+5.,
            posterior_or_block_sd=1.,q_issued=0.,
            standardized_score=score-(d['truegross']-5.),
            lower_covered=bool(d['truegross']-5.>=score),information_ready=ready,
            disagreement=d['disagreement'],anchor=d['anchor'],
            cal_context=d['cal_context'].tolist(),N=d['N'],
            source_descriptor=d['descriptor'].tolist(),active_source_ids=d['ids'].tolist()))
    return dict(rows=rows,control='Published active-objective native fused anchor plus the common signed calibration transform')
