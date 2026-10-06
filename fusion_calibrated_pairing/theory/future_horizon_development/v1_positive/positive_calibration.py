"""Used-data v1: one monotone calibrated score, no added gate/guard.

The immutable residual population, maturity, strata, history capacity and
nine-grid quantile remain V2. Only its signed correction is floored at0.
Raw scores, state and old V0 results remain byte-identical.
"""
import types
import immutable_conditional_calibration_v2 as original

class PositiveCalibration(original.ConditionalScoreCalibration):
    def correction(self,context):
        value=super().correction(context)
        value['correction']=max(0.,value['correction'])
        return value

_globals=dict(original.__dict__,ConditionalScoreCalibration=PositiveCalibration)
_run=types.FunctionType(original.calibrated_run.__code__,_globals,
                        'positive_calibrated_run',original.calibrated_run.__defaults__,
                        original.calibrated_run.__closure__)

def calibrated_run(stream,rows,config,initial_pool=(),selection=False,poison_after=None,selection_boundary=None):
    out=_run(stream,rows,config,initial_pool,selection,poison_after,selection_boundary)
    signed=original.calibrated_run(stream,rows,config,initial_pool,selection,poison_after,selection_boundary)
    for positive,old in zip(out['rows'],signed['rows']):
        assert positive['k']==old['k']
        positive['calibration_signed_correction']=old['calibration_correction']
        positive['signed_shadow_gate_score']=old['gate_score']
        positive['signed_shadow_proposal']=old['action']
        positive['signed_promoted_nonpositive_raw']=bool(old['action'] and old['raw_gate_score']<=0.)
        if positive['gate_score']>positive['raw_gate_score']+1e-10:
            raise AssertionError('Monotone correction increased raw lower-tail score')
        if positive['gate_score']>old['gate_score']+1e-10:
            raise AssertionError('Positive correction exceeds signed score')
    out['calibration_role']='DEVELOPMENT v1: qplus=max(0,signed empiricalquantile); same one score/history/strata; no automatic conditional-risk certificate'
    return out

configurations=original.configurations
actual_admission_report=original.actual_admission_report
