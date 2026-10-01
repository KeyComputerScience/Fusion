"""Declared controlled service shifts; no trace or hardware claims.

A routing permutation changes the meaning of a dispatch action, without changing
its logged identity. It models an unannounced dispatch-interface remapping. The
agent observes queues/resources and reward, never the scenario phase or map.
All methods use the same maps, inputs, training profiles and initial parameters.
"""
from __future__ import annotations
import copy, json, math, time
from pathlib import Path
import numpy as np
from closed_loop import ExperimentCoordinator, finish_transition, method_config
from backend import EdgeService
from coordinator import Decision
from fusion import EvidenceFusion,FusionSnapshot
from pipeline import FusionPipeline

class _MappedAgent:
    def __init__(self,agent,mapping): self.agent,self.mapping,self.raw_action=agent,mapping,None
    def __getattr__(self,name): return getattr(self.agent,name)
    def act(self,state,epsilon=0):
        action,logged,value=self.agent.act(state,epsilon)
        self.raw_action=action
        return self.mapping[action],logged,value

class ShiftService(EdgeService):
    def execute(self,decision,agent,state,epsilon=0):
        relative=self.slot-self.config['calibration_slots']
        phase=int(np.searchsorted(self.config['drift_onsets'],relative,side='right')) if relative>0 else 0
        mapping=self.config.get('action_permutations',[[0,1,2]]*3)[phase]
        proxy=_MappedAgent(agent,mapping)
        feedback,diagnostics,deployment=super().execute(decision,proxy,state,epsilon)
        diagnostics['executed_priority']=diagnostics['application_action']
        diagnostics['application_action']=proxy.raw_action
        diagnostics['environment_action_map']=mapping # audit log only; not in policy state or fusion inputs
        return feedback,diagnostics,deployment

class RevisionCoordinator(ExperimentCoordinator):
    def __init__(self,config,profiles,experiment,method,static_profile=None):
        super().__init__(config,profiles,experiment,method)
        self.static_profile=static_profile
    def choose(self,slot,training_load,inference_load,capacities,snapshot,remaining_slots,solver='auto',future_retentions=None,pending_gain_forecast=None):
        if self.method!='static':
            return super().choose(slot,training_load,inference_load,capacities,snapshot,remaining_slots,solver,future_retentions,pending_gain_forecast)
        i=self.training_by_id[self.base];j=self.inference_by_id[self.static_profile]
        feasible=self.feasible(i,j,0,inference_load,capacities)
        return Decision(slot,i['id'] if feasible else None,j['id'] if feasible else None,
                        self.service_quality(j) if feasible else None,'fixed_calibration_profile',
                        'feasible' if feasible else 'infeasible_admission_required',False,self.retention(snapshot),self.H,
                        snapshot.completed_slot,snapshot.gamma,snapshot.uncertainty,snapshot.coverage,1,[])

class FusionAdapter:
    def __init__(self,config,reference,method):
        self.base=EvidenceFusion(config['fusion'],reference)
        self.snapshot=self.base.snapshot
        self.method,self.context=config, None
        self.mode=method
        self.extra={}
        if method=='revised':
            from revised_fusion import RevisedFusion
            self.extension=RevisedFusion(config)
        elif method in ['equal','bpf','ewa','boa']:
            from modern_baselines import ReplayBaseline
            name={'equal':'EF','bpf':'IMSE','ewa':'EWA','boa':'BOA'}[method]
            self.extension=ReplayBaseline(name,config,reference)
        else:self.extension=None
    def complete_window(self,rows,index):
        if self.mode in ['equal','bpf','ewa','boa']:
            self.snapshot=self.extension.complete_window(rows,index)
            return self.snapshot
        snap=self.base.complete_window(rows,index)
        if self.mode=='revised':
            contexts={tuple(str(row.get(k)) for k in ['logged_training','logged_inference','policy_version']) for row in rows}
            context={'stable':len(contexts)==1,'key':next(iter(contexts)) if len(contexts)==1 else None}
            self.snapshot=self.extension.update(snap,context)
            self.extra=self.extension.extras(self.snapshot)
        else:self.snapshot=snap
        return self.snapshot

def run_revision(data,experiment,core,profiles,calibrated_agent,reference,method,seed,static_profile):
    started_at=time.perf_counter()
    config=method_config(core,method,seed)
    agent=copy.deepcopy(calibrated_agent)
    env=ShiftService(data,experiment,profiles)
    pipeline=FusionPipeline(config,profiles,reference,collect_history=True)
    pipeline.coordinator=RevisionCoordinator(config['coordination'],profiles,experiment,method,static_profile)
    pipeline.fusion=FusionAdapter(config,reference,method)
    pending,records,timings=None,[],[]
    for slot in range(pipeline.origin,pipeline.origin+experiment['evaluation_slots']):
        relative=slot-pipeline.origin+1
        obs,gains=env.begin(slot,agent)
        pipeline.coordinator.relative_slot=relative
        stamp=time.perf_counter()
        decision=pipeline.decide(slot,obs['training_load'],obs['inference_load'],obs['capacities'],pending_gain_forecast=gains)
        timings.append((time.perf_counter()-stamp)*1000)
        snapshot=pipeline.fusion.snapshot
        state=env.state(obs,snapshot,pipeline.coordinator.H,decision.inference_profile,relative)
        # Keep fusion/recovery/time features at their calibration constants to
        # isolate coordination/update effects from direct policy-input changes.
        state[5:9]=[0.0,1.0,0.0,0.5]; state[12]=0.0
        finish_transition(agent,pending,state)
        feedback,diag,deployment=env.execute(decision,agent,state,experiment['agent']['epsilon_end'] if agent.kind=='dqn' else 0)
        pending=(state,diag['application_action'],feedback['environment_reward'],diag['log_probability'],diag['state_value'],agent.version)
        outcomes=pipeline.observe(feedback)
        env.apply_deployment(agent,deployment)
        if len(pipeline.window_results) and pipeline.window_results[-1]['completed_slot']==slot:
            pipeline.window_results[-1].update(pipeline.fusion.extra)
        records.append({**feedback,**decision.to_dict(),**outcomes,**diag,'evaluation_slot':relative,
                        'source_scores':snapshot.scores,'source_weights':snapshot.weights,'source_masks':snapshot.masks,
                        'policy_version_after':agent.version,'backbone':agent.kind,'method':method,'seed':seed,
                        'scenario':experiment['scenario']})
    finish_transition(agent,pending,state,done=True)
    assert env.arrived==env.completed+env.expired+env.rejected+int(env.queue_counts().sum())
    assert all(e['delay']>=e['declared_delay'] for e in env.events)
    assert env.resource_violations==0
    summary={'method':method,'backbone':agent.kind,'seed':seed,'scenario':experiment['scenario'],
             'data_kind':str(data['provenance']),'return':sum(r['environment_reward'] for r in records),
             'completion_fraction':env.completed/max(env.arrived,1),'deadline_fraction':(env.expired+env.rejected)/max(env.arrived,1),
             'deployed_jobs':env.deployed_jobs,'training_jobs':env.started_jobs,'gradient_updates':env.training_steps,
             'training_cost':env.total_cost,'resource_violations':env.resource_violations,
             'arrivals':env.arrived,'completed':env.completed,'expired':env.expired,'rejected':env.rejected,
             'queue_end':int(env.queue_counts().sum()),'static_profile':static_profile,
             'initial_parameter_sha256':calibrated_agent.parameter_digest(),'final_parameter_sha256':agent.parameter_digest(),
             'runtime_seconds':time.perf_counter()-started_at}
    return records,pipeline.window_results,env.events,summary
