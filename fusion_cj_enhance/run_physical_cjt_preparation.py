"""Frozen HARTH physical replay for CJ-T and matched/legacy controllers.

Preparation imports code only. Root calls freeze only after algorithm readiness;
acquisition occurs separately. All calibration uses the same nine configurations
and guarded second-half return. Test entry requires the shared selection lock.
"""
from pathlib import Path
import argparse
import datetime
import gzip
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parent
METHOD_PATH = ROOT.parent / "design_transport/transport.py"


def module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    answer = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(answer)
    return answer


r = module("harth_target_method", METHOD_PATH)
ad = module("harth_physical_adapter", ROOT / "adapter.py")
c = r.c
ARMS = r.ARMS
PRIMARY = "t_joint"
CAL_SEEDS = (122001, 122002, 122003)
TEST_SEEDS = (123001, 123002, 123003, 123004, 123005)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    text = json.dumps(value, indent=2, allow_nan=False)
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt") as handle:
            handle.write(text)
    else:
        Path(path).write_text(text + "\n")


def load(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(Path(path).read_text())


def base_cfg(cfg):
    out = dict(cfg, lease_archive=48, prior_sd=.1, threshold=0.)
    assert out["window"] == 32 and out["horizon"] == 4 and out["deploy_drop"] == 2
    return out


def scientific_sources():
    engine, recovery, guard, source, _, study, service, fork = r.binding()
    files = {Path(__file__), METHOD_PATH, ROOT/"adapter.py", ROOT/"protocol.json",
        ROOT/"run_external.py", ROOT/"external/protocol.json", ROOT/"acquire.py", ROOT.parent/"design_transport/protocol.json",
        Path(r.cp.__file__), Path(c.__file__), c.PARENT, c.parent.FACTOR,
        c.parent.parent.MATRIX_SOURCE, c.parent.CORE, c.parent.ADAPTER,
        Path(c.PROJECT)/"outputs/Fusion_Recovery_Repro/base/independent_bayes_fusion.py"}
    for obj in (engine,recovery,guard,source,study,service,fork):
        if hasattr(obj,"__file__"):files.add(Path(obj.__file__))
        elif hasattr(obj,"__code__"):files.add(Path(obj.__code__.co_filename))
    design = load(ROOT.parent/"design_transport/protocol.json")
    for path, expected in design["source_hashes"].items():
        assert sha(path) == expected, path
        files.add(Path(path))
    external = load(ROOT/"external/protocol.json")
    for path,expected in external["input_sha256"].items():
        assert sha(path)==expected,path
        files.add(Path(path))
    return sorted(files)


def prepare_protocol():
    """Finalize protocol bytes before both internal/external source freezes."""
    assert not (ROOT/"protocol.json").exists()
    assert not (ROOT/"raw/harth779.zip").exists() and not (ROOT/"data/cached_dataset.npz").exists()
    protocol = load(ROOT/"protocol_draft.json")
    assert protocol["status"] == "metadata_only_draft_not_authorized_for_acquisition"
    protocol.update(status="frozen_before_acquisition", utc=now(), adapter_sha256=sha(ROOT/"adapter.py"),
        runner_sha256=sha(__file__), method_sha256=sha(METHOD_PATH))
    dump(ROOT/"protocol.json", protocol)


def freeze():
    """Root-only explicit command; never called during metadata preparation."""
    assert not (ROOT/"algorithm_freeze.json").exists()
    assert not (ROOT/"raw/harth779.zip").exists() and not (ROOT/"data/cached_dataset.npz").exists()
    protocol=load(ROOT/"protocol.json")
    assert protocol["status"]=="frozen_before_acquisition"
    assert protocol["adapter_sha256"]==sha(ROOT/"adapter.py")
    assert protocol["runner_sha256"]==sha(__file__) and protocol["method_sha256"]==sha(METHOD_PATH)
    _, _, _, _, cfg, _, _, _ = r.binding()
    snapshot = dict(utc=now(), status="CJ-T algorithm and HARTH physical protocol frozen before raw acquisition",
        source_hashes={str(path):sha(path) for path in scientific_sources()}, arms=ARMS,
        primary=PRIMARY, base_cfg=base_cfg(cfg), calibration_seeds=CAL_SEEDS, test_seeds=TEST_SEEDS,
        grid=dict(conditional_bandwidth=c.BANDWIDTHS, legacy_prior_sd=(.05,.1,.2), q_floor=c.FLOORS, configurations_per_arm=9),
        select="maximum pooled guarded B130 second-half complete return; ties largerqfloor then largerbandwidth(legacy smallerprior_sd)",
        initial_q="pooled first-half ready fully-matured90th percentile, higher method",
        conditioning="t_* arms retain mature paired H,U; raw relevance unchanged; shared state cache by(bandwidth,prior_sd)",
        execution=dict(primary_budget=130., reserve=130., diagnostic_budgets=[0.,110.,130.,260.],
            budget_scope="B per actual participant/history-run chain; aggregate−B*actual_test_run_count"),
        info="same physical worlds, currenth, source masks/models, delayed mature histories, costs, prefixes and grid",
        causal="offline truegross produces callback standardized score only; F/S/action are invariant under current future-truth poisoning",
        runtime=dict(python=sys.version, numpy=np.__version__))
    dump(ROOT/"algorithm_freeze.json", snapshot)
    dump(ROOT/"acquisition_authorization.json", dict(utc=now(),
        authorized_stage="algorithm_and_physical_protocol_frozen_before_acquisition",
        adapter_sha256=sha(ROOT/"adapter.py"), physical_protocol_sha256=sha(ROOT/"protocol.json"),
        algorithm_freeze_path="algorithm_freeze.json", algorithm_freeze_sha256=sha(ROOT/"algorithm_freeze.json")))


def verify():
    frozen = load(ROOT/"algorithm_freeze.json")
    for path, expected in frozen["source_hashes"].items():
        assert sha(path) == expected, path
    ad.verify_authorization(ROOT/"acquisition_authorization.json")
    return frozen


def ingest():
    verify()
    meta = ad.process(ROOT/"raw/harth779.zip", ROOT/"data", ROOT/"acquisition_authorization.json")
    print(json.dumps({key:meta[key] for key in ("n","raw_rows","fit_participants","calibration_participants","test_participants")},indent=2),flush=True)


def prepared():
    verify()
    engine, recovery, guard, _, cfg, _, service, fork = r.binding()
    cfg = base_cfg(cfg)
    data = ad.load_data(ROOT/"data")
    pre = ad.make_prefix(data,cfg,engine)
    return engine,recovery,guard,cfg,service,fork,data,pre


def augmented(engine, events, pre, cfg, cache, base):
    key = (cfg["slice_bandwidth"],cfg["prior_sd"])
    if key not in cache:
        cache[key] = r.core.augment(events,pre,base,cfg)
    return cache[key]


def calibrate():
    verify()
    assert not (ROOT/"selection.json").exists()
    engine,recovery,guard,cfg0,service,fork,data,pre = prepared()
    records = [z for z in pre["recording_streams"] if z["partition"] == "calibration"]
    assert set(z["person"] for z in records) == set(data["metadata"]["calibration_participants"])
    worlds = [(record,seed,engine.world(record["x"],record["y"],record["timestamp"],pre,seed,cfg0))
              for record in records for seed in CAL_SEEDS]
    bases = [engine.precompute(e,pre,cfg0) for _,_,e in worlds]
    caches = [{} for _ in worlds]
    grid,trials = [],[]
    max_service_error = 0.
    for arm in ARMS:
        legacy = arm in ("legacy_factorized","legacy_sandwich")
        values = (.05,.1,.2) if legacy else c.BANDWIDTHS
        for value in values:
            bandwidth = 1. if legacy else value
            cfg = r.configure(dict(cfg0,slice_bandwidth=bandwidth,prior_sd=value if legacy else cfg0["prior_sd"]),arm)
            streams = [augmented(engine,e,pre,cfg,cache,base) for (_,_,e),cache,base in zip(worlds,caches,bases)]
            for floor in c.FLOORS:
                cc = dict(cfg,q_floor=floor)
                qinitial,scores = r.core.initial_q(streams,pre,cc,arm)
                results = [r.core.run(engine,stream,pre,cc,arm,qinitial,True) for stream in streams]
                chains = []
                for (record,seed,events),stream,result in zip(worlds,streams,results):
                    assert result["solver_failures"] == 0
                    check = recovery.execute_check(events,pre,cc,result,service,fork)
                    assert check["passed"]
                    guarded = guard.replay(result["rows"],len(events),130.,"gross_loss",{row["k"]:130. for row in result["rows"]})
                    boundary = stream["windows"]//2
                    assert not any(row["action"] and row["k"] < boundary for row in guarded["rows"])
                    actual,reference = service(events,pre,guarded["rows"],cc),service(events,pre,[],cc)
                    net = sum(row["local_net"] for row in guarded["rows"] if row["action"])
                    error = abs(actual["net"]-reference["net"]-net)
                    max_service_error = max(max_service_error,error)
                    assert error < 1e-8
                    assert all(a["spent_loss"]+a["reserved"] <= 130.+1e-8 for a in guarded["ledger"])
                    chains.append(dict(recording=record["recording"],person=record["person"],seed=seed,
                        guarded_second_half_increment=net,result=result,guarded=guarded,service_check=check))
                selected_net = float(sum(chain["guarded_second_half_increment"] for chain in chains)/len(CAL_SEEDS))
                candidate = dict(arm=arm,bandwidth=bandwidth,q_floor=floor,q_initial=qinitial,net=selected_net,fit_scores=scores,
                    prior_sd=cfg["prior_sd"])
                grid.append(candidate)
                trials.append(dict(configuration=candidate,chains=chains))
            print("CAL",arm,"bandwidth",bandwidth,"prior_sd",cfg["prior_sd"],flush=True)
    selected = {arm:max((g for g in grid if g["arm"] == arm),key=lambda g:(g["net"],g["q_floor"],
                    -g["prior_sd"] if arm.startswith("legacy_") else g["bandwidth"])) for arm in ARMS}
    dump(ROOT/"calibration_trials.json.gz",trials)
    dump(ROOT/"selection.json",dict(selected=selected,grid=grid,split=pre["split"],data_hashes=data["hashes"],
        calibration_participant_count=len(data["metadata"]["calibration_participants"]),calibration_run_count=len(records),
        max_service_error=max_service_error,criterion="pooled guarded B130 second-half complete paid return"))
    dump(ROOT/"before_test.json",dict(utc=now(),algorithm_freeze_sha256=sha(ROOT/"algorithm_freeze.json"),
        selection_sha256=sha(ROOT/"selection.json"),cache_sha256=data["metadata"]["cache_sha256"]))
    print("SELECT",{a:(v["bandwidth"],v["q_floor"],v["q_initial"],v["net"]) for a,v in selected.items()},flush=True)


def locktest():
    """Root invokes only after internal and external selection locks exist."""
    verify()
    assert not (ROOT/"all_selections_before_test.json").exists()
    before = load(ROOT/"before_test.json")
    assert before["selection_sha256"] == sha(ROOT/"selection.json")
    assert set(load(ROOT/"selection.json")["selected"]) == set(ARMS)
    external_before = load(ROOT/"external/selection_freeze.json")
    assert external_before["selection_sha256"] == sha(ROOT/"external/selection.json")
    paths = [ROOT/"algorithm_freeze.json",ROOT/"protocol.json",ROOT/"selection.json",ROOT/"before_test.json",
             ROOT/"external/protocol.json",ROOT/"external/selection.json",ROOT/"external/selection_freeze.json"]
    dump(ROOT/"all_selections_before_test.json",dict(utc=now(),stage="all_physical_controller_selections_locked_before_test",
        inputs={str(path):sha(path) for path in paths},cache_sha256=before["cache_sha256"],arms=ARMS,
        declared_scope="all8internal/matched/legacy arms and both external protocol-declared arms"))


def test():
    frozen = verify()
    before = load(ROOT/"before_test.json")
    shared = load(ROOT/"all_selections_before_test.json")
    assert shared["stage"] == "all_physical_controller_selections_locked_before_test"
    for path, expected in shared["inputs"].items():
        assert sha(path) == expected,path
    assert before["selection_sha256"] == sha(ROOT/"selection.json")
    assert before["algorithm_freeze_sha256"] == sha(ROOT/"algorithm_freeze.json")
    engine,recovery,guard,cfg0,service,fork,data,pre = prepared()
    assert before["cache_sha256"] == data["metadata"]["cache_sha256"] == shared["cache_sha256"]
    records = [z for z in pre["recording_streams"] if z["partition"] == "test"]
    assert set(z["person"] for z in records) == set(data["metadata"]["test_participants"])
    selected = load(ROOT/"selection.json")["selected"]
    summary = dict(test_participants=data["metadata"]["test_participants"],test_participant_count=10,
        test_run_count=len(records),aggregate_guarantee="negative loss<=B*actual_test_run_count per shared delay schedule",
        records=[],arms=ARMS)
    for record in records:
        stem = record["recording"].replace("/","_")
        path = ROOT/(stem+"_results.json.gz")
        assert not path.exists()
        trials,guarded_rows,fixed = [],[],[]
        max_error,max_quad,poisons = 0.,0.,0
        for seed in TEST_SEEDS:
            events = engine.world(record["x"],record["y"],record["timestamp"],pre,seed,cfg0)
            base = engine.precompute(events,pre,cfg0)
            reference = service(events,pre,[],cfg0)
            caches,results,checks = {},{},{}
            for arm in ARMS:
                ss = selected[arm]
                cfg = r.configure(dict(cfg0,slice_bandwidth=ss["bandwidth"],q_floor=ss["q_floor"],prior_sd=ss["prior_sd"]),arm)
                stream = augmented(engine,events,pre,cfg,caches,base)
                result = r.core.run(engine,stream,pre,cfg,arm,ss["q_initial"])
                assert result["solver_failures"] == 0
                check = recovery.execute_check(events,pre,cfg,result,service,fork)
                assert check["passed"]
                results[arm],checks[arm] = result,check
                for decision,row in zip(stream["decisions"],result["rows"]):
                    one = r.forecast(decision,pre,cfg,arm,row["q_issued"])
                    two = r.forecast(dict(decision,truegross=1e50,localnet=-1e50),pre,cfg,arm,row["q_issued"])
                    assert one[2:4] == two[2:4]
                    assert all(item["maturity"] <= decision["k"] for item in decision["temporal"]["eligible"])
                    assert all(item["maturity"] <= decision["k"] for item in decision["temporal"]["transport_eligible"])
                    poisons += 1
                    max_quad = max(max_quad,row.get("quadrature_error",0.))
                for budget in frozen["execution"]["diagnostic_budgets"]:
                    gg = guard.replay(result["rows"],len(events),budget,"gross_loss",{row["k"]:130. for row in result["rows"]})
                    actual = service(events,pre,gg["rows"],cfg)
                    prefix = guard.reconstruct_prefix_increment(events,gg["rows"],cfg)
                    acts = [row for row in gg["rows"] if row["action"]]
                    increment = actual["net"]-reference["net"]
                    loss = sum(max(0.,-row["local_net"]) for row in acts)
                    error = max(abs(increment-sum(row["local_net"] for row in acts)),abs(increment-prefix["final"]),abs(increment-gg["final_settled_increment"]))
                    max_error = max(max_error,error)
                    assert error < 1e-8 and loss <= budget+1e-8 and prefix["minimum"] >= -budget-1e-8
                    assert all(row["spent_loss"]+row["reserved"] <= budget+1e-8 for row in gg["ledger"])
                    guarded_rows.append(dict(recording=record["recording"],person=record["person"],seed=seed,arm=arm,budget=budget,
                        increment=increment,net=actual["net"],negative_loss=loss,minimum_prefix=prefix["minimum"],
                        admissions=len(acts),beneficial=sum(row["local_net"]>0 for row in acts),harmful=sum(row["local_net"]<0 for row in acts),
                        covered_admitted=sum(row["lower_covered"] for row in acts),
                        optimistic_excess=sum(row["posterior_or_block_sd"]*max(row["standardized_score"]-row["q_issued"],0.) for row in acts),
                        refused=sum(row["proposed_action"] and not row["action"] for row in gg["rows"]),rows=gg["rows"],ledger=gg["ledger"]))
            ss = selected[PRIMARY]
            cfg = r.configure(dict(cfg0,slice_bandwidth=ss["bandwidth"],q_floor=ss["q_floor"],prior_sd=ss["prior_sd"]),PRIMARY)
            primary_stream = augmented(engine,events,pre,cfg,caches,base)
            # Pairing controls share the issued primary target law; old
            # controllers also see that state here, a descriptive intervention.
            for decision,row in zip(primary_stream["decisions"],results[PRIMARY]["rows"]):
                predictions = {}
                for arm in ARMS:
                    _,_,F,S,_,_ = r.forecast(decision,pre,cfg,arm,row["q_issued"])
                    predictions[arm] = dict(F=F,S=S,score=F-row["q_issued"]*S-5,action=bool(F-row["q_issued"]*S-5>0))
                fixed.append(dict(recording=record["recording"],seed=seed,k=decision["k"],local_net=decision["localnet"],
                    scope="same t_joint issued state/q/bandwidth; proposed-action only",predictions=predictions,
                    h=decision["h"].tolist(),N=decision["N"],q=row["q_issued"],ready=decision["temporal"]["transport_ready"],
                    law={key:decision["temporal"]["transport_law"][key].tolist() for key in ("alpha","locations","covariances","mean","C")}))
            trials.append(dict(recording=record["recording"],seed=seed,reference=reference,results=results,checks=checks))
            print("TEST",record["recording"],seed,{a:round(v["net"]-reference["net"],3) for a,v in results.items()},flush=True)
        dump(path,trials)
        dump(ROOT/(stem+"_guarded.json.gz"),guarded_rows)
        dump(ROOT/(stem+"_fixed_state.json.gz"),fixed)
        check = dict(passed=True,max_service_error=max_error,future_truth_poison_checks=poisons,max_quadrature_error=max_quad,
            selected_policy_trajectories=len(trials)*len(ARMS),budget_trajectories=len(guarded_rows),unchanged_formula_hashes=True)
        dump(ROOT/(stem+"_verification.json"),check)
        summary["records"].append(dict(recording=record["recording"],person=record["person"],verification=check))
    dump(ROOT/"test_chain_summary.json",summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("prepare-protocol","freeze","ingest","calibrate","locktest","test"))
    args = parser.parse_args()
    {"prepare-protocol":prepare_protocol,"freeze":freeze,"ingest":ingest,"calibrate":calibrate,"locktest":locktest,"test":test}[args.phase]()
