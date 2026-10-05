"""Synthetic-only physical adapter readiness; never opens a dataset archive."""
from pathlib import Path
import ast
import datetime as dt
import hashlib
import importlib.util
import io
import json
import sys

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parent


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(gap=False, label_shift=0):
    result = ["timestamp,back_x,back_y,back_z,thigh_x,thigh_y,thigh_z,label"]
    base = dt.datetime(2026,1,1)
    codes = (1,2,3,4,5,6,7,8,13,14,130,140)
    for index in range(201):
        # Ordinary retained-interval1microsecond jitter cannot create a run.
        micros = index*20000+(1 if index >= 50 else 0)+(2000000 if gap and index >= 125 else 0)
        stamp = (base+dt.timedelta(microseconds=micros)).isoformat(sep=" ",timespec="microseconds")
        values = [str(index+j) for j in range(6)]
        result.append(",".join([stamp]+values+[str(codes[(index//50+label_shift)%12])]))
    return io.StringIO("\n".join(result)+"\n")


def synthetic_data(ad):
    rng = np.random.default_rng(4401)
    size = 160
    people = [f"S{i:03d}" for i in range(1,23)]
    recording = np.repeat(people,size)
    raw = rng.normal(size=(len(recording),6))
    labels = rng.integers(0,12,len(recording))
    rank = np.repeat(np.arange(1,23),size)
    origins = [f"synthetic/{person}/{row:05d}" for person in people for row in range(size)]
    data = dict(raw=raw,y=labels,recording=recording,session=recording.copy(),timestamp=origins,
        metadata=dict(split_boundaries=dict(ntrain=8*size,ncal=12*size),fit_participants=people[:8],
            calibration_participants=people[8:12],test_participants=people[12:],participants=people),
        subject_rank=rank)
    return data


def run():
    for path in (ROOT/"adapter.py",ROOT/"run_physical.py"):
        ast.parse(path.read_text())
    ad = module("harth_synthetic_adapter",ROOT/"adapter.py")
    continuous,audit = ad.parse_recording(fixture(),"Ssynthetic")
    shifted,_ = ad.parse_recording(fixture(label_shift=4),"Ssynthetic")
    gapped,gap_audit = ad.parse_recording(fixture(gap=True),"Ssynthetic")
    assert [row["raw_row"] for row in continuous] == [0,50,100,150,200]
    assert audit["history_runs"] == 1 and gap_audit["history_runs"] == 2
    assert [row["session"] for row in gapped] == ["Ssynthetic/run0001"]*3+["Ssynthetic/run0002"]*2
    for a,b in zip(continuous,shifted):
        for field in ("raw_row","tick","session","recording","origin"):
            assert a[field] == b[field]
        assert np.array_equal(a["raw"],b["raw"])
    lagged = ad.boundary_lagged(np.arange(30,dtype=float).reshape(5,6),np.zeros(6),np.ones(6)*10,
        np.asarray(["A","A","A","B","B"]))
    assert np.array_equal(lagged[3,:18].reshape(6,3)[:,0],lagged[3,:18].reshape(6,3)[:,1])
    assert np.array_equal(lagged[3,:18].reshape(6,3)[:,1],lagged[3,:18].reshape(6,3)[:,2])
    try:
        ad.process(ROOT/"DOES_NOT_EXIST_REAL_DATA.zip",ROOT/"NO_CACHE_SHOULD_BE_CREATED",ROOT/"acquisition_authorization_draft.json")
    except ValueError:
        pass
    else:
        raise AssertionError("Nonauthorizing metadata draft allowed ingestion")
    assert not (ROOT/"NO_CACHE_SHOULD_BE_CREATED").exists()

    physical = module("harth_runner_api_readiness",ROOT/"run_physical.py")
    binding = physical.r.binding()
    assert len(binding) == 8
    engine,_,_,_,cfg,_,_,_ = binding
    cfg = physical.base_cfg(cfg)
    data = synthetic_data(ad)
    pre = ad.make_prefix(data,cfg,engine)
    poisoned = dict(data,raw=data["raw"].copy(),y=data["y"].copy())
    cut = data["metadata"]["split_boundaries"]["ntrain"]
    poisoned["raw"][cut:] += 10000
    poisoned["y"][cut:] = (poisoned["y"][cut:]+3)%12
    altered = ad.make_prefix(poisoned,cfg,engine)
    for field in ("mean","scale","q","train_x","train_y"):
        assert np.array_equal(pre[field],altered[field]),field
    for a,b in zip(pre["models"],altered["models"]):
        assert np.array_equal(a,b)
    assert len(pre["recording_streams"]) == 14
    assert len([r for r in pre["recording_streams"] if r["partition"] == "calibration"]) == 4
    assert len([r for r in pre["recording_streams"] if r["partition"] == "test"]) == 10
    first_x = pre["recording_streams"][0]["x"]
    expected_context = [float(np.mean(first_x[:,[0,3,6]])),float(np.mean(first_x[:,[9,12,15]]))]
    assert np.array_equal(engine.context(first_x,ad.GROUPS),np.asarray(expected_context))
    readiness = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),passed=True,
        scope="synthetic-only adapter/provenance/API readiness; no HARTH raw/outcome access and no algorithm approval",
        physical_algorithm_status="pending final CJ-T API; existing runner remains CJ-R preparation draft",
        checks=dict(syntax=True,non_authorizing_draft_blocks_before_archive_open=True,
            fixed_acquisition_stride=True,labels_do_not_change_provenance_or_row_selection=True,
            raw_timestamp_gap_resets_history=True,nominal_1Hz_jitter_does_not_reset=True,
            no_cross_participant_lags=True,cal_test_poison_preserves_fitted_models_scaling_and_source_quality=True,
            physical_context_groups_use_current_back_and_thigh_coordinates=True,runner_binding_eight_tuple=True,
            synthetic_recording_stream_roles_4cal_10test=True),
        hashes={str(path):sha(path) for path in (ROOT/"adapter.py",ROOT/"run_physical.py",ROOT/"protocol_draft.json",
            ROOT/"acquisition_authorization_draft.json",Path(__file__))},
        runtime=dict(python=sys.version,numpy=np.__version__),
        real_archive_exists=(ROOT/"raw/harth779.zip").exists(),real_cache_exists=(ROOT/"data/cached_dataset.npz").exists())
    assert not readiness["real_archive_exists"] and not readiness["real_cache_exists"]
    (ROOT/"metadata_readiness.json").write_text(json.dumps(readiness,indent=2)+"\n")
    print(json.dumps(readiness,indent=2))


if __name__ == "__main__":
    run()
