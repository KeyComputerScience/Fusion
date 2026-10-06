"""Synthetic interface checks only; these are not action/value results."""
from pathlib import Path
import json, copy, hashlib
import numpy as np
from horizon_descriptor import HorizonDescriptor, issued_snapshot

ROOT = Path(__file__).resolve().parent


def event():
    return dict(x=np.ones((4, 1)), candidate=np.array([[1.4, 0.]]),
                reference=np.array([[.3, 0.]]),
                p=np.tile(np.array([[[.9, .1]], [[.55, .45]]]), (1, 4, 1)),
                external_probability=np.tile([.7, .3], (4, 1)),
                mask=np.array([True, True]))


def comparable(d):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()}


def run():
    e = event(); first = HorizonDescriptor().push(e, "synthetic-a")
    assert first["hard_disagreement"] == 0.
    assert first["hard_anchor"] == 0.
    assert first["soft_disagreement"] > 0.
    assert first["state_weight_support"] > 0.
    assert np.max(abs(first["source_blocks"][0]-first["source_blocks"][1])) > .1
    assert first["source_coordinate_mask"].sum() == 2*(2+4)
    poisoned = copy.deepcopy(e)
    poisoned.update(y=np.array([91, 92, 93, 94]), truegross=-999,
                    future_x=np.full((128, 1), 123.), localnet=-999,
                    maturity=999999)
    assert comparable(first) == comparable(HorizonDescriptor().push(poisoned, "synthetic-a"))
    path = HorizonDescriptor(); path.push(e, "synthetic-a")
    changed = copy.deepcopy(e); changed["p"] = changed["p"][::-1].copy()
    second = path.push(changed, "synthetic-a")
    assert second["history_frames"] == 2
    assert not np.array_equal(second["source_blocks"][:, :6], second["source_blocks"][:, 6:12])
    masked = copy.deepcopy(e); masked["mask"][1] = False
    third = path.push(masked, "synthetic-a")
    assert not third["source_coordinate_mask"][1, :6].any()
    assert np.all(third["source_blocks"][1, :6] == 0.)
    assert third["source_coordinate_mask"][1, 6:12].all()
    reset = path.push(e, "synthetic-b")
    assert reset["history_frames"] == 1
    assert not reset["source_coordinate_mask"][:, 6:].any()
    permutation = copy.deepcopy(e)
    permutation["p"] = permutation["p"][[1, 0]]
    permutation["mask"] = permutation["mask"][[1, 0]]
    p = HorizonDescriptor().push(permutation, "synthetic-a")
    assert np.allclose(p["source_blocks"], first["source_blocks"][[1, 0]])
    assert np.allclose(p["common_descriptor"], first["common_descriptor"])
    empty = copy.deepcopy(e); empty["mask"][:] = False
    absent = HorizonDescriptor().push(empty, "synthetic-a")
    assert not absent["active_sources"].size
    assert not absent["source_coordinate_mask"].any()
    report = dict(status="synthetic descriptor checks only; unfrozen development",
                  hard_agreement_soft_information_survives=True,
                  label_future_target_independence=True,
                  source_permutation_equivariance=True,
                  masked_forecasts_not_observed=True,
                  physical_session_reset=True,
                  immutable_three_issue_path=True,
                  hard_anchor=first["hard_anchor"],
                  soft_disagreement=first["soft_disagreement"],
                  source_block_width=first["source_block_width"],
                  prototype_sha256=hashlib.sha256((ROOT/"horizon_descriptor.py").read_bytes()).hexdigest())
    (ROOT/"synthetic_checks.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
