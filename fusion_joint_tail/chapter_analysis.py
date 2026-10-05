"""Analyze locked outcomes for the revised chapters; never select a policy."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def read(name):
    path = ROOT / name
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as f:
        return json.load(f)


def action_map(records, arm, budget=None):
    result = {}
    for trajectory in records:
        if trajectory["arm"] != arm:
            continue
        if budget is not None and trajectory["budget"] != budget:
            continue
        for row in trajectory["rows"]:
            key = (trajectory["seed"], row["k"])
            assert key not in result
            result[key] = row
    return result


def four_terms(joint, control):
    assert joint.keys() == control.keys()
    sums = dict(added_gain=0., avoided_loss=0., missed_gain=0., incurred_loss=0.)
    counts = dict(changed=0, added_beneficial=0, avoided_harmful=0,
                  missed_beneficial=0, incurred_harmful=0, changed_zero=0)
    difference = 0.
    for key, j in joint.items():
        c = control[key]
        assert abs(j["local_net"] - c["local_net"]) < 1e-9
        payoff = j["local_net"]
        delta = int(j["action"]) - int(c["action"])
        difference += delta * payoff
        if not delta:
            continue
        counts["changed"] += 1
        if payoff == 0:
            counts["changed_zero"] += 1
        elif delta > 0 and payoff > 0:
            sums["added_gain"] += payoff
            counts["added_beneficial"] += 1
        elif delta < 0 and payoff < 0:
            sums["avoided_loss"] -= payoff
            counts["avoided_harmful"] += 1
        elif delta < 0 and payoff > 0:
            sums["missed_gain"] += payoff
            counts["missed_beneficial"] += 1
        else:
            sums["incurred_loss"] -= payoff
            counts["incurred_harmful"] += 1
    decomposition = (sums["added_gain"] + sums["avoided_loss"] -
                     sums["missed_gain"] - sums["incurred_loss"])
    assert abs(difference - decomposition) < 1e-9
    return dict(**sums, **counts, pooled_increment=difference,
                five_delay_mean=difference / 5)


def coverage(issued, guarded, arm):
    rows = [r for t in issued if t["arm"] == arm for r in t["rows"]]
    actual = list(action_map(guarded, arm, 130.).values())
    def subset(values):
        return [sum(r["lower_covered"] for r in values), len(values)]
    admitted = [r for r in actual if r["action"]]
    excess = [max(0., r["gate_score"] - (r["truegross"] - 5)) for r in admitted]
    return dict(issued=subset(rows), ready=subset([r for r in rows if r["information_ready"]]),
                informative=subset([r for r in rows if r["disagreement"] > 0]),
                admitted=subset(admitted), admitted_excess_sum=sum(excess),
                admitted_excess_max=max(excess, default=0.),
                realized_negative_loss=sum(max(0., -r["local_net"]) for r in admitted))


def main():
    selected = read("rss_development/results.json.gz")
    fixed = read("fixed_information/results.json")
    joint = action_map(selected["guarded"], "joint", 130.)
    fixed_joint = action_map(fixed["guarded"], "joint")
    assert all(joint[k]["action"] == fixed_joint[k]["action"] for k in joint)
    arms = ("marginal", "factorized", "gaussian", "sandwich")
    report = dict(
        status="descriptive analysis of all stored locked outcomes; no tuning",
        selected={a: four_terms(joint, action_map(selected["guarded"], a, 130.)) for a in arms},
        fixed_information={a: four_terms(fixed_joint, action_map(fixed["guarded"], a)) for a in arms},
        coverage={a: coverage(selected["issued"], selected["guarded"], a) for a in ("joint",) + arms},
        counterfactual_opportunities=dict(positive=sum(r["local_net"] > 0 for r in joint.values()),
                                         negative=sum(r["local_net"] < 0 for r in joint.values()),
                                         zero=sum(r["local_net"] == 0 for r in joint.values())),
        source_hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in ("rss_development/results.json.gz", "fixed_information/results.json")},
        scope="Five delays share the RSS development trace; counts are not independent sites.")
    ready = [(t["seed"], r) for t in selected["issued"] if t["arm"] == "joint"
             for r in t["rows"] if r["information_ready"]]
    minimum, seed, origin, score = min((abs(r["gate_score"]), s, r["k"], r["gate_score"])
                                       for s, r in ready)
    offsets = {}
    for offset in (-.25, .25):
        changed = sum((r["gate_score"] + offset > 0) != (r["gate_score"] > 0)
                      for _, r in ready)
        assert changed == 0
        offsets[str(offset)] = dict(changed_ready_proposals=changed,
                                   scope="fixed issued alpha, forecasts, readiness and returns")
    report["working_score_stability"] = dict(min_absolute_ready_score=minimum,
        origin=origin, seed=seed, signed_score=score, ready_count=len(ready),
        uniform_offsets=offsets,
        interpretation="Numerical decision margin only; not a physical distribution-error bound.")
    report["archived_full_policy_comparisons"] = {}
    for arm, result_name, budget_name in (
        ("factorized_information", "work/fusion_strengthening_20261003/matrix/factorized/results.json",
         "work/fusion_strengthening_20261003/matrix/factorized/budget_trials.json.gz"),
        ("block_sandwich", "work/fusion_strengthening_20261003/matrix/rss348_results.json",
         "work/fusion_strengthening_20261003/matrix/rss348_budget_results.json.gz")):
        native = json.loads((PROJECT / result_name).read_text())
        with gzip.open(PROJECT / budget_name, "rt") as f:
            budgets = json.load(f)
        retained = [t for t in budgets if t["arm"] == arm and t["budget"] == 130.
                    and t["reserve"] == "fixed130"]
        assert len(retained) == 5
        candidate = {}
        max_target_error = 0.
        for trajectory in retained:
            seed = trajectory["seed"]
            native_trial = next(t for t in native["trials"] if t["seed"] == seed)
            actions = {r["k"]: r["action"] for r in trajectory["ledger"]}
            total = 0.
            for row in native_trial["results"][arm]["rows"]:
                key = (seed, row["k"])
                j = joint[key]
                max_target_error = max(max_target_error, abs(j["truegross"] - row["truegross"]),
                                       abs(j["local_net"] - row["local_net"]))
                candidate[key] = dict(row, action=actions[row["k"]])
                total += row["local_net"] * actions[row["k"]]
            assert abs(total - trajectory["increment"]) < 1e-8
        assert max_target_error < 1e-8
        report["archived_full_policy_comparisons"][arm] = dict(
            **four_terms(joint, candidate), common_target_count=len(candidate),
            max_target_error=max_target_error,
            scope="Saved native-tuning policies on identical payoffs; not a newly aligned tuning experiment.")
        for name in (result_name, budget_name):
            report["source_hashes"][name] = hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
    (ROOT / "manuscript_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
