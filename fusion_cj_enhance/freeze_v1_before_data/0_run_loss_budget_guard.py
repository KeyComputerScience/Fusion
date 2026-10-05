"""Deterministic delayed-feedback deployment-loss guard on frozen fusion proposals.

The guard API receives proposals and matured admitted-lease callbacks only.
Offline truth is isolated in the replay scheduler; it cannot enter admission.
Request-level reconstruction regenerates the frozen exogenous model world.
Both budget grids are diagnostic and do not select a replacement primary arm.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE = ROOT.parents[2] / "outputs" / "Fusion_Recovery_Repro"
ARMS = ("mean_transfer", "diagonal_transfer", "joint_transfer", "full_transfer", "bound_only", "point_transfer")
BUDGETS = (0, 133, 266, 532)
MODES = ("gross_loss", "net_floor")
RESERVES = ("adaptive", "fixed")
M = 133.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


@dataclass
class DelayedLossBudgetGuard:
    """Online state with no unarrived returns or posterior/oracle fields."""
    budget: float
    mode: str = "gross_loss"
    spent_loss: float = 0.0
    settled_increment: float = 0.0
    pending: dict = field(default_factory=dict)

    def capacity(self):
        reserved = sum(self.pending.values())
        if self.mode == "gross_loss":
            return self.budget - self.spent_loss - reserved
        if self.mode == "net_floor":
            return self.budget + self.settled_increment - reserved
        raise ValueError("Unknown guard mode")

    def step(self, current_time, proposal, matured_callbacks=(), maximum_loss=M):
        # The caller delivers only callbacks with verified maturity <= now.
        arrived = []
        for origin, maturity, complete_increment in matured_callbacks:
            assert maturity <= current_time
            assert origin in self.pending, "Callback must identify an admitted pending lease"
            reserved = self.pending.pop(origin)
            assert complete_increment >= -reserved - 1e-10
            self.spent_loss += max(0.0, -float(complete_increment))
            self.settled_increment += float(complete_increment)
            arrived.append(dict(origin=origin, maturity=maturity, increment=complete_increment))
        before = self.capacity()
        admit = bool(proposal and before + 1e-12 >= maximum_loss)
        if admit:
            assert current_time not in self.pending
            self.pending[current_time] = float(maximum_loss)
        after = self.capacity()
        assert after >= -1e-10, "Budget invariant failed"
        return admit, dict(k=int(current_time), proposal=bool(proposal), action=admit,
                          arrived=arrived, spent_loss=self.spent_loss,
                          settled_increment=self.settled_increment,
                          reserved=sum(self.pending.values()),
                          pending_origins=sorted(self.pending),
                          capacity_before=before, capacity_after=after)


def replay(rows, windows, budget, mode, reserves, perturb_after=None):
    """Offline scheduler exposes lease truth only in matured callbacks.

    perturb_after is a causality diagnostic: outcomes whose maturity is later
    than that cutoff are changed while all earlier proposal actions must stay.
    """
    decisions = {int(r["k"]): r for r in rows}
    guard = DelayedLossBudgetGuard(float(budget), mode)
    queued = []
    logs = []
    guarded_rows = []
    for k in range(windows + 1):
        due = [c for c in queued if c[1] <= k]
        queued = [c for c in queued if c[1] > k]
        row = decisions.get(k)
        reserve = float(reserves.get(k, M))
        action, ledger = guard.step(k, bool(row and row["action"]), due, maximum_loss=reserve)
        ledger["issued_reserve"] = reserve if row is not None else None
        if row is not None:
            new = dict(row, action=action, proposed_action=bool(row["action"]))
            guarded_rows.append(new)
            if action:
                outcome = float(row["local_net"])
                if perturb_after is not None and row["maturity"] > perturb_after:
                    outcome = -reserve if outcome >= 0 else reserve
                # Only the scheduler has this offline complete outcome.
                queued.append((k, int(row["maturity"]), outcome))
        logs.append(ledger)
    test_end_state = dict(spent_loss=guard.spent_loss,
                          settled_increment=guard.settled_increment,
                          reserved=sum(guard.pending.values()),
                          pending_origins=sorted(guard.pending))
    for maturity in sorted({c[1] for c in queued}):
        due = [c for c in queued if c[1] == maturity]
        _, ledger = guard.step(maturity, False, due)
        logs.append(ledger)
    assert not guard.pending
    return dict(rows=guarded_rows, ledger=logs, test_end_state=test_end_state,
                final_spent_loss=guard.spent_loss,
                final_settled_increment=guard.settled_increment)


def current_decision_reserve(x, candidate, reference, common_drop, cfg):
    """Label-free current contrast bound; no future covariates enter this API."""
    n = len(x)
    assert n <= cfg["window"]
    candidate_predictions = np.argmax(x @ candidate, axis=1)
    reference_predictions = np.argmax(x @ reference, axis=1)
    extra = min(cfg["deploy_drop"], max(0, n-common_drop))
    differing = int(np.sum(candidate_predictions[common_drop+extra:] != reference_predictions[common_drop+extra:]))
    reserve = 3.0 + extra + differing + (cfg["horizon"]-1) * cfg["window"]
    assert 0 < reserve <= M
    return reserve, dict(current_requests=n, common_drop=common_drop,
                         extra_dropped_requests=extra,
                         current_served_disagreement=differing,
                         future_request_bound=(cfg["horizon"]-1)*cfg["window"],
                         fixed_action_fees=3, issued_reserve=reserve)


def reconstruct_prefix_increment(events, rows, cfg):
    """Independent request/fee differences at every request and fee boundary."""
    actions = {int(r["k"]): bool(r["action"]) for r in rows}
    current = None
    until = -1
    increment = 0.0
    lease_increments = {}
    current_origin = None
    minimum = 0.0
    maximum_prefix_negative_loss = 0.0
    trace = []
    def record_prefix():
        nonlocal minimum, maximum_prefix_negative_loss
        minimum = min(minimum, increment)
        maximum_prefix_negative_loss = max(maximum_prefix_negative_loss,
                                            sum(max(0., -d) for d in lease_increments.values()))
    for k, event in enumerate(events):
        if current is not None and k == until:
            increment -= 1.0  # restoration fee is paid at closure
            lease_increments[current_origin] -= 1.0
            record_prefix()
            current = None
            current_origin = None
        admission = actions.get(k, False)
        if admission:
            assert current is None
            current = event["candidate"].copy()
            current_origin = k
            lease_increments[k] = -2.0
            until = k + cfg["horizon"]
            increment -= 2.0
            record_prefix()
        ref_predictions = np.argmax(event["x"] @ event["reference"], axis=1)
        used = event["reference"] if current is None else current
        predictions = np.argmax(event["x"] @ used, axis=1)
        common = (cfg["probe_drop"] if event["probe"] else 0) + (2 if event["refresh"] else 0)
        extra = cfg["deploy_drop"] if admission else 0
        guarded_keep = np.arange(len(predictions)) >= common + extra
        reference_keep = np.arange(len(predictions)) >= common
        guarded_correct = int(np.sum(predictions[guarded_keep] == event["y"][guarded_keep]))
        reference_correct = int(np.sum(ref_predictions[reference_keep] == event["y"][reference_keep]))
        for request in range(len(predictions)):
            delta = int(guarded_keep[request] and predictions[request] == event["y"][request]) - int(reference_keep[request] and ref_predictions[request] == event["y"][request])
            increment += delta
            if current_origin is not None:
                lease_increments[current_origin] += delta
            else:
                assert delta == 0
            record_prefix()
        trace.append(dict(window=k, increment=increment,
                          guarded_correct=guarded_correct, reference_correct=reference_correct,
                          admission=admission,
                          prefix_negative_loss=sum(max(0., -d) for d in lease_increments.values())))
    if current is not None:
        assert until == len(events)
        increment -= 1.0
        lease_increments[current_origin] -= 1.0
        record_prefix()
    return dict(final=increment, minimum=minimum,
                maximum_prefix_negative_loss=maximum_prefix_negative_loss,
                independent_lease_increments=lease_increments,
                trace=trace)


def compact_summary(trajectories):
    summary = {}
    for mode, reserve_mode in (("gross_loss", "adaptive"), ("gross_loss", "fixed"), ("net_floor", "fixed")):
        mode_summary = {}
        for dataset in dict.fromkeys(t["dataset"] for t in trajectories):
            task_summary = {}
            for arm in ARMS:
                budget_summary = {}
                for budget in BUDGETS:
                    selected = [t for t in trajectories if t["mode"] == mode and t["reserve_mode"] == reserve_mode and t["dataset"] == dataset and t["arm"] == arm and t["budget"] == budget]
                    values = np.array([t["increment"] for t in selected])
                    half = 2.776445105 * values.std(ddof=1) / np.sqrt(5)
                    budget_summary[str(budget)] = dict(
                        mean_increment=float(values.mean()), increments=values.tolist(),
                        conditional_t4_ci=[float(values.mean()-half), float(values.mean()+half)],
                        mean_admissions=float(np.mean([t["admissions"] for t in selected])),
                        mean_blocked=float(np.mean([t["blocked"] for t in selected])),
                        harmful=sum(t["harmful"] for t in selected),
                        beneficial=sum(t["beneficial"] for t in selected),
                        total_negative_loss=sum(t["negative_loss"] for t in selected),
                        max_scenario_negative_loss=max(t["negative_loss"] for t in selected),
                        minimum_scenario_prefix_increment=min(t["minimum_prefix_increment"] for t in selected),
                        mean_unguarded_increment=float(np.mean([t["unguarded_increment"] for t in selected])),
                        mean_guard_cost=float(np.mean([t["increment"]-t["unguarded_increment"] for t in selected])))
                task_summary[arm] = budget_summary
            mode_summary[dataset] = task_summary
        summary[mode+"_"+reserve_mode] = mode_summary
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    package = args.package.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = package / "extension" / "results.json"
    protocol_path = output / "protocol.json"
    assert protocol_path.exists(), "Write the complete diagnostic protocol before execution"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["source_results_sha256"] == sha(source)
    assert protocol["script_sha256"] == sha(__file__)
    assert protocol["budgets"] == list(BUDGETS) and protocol["modes"] == list(MODES)
    saved = json.loads(source.read_text())
    original = {p.relative_to(package).as_posix():sha(p) for p in package.rglob("*") if p.is_file()}
    sys.path.insert(0, str(package / "base"))
    from cached_runner import install_cached_loaders
    engine, _ = install_cached_loaders()
    from audit_canary_execution import load_study, explicit_service, explicit_fork
    sys.path.insert(0, str(package / "extension"))
    from new_data_adapter import load_dataset, make_prefix
    enhanced_protocol = json.loads((package / "extension" / "protocol.json").read_text())
    cfg = dict(engine.BASE, **enhanced_protocol["extension"])
    assert cfg["horizon"] * cfg["window"] + cfg["deploy_drop"] + 3 == M
    trajectories = []
    verification = dict(service_checks=0, independent_fork_checks=0,
                        max_fork_error=0.0, max_service_accounting_error=0.0,
                        max_reference_error=0.0, min_ledger_capacity=1e99,
                        worst_net_floor_violation=0.0, worst_gross_loss_violation=0.0,
                        future_truth_invariance_checks=0, source_unchanged=False)
    issued_reserve_logs = []
    for name, result in saved.items():
        spec = enhanced_protocol["studies"][name]
        if spec.get("new_task"):
            data, data_spec = load_dataset(package / "extension" / "data", name)
            pre = make_prefix(data, data_spec, cfg, engine)
        else:
            _, pre, _ = load_study(package / "base" / "independent_data", package / "base" / spec["folder"], name)
        assert pre["split"] == result["split"]
        for trial in result["trials"]:
            seed = trial["seed"]
            events = engine.world(pre["test_x"], pre["test_y"], pre["test_timestamp"], pre, seed, cfg)
            reference = explicit_service(events, pre, [], cfg)
            verification["max_reference_error"] = max(verification["max_reference_error"], abs(reference["net"] - trial["results"]["reference"]["net"]))
            for arm in ARMS:
                rows = trial["results"][arm]["rows"]
                true_returns = []
                adaptive_reserves = {}
                for row in rows:
                    origin = int(row["k"])
                    event = events[origin]
                    common_drop = (cfg["probe_drop"] if event["probe"] else 0) + (2 if event["refresh"] else 0)
                    reserve, reserve_log = current_decision_reserve(event["x"], event["candidate"], event["reference"], common_drop, cfg)
                    adaptive_reserves[origin] = reserve
                    issued_reserve_logs.append(dict(dataset=name, seed=seed, arm=arm, origin=origin, **reserve_log))
                    fork = explicit_fork(events, row, cfg)
                    verification["independent_fork_checks"] += 1
                    verification["max_fork_error"] = max(verification["max_fork_error"], abs(fork["actual"]-row["local_net"]))
                    assert fork["actual"] >= -M
                    assert fork["actual"] >= -reserve
                    true_returns.append(fork["actual"])
                for mode, reserve_mode in (("gross_loss", "adaptive"), ("gross_loss", "fixed"), ("net_floor", "fixed")):
                    reserves = adaptive_reserves if reserve_mode == "adaptive" else {int(r["k"]):M for r in rows}
                    for budget in BUDGETS:
                        guarded = replay(rows, trial["windows"], budget, mode, reserves)
                        audit = explicit_service(events, pre, guarded["rows"], cfg)
                        cumulative = reconstruct_prefix_increment(events, guarded["rows"], cfg)
                        chosen = np.array([r["action"] for r in guarded["rows"]], bool)
                        returns = np.array(true_returns)
                        increment = float(audit["net"]-reference["net"])
                        loss = float(np.maximum(-returns[chosen], 0).sum())
                        discrepancy = max(abs(increment-float(returns[chosen].sum())), abs(increment-cumulative["final"]), abs(increment-guarded["final_settled_increment"]), abs(loss-guarded["final_spent_loss"]))
                        verification["service_checks"] += 1
                        verification["max_service_accounting_error"] = max(verification["max_service_accounting_error"], discrepancy)
                        verification["min_ledger_capacity"] = min(verification["min_ledger_capacity"], min(l["capacity_after"] for l in guarded["ledger"]))
                        verification["worst_net_floor_violation"] = max(verification["worst_net_floor_violation"], max(0., -budget-cumulative["minimum"]))
                        if mode == "gross_loss":
                            verification["worst_gross_loss_violation"] = max(verification["worst_gross_loss_violation"], max(0., loss-budget), max(0., cumulative["maximum_prefix_negative_loss"]-budget))
                        assert discrepancy < 1e-8
                        assert cumulative["minimum"] >= -budget-1e-8
                        if mode == "gross_loss":
                            assert loss <= budget+1e-8
                            assert cumulative["maximum_prefix_negative_loss"] <= budget+1e-8
                        if budget == 0: assert not chosen.any()
                        cutoffs = sorted({0, trial["windows"]//2, max(0,trial["windows"]-1)})
                        for cutoff in cutoffs:
                            altered = replay(rows, trial["windows"], budget, mode, reserves, perturb_after=cutoff)
                            original_actions = [r["action"] for r in guarded["rows"] if r["k"] <= cutoff]
                            altered_actions = [r["action"] for r in altered["rows"] if r["k"] <= cutoff]
                            assert original_actions == altered_actions
                            verification["future_truth_invariance_checks"] += 1
                        trajectories.append(dict(dataset=name, seed=seed, arm=arm, mode=mode, reserve_mode=reserve_mode,
                            budget=budget, increment=increment, net=audit["net"], reference_net=reference["net"],
                            unguarded_increment=trial["results"][arm]["net"]-reference["net"],
                            negative_loss=loss, harmful=int(np.sum(returns[chosen] < 0)), beneficial=int(np.sum(returns[chosen] > 0)),
                            admissions=int(chosen.sum()), blocked=sum(r["proposed_action"] and not r["action"] for r in guarded["rows"]),
                            minimum_prefix_increment=cumulative["minimum"], actions=chosen.tolist(),
                            maximum_prefix_negative_loss=cumulative["maximum_prefix_negative_loss"],
                            independent_lease_increments=cumulative["independent_lease_increments"],
                            origins=[r["k"] for r in rows], admitted_returns=returns[chosen].tolist(),
                            ledger=guarded["ledger"], test_end_state=guarded["test_end_state"],
                            independent_service=audit, accounting_discrepancy=discrepancy))
            print("GUARD", name, seed, "complete", flush=True)
    verification["source_unchanged"] = original == {p.relative_to(package).as_posix():sha(p) for p in package.rglob("*") if p.is_file()}
    assert verification["source_unchanged"]
    verification["passed"] = all((verification["max_fork_error"]<1e-8, verification["max_reference_error"]<1e-8, verification["max_service_accounting_error"]<1e-8, verification["min_ledger_capacity"]>=-1e-8, verification["worst_net_floor_violation"]==0., verification["worst_gross_loss_violation"]==0.))
    dump(output / "results.json", dict(protocol_sha256=sha(protocol_path), script_sha256=sha(__file__), trajectories=trajectories))
    dump(output / "summary.json", compact_summary(trajectories))
    dump(output / "issued_reserves.json", issued_reserve_logs)
    dump(output / "verification.json", verification)
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
