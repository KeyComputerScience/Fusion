#!/usr/bin/env python3
"""
Enhanced fusion experiments for the Information Fusion manuscript.

What this script does
---------------------
1. Implements the proposed reliability -> fusion -> uncertainty representation.
2. Implements comparison baselines:
   Equal, Fixed, InvVar, Entropy, DS-Reliability, Predictive.
3. Loads Alibaba Cluster Trace v2018 machine_usage.csv and batch_task.csv.
4. Builds a 60-s trace-driven stream with a 20-h calibration prefix and 50-h evaluation.
5. Generates workload/state evidence, corruption scenarios, sensitivity sweeps,
   Pareto summaries, and LaTeX-ready CSV/TEX outputs.

Important
---------
This script does NOT invent trace results. Real trace tables are produced only
when the official Alibaba files are supplied with --machine-usage and --batch-task.

The Alibaba v2018 public documentation states that the trace contains about
4000 machines over 8 days. Timestamps are seconds relative to trace start.
Some resource values are normalized. Check the released schema for the exact
column order of your downloaded copy.
"""

from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = average_precision_score = None


EPS = 1e-12
SOURCES = ("workload", "state", "performance")


@dataclass
class FusionConfig:
    n0: float = 20.0
    tau_age: float = 50.0
    tau_error: float = 0.20
    eps_var: float = 0.0025
    weight_cap: float = 0.70
    kappa_missing: float = 0.50
    kappa_entropy: float = 0.15
    ewma_error: float = 0.15
    gamma_on: float = 0.15
    gamma_off: float = 0.08
    u_max: float = 0.80
    coverage_min: float = 2.0 / 3.0
    window: int = 50
    bootstrap_reps: int = 32
    bootstrap_block: int = 5


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = p / max(p.sum(), EPS)
    q = q / max(q.sum(), EPS)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], EPS))))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m)) / math.log(2.0)


def capped_kl_projection(pi: np.ndarray, valid: np.ndarray, cap: float) -> np.ndarray:
    """Solve alpha_s=min(cap_eff, z*pi_s), sum alpha=1 on valid sources."""
    pi = np.asarray(pi, float)
    valid = np.asarray(valid, bool)
    out = np.zeros_like(pi)
    idx = np.flatnonzero(valid & (pi > 0))
    if len(idx) == 0:
        return out
    cap_eff = max(float(cap), 1.0 / len(idx))
    p = pi[idx]
    p = p / p.sum()

    lo, hi = 0.0, 1.0
    while np.minimum(cap_eff, hi * p).sum() < 1.0:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.minimum(cap_eff, mid * p).sum() < 1.0:
            lo = mid
        else:
            hi = mid
    a = np.minimum(cap_eff, hi * p)
    a /= a.sum()
    out[idx] = a
    return out


def normalized_weight_entropy(alpha: np.ndarray, valid: np.ndarray) -> float:
    a = np.asarray(alpha, float)[np.asarray(valid, bool)]
    a = a[a > 0]
    if len(a) <= 1:
        return 0.0
    h = -float(np.sum(a * np.log(a + EPS)))
    return h / math.log(len(a))


def proposed_weights(scores, valid, n_eff, age, variance, pred_error, cfg: FusionConfig):
    scores = np.asarray(scores, float)
    valid = np.asarray(valid, bool)
    n_eff = np.asarray(n_eff, float)
    age = np.asarray(age, float)
    variance = np.asarray(variance, float)
    pred_error = np.asarray(pred_error, float)

    reliability = (
        n_eff / (n_eff + cfg.n0)
        * np.exp(-age / cfg.tau_age)
        * np.exp(-pred_error / cfg.tau_error)
        / (variance + cfg.eps_var)
    )
    reliability[~valid] = 0.0
    if reliability.sum() <= EPS:
        return np.zeros_like(scores), reliability
    pi = reliability / reliability.sum()
    return capped_kl_projection(pi, valid, cfg.weight_cap), reliability


def baseline_weights(name, scores, valid, variance, pred_error, fixed=None):
    d = np.asarray(scores, float)
    v = np.asarray(valid, bool)
    var = np.asarray(variance, float)
    err = np.asarray(pred_error, float)
    w = np.zeros_like(d)

    if not np.any(v):
        return w

    if name == "Equal":
        w[v] = 1.0 / v.sum()
    elif name == "Fixed":
        f = np.asarray(fixed if fixed is not None else [0.40, 0.35, 0.25], float)
        f[~v] = 0.0
        w = f / max(f.sum(), EPS)
    elif name == "InvVar":
        q = 1.0 / np.maximum(var, 1e-6)
        q[~v] = 0.0
        w = q / max(q.sum(), EPS)
    elif name == "Entropy":
        # Larger |d-0.5| is treated as more informative; entropy regularizes near-ambiguous evidence.
        p = np.clip(d, 1e-6, 1 - 1e-6)
        h = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / math.log(2.0)
        q = 1.0 - h + 1e-3
        q[~v] = 0.0
        w = q / max(q.sum(), EPS)
    elif name == "Predictive":
        q = np.exp(-err / 0.20)
        q[~v] = 0.0
        w = q / max(q.sum(), EPS)
    elif name == "DS-Reliability":
        # Reliability-discounted evidence surrogate:
        # discount each scalar evidence by predictive consistency and inverse variance.
        q = np.exp(-err / 0.20) / np.maximum(var, 1e-4)
        q[~v] = 0.0
        w = q / max(q.sum(), EPS)
    else:
        raise ValueError(name)
    return w


def fuse(scores, alpha, valid, cfg: FusionConfig):
    d = np.asarray(scores, float)
    a = np.asarray(alpha, float)
    valid = np.asarray(valid, bool)
    if a.sum() <= EPS:
        return dict(gamma=np.nan, disagreement=0.25, coverage=0.0, uncertainty=1.0)
    gamma = float(np.sum(a * d))
    disagreement = float(np.sum(a * (d - gamma) ** 2))
    coverage = float(valid.mean())
    ent = normalized_weight_entropy(a, valid)
    uncertainty = min(
        1.0,
        4.0 * disagreement
        + cfg.kappa_missing * (1.0 - coverage)
        + cfg.kappa_entropy * ent,
    )
    return dict(
        gamma=gamma,
        disagreement=disagreement,
        coverage=coverage,
        uncertainty=uncertainty,
    )


def moving_block_bootstrap_variance(x: np.ndarray, reps=32, block=5, rng=None) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return 0.05
    rng = np.random.default_rng(rng)
    n = len(x)
    vals = []
    for _ in range(reps):
        out = []
        while len(out) < n:
            start = int(rng.integers(0, n))
            ids = (start + np.arange(block)) % n
            out.extend(x[ids].tolist())
        vals.append(np.mean(out[:n]))
    return float(np.var(vals, ddof=1))


# ------------------------ Alibaba trace loader ------------------------

MACHINE_USAGE_CANDIDATES = {
    "machine_id": ["machine_id", "machineid", "machine"],
    "time_stamp": ["time_stamp", "timestamp", "time"],
    "cpu": ["cpu_util_percent", "cpu_util", "cpu"],
    "mem": ["mem_util_percent", "mem_util", "memory", "mem"],
    "net_in": ["net_in", "network_in"],
    "net_out": ["net_out", "network_out"],
}

BATCH_TASK_CANDIDATES = {
    "create_time": ["create_timestamp", "create_time", "start_time", "time_stamp"],
    "cpu": ["plan_cpu", "cpu_request", "cpu"],
    "instances": ["instance_num", "instances", "instance_count"],
}


def _find_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def read_csv_flexible(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Headerless official copies are common. If no useful names are detected,
    # the user should pass a preprocessed CSV with schema names.
    return df


def load_alibaba(machine_usage: Path, batch_task: Path,
                 slot_seconds=60, prefix_slots=1200, eval_slots=3000,
                 n_machines=100) -> pd.DataFrame:
    mu = read_csv_flexible(machine_usage)
    bt = read_csv_flexible(batch_task)

    mc = {k: _find_col(mu, v) for k, v in MACHINE_USAGE_CANDIDATES.items()}
    bc = {k: _find_col(bt, v) for k, v in BATCH_TASK_CANDIDATES.items()}

    required_m = ["machine_id", "time_stamp", "cpu", "mem"]
    required_b = ["create_time"]
    if any(mc[k] is None for k in required_m):
        raise ValueError(
            "machine_usage.csv columns were not recognized. "
            "Provide a schema-named/preprocessed CSV containing machine_id, time_stamp, "
            "cpu_util_percent, mem_util_percent (net_in/net_out optional)."
        )
    if any(bc[k] is None for k in required_b):
        raise ValueError(
            "batch_task.csv columns were not recognized. "
            "Provide a schema-named/preprocessed CSV containing create_timestamp "
            "(plan_cpu and instance_num are optional)."
        )

    horizon = prefix_slots + eval_slots
    end_sec = horizon * slot_seconds
    mu = mu[(pd.to_numeric(mu[mc["time_stamp"]], errors="coerce") >= 0) &
            (pd.to_numeric(mu[mc["time_stamp"]], errors="coerce") < end_sec)].copy()
    mu["_slot"] = (pd.to_numeric(mu[mc["time_stamp"]], errors="coerce") // slot_seconds).astype("Int64")

    prefix = mu[mu["_slot"] < prefix_slots]
    coverage = prefix.groupby(mc["machine_id"])["_slot"].nunique().sort_values(ascending=False)
    selected = coverage.head(n_machines).index
    mu = mu[mu[mc["machine_id"]].isin(selected)].copy()

    for key in ["cpu", "mem", "net_in", "net_out"]:
        if mc.get(key) is not None:
            mu[key] = pd.to_numeric(mu[mc[key]], errors="coerce")
    # Public trace utilization fields are commonly on a 0..100 normalized scale.
    for key in ["cpu", "mem"]:
        if key in mu:
            mu[key] = mu[key] / 100.0

    agg_cols = [c for c in ["cpu", "mem", "net_in", "net_out"] if c in mu]
    state = mu.groupby("_slot")[agg_cols].mean().reindex(range(horizon))

    bt_time = pd.to_numeric(bt[bc["create_time"]], errors="coerce")
    bt = bt[(bt_time >= 0) & (bt_time < end_sec)].copy()
    bt["_slot"] = (pd.to_numeric(bt[bc["create_time"]], errors="coerce") // slot_seconds).astype("Int64")
    arrivals = bt.groupby("_slot").size().reindex(range(horizon), fill_value=0).astype(float)

    if bc.get("cpu") is not None:
        bt["_cpu_req"] = pd.to_numeric(bt[bc["cpu"]], errors="coerce").fillna(0.0)
        cpu_req = bt.groupby("_slot")["_cpu_req"].sum().reindex(range(horizon), fill_value=0.0)
    else:
        cpu_req = arrivals.copy()

    if bc.get("instances") is not None:
        bt["_inst"] = pd.to_numeric(bt[bc["instances"]], errors="coerce").fillna(1.0)
        inst = bt.groupby("_slot")["_inst"].sum().reindex(range(horizon), fill_value=0.0)
    else:
        inst = arrivals.copy()

    stream = state.copy()
    stream["arrivals"] = arrivals.values
    stream["cpu_request"] = cpu_req.values
    stream["instances"] = inst.values

    # Limited forward fill; longer gaps remain NaN and become missing evidence.
    stream[agg_cols] = stream[agg_cols].ffill(limit=2)
    stream.index.name = "slot"
    return stream


def prefix_normalize(stream: pd.DataFrame, prefix_slots=1200):
    ref = stream.iloc[:prefix_slots]
    cols = [c for c in ["cpu", "mem", "net_in", "net_out", "arrivals"] if c in stream]
    mu = ref[cols].mean()
    sd = ref[cols].std().replace(0, 1.0)
    z = (stream[cols] - mu) / sd
    return z, mu, sd


def workload_regimes(arrivals: pd.Series, prefix_slots=1200):
    q = arrivals.iloc[:prefix_slots].quantile([0.33, 0.67]).values
    x = arrivals.to_numpy(float)
    return np.digitize(x, q, right=True)


def build_window_evidence(stream: pd.DataFrame, cfg: FusionConfig, prefix_slots=1200):
    z, _, _ = prefix_normalize(stream, prefix_slots)
    regimes = workload_regimes(stream["arrivals"], prefix_slots)
    eval_stream = stream.iloc[prefix_slots:].reset_index(drop=True)
    z_eval = z.iloc[prefix_slots:].reset_index(drop=True)
    reg_eval = regimes[prefix_slots:]

    # Performance proxy from observable trace pressure:
    # higher free capacity and lower arrival pressure -> higher service score.
    cpu = eval_stream["cpu"].fillna(eval_stream["cpu"].median()).clip(0, 1)
    mem = eval_stream["mem"].fillna(eval_stream["mem"].median()).clip(0, 1)
    arr_z = z_eval["arrivals"].fillna(0)
    perf = clip01(0.75 - 0.30 * cpu - 0.15 * mem - 0.08 * np.maximum(arr_z, 0))
    perf = pd.Series(perf)

    rows = []
    W = cfg.window
    nwin = len(eval_stream) // W
    for k in range(1, nwin):
        a0, a1 = (k - 1) * W, k * W
        b0, b1 = k * W, (k + 1) * W

        # workload JSD
        p = np.bincount(reg_eval[a0:a1], minlength=3)
        q = np.bincount(reg_eval[b0:b1], minlength=3)
        d1 = jsd(p, q)

        # operating-state mean shift
        state_cols = [c for c in ["cpu", "mem", "net_in", "net_out", "arrivals"] if c in z_eval]
        za, zb = z_eval.iloc[a0:a1][state_cols], z_eval.iloc[b0:b1][state_cols]
        common_cols = [c for c in state_cols if za[c].notna().sum() >= 10 and zb[c].notna().sum() >= 10]
        if common_cols:
            d2 = float(np.mean(np.minimum(
                1.0,
                np.abs(zb[common_cols].mean().to_numpy() - za[common_cols].mean().to_numpy())
            )))
            state_valid = True
        else:
            d2, state_valid = 0.0, False

        # performance degradation
        pa, pb = perf.iloc[a0:a1], perf.iloc[b0:b1]
        d3 = float(min(1.0, max(0.0, pa.mean() - pb.mean()) / (abs(pa.mean()) + 1e-6)))

        valid = np.array([True, state_valid, True], bool)
        n_eff = np.array([W, min(za.notna().all(axis=1).sum(), zb.notna().all(axis=1).sum()), W], float)
        age = np.zeros(3)
        variance = np.array([
            moving_block_bootstrap_variance(reg_eval[b0:b1].astype(float), cfg.bootstrap_reps, cfg.bootstrap_block, k),
            moving_block_bootstrap_variance(zb[common_cols].mean(axis=1).to_numpy() if common_cols else np.array([]),
                                            cfg.bootstrap_reps, cfg.bootstrap_block, k + 100),
            moving_block_bootstrap_variance(pb.to_numpy(), cfg.bootstrap_reps, cfg.bootstrap_block, k + 200),
        ])
        rows.append(dict(
            window=k,
            d1=d1, d2=d2, d3=d3,
            m1=int(valid[0]), m2=int(valid[1]), m3=int(valid[2]),
            n1=n_eff[0], n2=n_eff[1], n3=n_eff[2],
            a1=age[0], a2=age[1], a3=age[2],
            v1=variance[0], v2=variance[1], v3=variance[2],
        ))
    return pd.DataFrame(rows)


def evaluate_fusion_table(ev: pd.DataFrame, cfg: FusionConfig):
    methods = ["Equal", "Fixed", "InvVar", "Entropy", "DS-Reliability", "Predictive", "Proposed"]
    pred_err = np.full(3, 0.05)
    out = []
    for method in methods:
        records = []
        local_err = pred_err.copy()
        for _, r in ev.iterrows():
            scores = np.array([r.d1, r.d2, r.d3], float)
            valid = np.array([r.m1, r.m2, r.m3], bool)
            var = np.array([r.v1, r.v2, r.v3], float)
            n_eff = np.array([r.n1, r.n2, r.n3], float)
            age = np.array([r.a1, r.a2, r.a3], float)

            if method == "Proposed":
                alpha, _ = proposed_weights(scores, valid, n_eff, age, var, local_err, cfg)
            else:
                alpha = baseline_weights(method, scores, valid, var, local_err)

            f = fuse(scores, alpha, valid, cfg)
            records.append({**f, "window": int(r.window), "method": method})

            # delayed predictive-consistency update: each source predicts next performance degradation.
            target = scores[2]
            local_err = (1 - cfg.ewma_error) * local_err + cfg.ewma_error * (scores - target) ** 2

        out.append(pd.DataFrame(records))
    return pd.concat(out, ignore_index=True)


# ------------------------ stress tests ------------------------

def corrupt_evidence(ev: pd.DataFrame, kind: str, level: float, seed=0):
    rng = np.random.default_rng(seed)
    x = ev.copy()
    n = len(x)

    if kind == "missing":
        for s in [1, 2, 3]:
            mask = rng.random(n) < level
            x.loc[mask, f"m{s}"] = 0
    elif kind == "noise":
        x["d2"] = clip01(x["d2"].to_numpy() + rng.normal(0, level, n))
    elif kind == "conflict":
        # Invert workload evidence on a fraction of windows.
        mask = rng.random(n) < level
        x.loc[mask, "d1"] = 1.0 - x.loc[mask, "d1"]
    elif kind == "delay":
        lag = int(level)
        if lag > 0:
            x["d3"] = x["d3"].shift(lag).fillna(0.0)
            x["m3"] = x["m3"].shift(lag).fillna(0).astype(int)
    else:
        raise ValueError(kind)
    return x


def run_stress(ev, cfg, outdir: Path):
    rows = []
    grids = {
        "missing": [0.0, 0.1, 0.2, 0.3, 0.4],
        "noise": [0.0, 0.05, 0.10, 0.20, 0.30],
        "conflict": [0.0, 0.25, 0.50, 0.75, 1.0],
        "delay": [0, 1, 2, 3, 4],
    }
    for kind, levels in grids.items():
        for level in levels:
            for seed in range(10):
                ce = corrupt_evidence(ev, kind, level, seed)
                ft = evaluate_fusion_table(ce, cfg)
                for method, g in ft.groupby("method"):
                    rows.append({
                        "stress": kind, "level": level, "seed": seed, "method": method,
                        "mean_gamma": g.gamma.mean(),
                        "mean_uncertainty": g.uncertainty.mean(),
                        "mean_coverage": g.coverage.mean(),
                        "alarm_rate": (g.gamma >= cfg.gamma_on).mean(),
                    })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "stress_results.csv", index=False)
    return df


def run_sensitivity(ev, cfg, outdir: Path):
    sweeps = {
        "weight_cap": [0.45, 0.55, 0.65, 0.70, 0.80, 0.90, 1.0],
        "kappa_missing": [0.0, 0.25, 0.50, 0.75, 1.0],
        "tau_age": [10, 25, 50, 100, 200],
        "tau_error": [0.05, 0.10, 0.20, 0.40, 0.80],
    }
    rows = []
    for field, values in sweeps.items():
        for val in values:
            c = replace(cfg, **{field: val})
            ft = evaluate_fusion_table(ev, c)
            g = ft[ft.method == "Proposed"]
            rows.append({
                "parameter": field, "value": val,
                "mean_gamma": g.gamma.mean(),
                "mean_uncertainty": g.uncertainty.mean(),
                "alarm_rate": (g.gamma >= c.gamma_on).mean(),
                "mean_coverage": g.coverage.mean(),
            })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sensitivity_results.csv", index=False)
    return df


# ------------------------ Pareto ------------------------

def pareto_front(df: pd.DataFrame, return_col="return", cost_col="training_cost"):
    """Maximize return, minimize cost."""
    keep = []
    for i, r in df.iterrows():
        dominated = False
        for j, q in df.iterrows():
            if i == j:
                continue
            if (q[return_col] >= r[return_col] and q[cost_col] <= r[cost_col] and
                (q[return_col] > r[return_col] or q[cost_col] < r[cost_col])):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return df.loc[keep].copy()


def existing_pareto(outdir: Path):
    rows = [
        ("DQN", "Proposed", 1905.93, 0.70, 2.0),
        ("DQN", "Periodic", 1911.98, 2.80, 14.0),
        ("DQN", "Deficit", 1893.85, 7.70, 17.0),
        ("PPO", "Proposed", 1918.98, 0.70, 2.0),
        ("PPO", "Periodic", 1917.09, 2.80, 14.0),
        ("PPO", "Deficit", 1880.28, 7.70, 17.0),
    ]
    df = pd.DataFrame(rows, columns=["backbone", "method", "return", "training_cost", "deployments"])
    fronts = []
    for b, g in df.groupby("backbone"):
        fronts.append(pareto_front(g))
    front = pd.concat(fronts)
    df.to_csv(outdir / "existing_service_cost_points.csv", index=False)
    front.to_csv(outdir / "existing_pareto_front.csv", index=False)
    return df, front


def write_latex_summary(fusion_df: pd.DataFrame, outdir: Path):
    summary = fusion_df.groupby("method").agg(
        mean_gamma=("gamma", "mean"),
        mean_uncertainty=("uncertainty", "mean"),
        mean_coverage=("coverage", "mean"),
    ).reset_index()

    lines = []
    for _, r in summary.iterrows():
        lines.append(
            f"{r['method']} & {r['mean_gamma']:.4f} & "
            f"{r['mean_uncertainty']:.4f} & {r['mean_coverage']:.4f} \\\\"
        )
    (outdir / "trace_fusion_rows.tex").write_text("\n".join(lines), encoding="utf-8")
    summary.to_csv(outdir / "trace_fusion_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine-usage", type=Path)
    ap.add_argument("--batch-task", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("fusion_results"))
    ap.add_argument("--machines", type=int, default=100)
    ap.add_argument("--prefix-slots", type=int, default=1200)
    ap.add_argument("--eval-slots", type=int, default=3000)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg = FusionConfig()

    # Always emit the existing Pareto calculation from manuscript-reported values.
    _, front = existing_pareto(args.outdir)
    print("Existing controlled Pareto set:")
    print(front.to_string(index=False))

    if args.machine_usage is None or args.batch_task is None:
        print("\nNo Alibaba trace files supplied.")
        print("Real trace results were NOT generated.")
        print("Run, for example:")
        print("  python fusion_enhanced_experiments.py "
              "--machine-usage machine_usage.csv --batch-task batch_task.csv "
              "--outdir fusion_results")
        return

    stream = load_alibaba(
        args.machine_usage, args.batch_task,
        prefix_slots=args.prefix_slots,
        eval_slots=args.eval_slots,
        n_machines=args.machines,
    )
    stream.to_csv(args.outdir / "alibaba_stream_60s.csv")
    ev = build_window_evidence(stream, cfg, prefix_slots=args.prefix_slots)
    ev.to_csv(args.outdir / "trace_window_evidence.csv", index=False)

    fused = evaluate_fusion_table(ev, cfg)
    fused.to_csv(args.outdir / "trace_fusion_windows.csv", index=False)
    write_latex_summary(fused, args.outdir)

    run_stress(ev, cfg, args.outdir)
    run_sensitivity(ev, cfg, args.outdir)

    metadata = {
        "machines": args.machines,
        "slot_seconds": 60,
        "prefix_slots": args.prefix_slots,
        "eval_slots": args.eval_slots,
        "window": cfg.window,
        "methods": ["Equal", "Fixed", "InvVar", "Entropy",
                    "DS-Reliability", "Predictive", "Proposed"],
    }
    (args.outdir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nOutputs written to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
