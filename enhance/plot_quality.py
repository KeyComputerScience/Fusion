"""Plot all frozen quality scenarios and component ablations, including failures."""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
TASK_CACHE = Path(tempfile.gettempdir()) / "information-fusion-revision-matplotlib-cache"
TASK_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TASK_CACHE))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
ROWS = json.loads((ROOT / "results/quality/summary.json").read_text())
INDEX = {(row["scenario"], row["method"]): row for row in ROWS}
SCENARIOS = ["clean", "service_outage", "operating_noise", "workload_conflict", "simultaneous_loss"]
LABELS = ["Clean", "Svc loss", "Op noise", "Conflict", "All loss"]
COLORS = {"original_reliability": "#0072B2", "revised_residual_fusion": "#D55E00", "baseline_EF": "#009E73", "baseline_DS": "#6C4E91"}
NAMES = {"original_reliability": "Original reliability", "revised_residual_fusion": "Revised residual fusion",
         "baseline_EF": "Equal fusion", "baseline_DS": "Dempster (forecast only)"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "savefig.dpi": 180})


def save(figure, name):
    destination = ROOT / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(destination / f"{name}.{extension}", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def grouped(axis, key, title, ylabel, *, scale=1.0, methods=None, coverage=False):
    methods = methods or ["original_reliability", "revised_residual_fusion", "baseline_EF"]
    positions = np.arange(len(SCENARIOS))
    width = .78 / len(methods)
    for index, method in enumerate(methods):
        selected = [INDEX[scenario, method] for scenario in SCENARIOS]
        if coverage:
            values = [row[key] / row["event_trials_total"] * 100 for row in selected]
            errors = None
        else:
            values = [row[key + "_mean"] * scale for row in selected]
            errors = [row[key + "_sd"] * scale for row in selected]
        offset = positions + (index - (len(methods) - 1) / 2) * width
        axis.bar(offset, values, width=width, color=COLORS[method], edgecolor="white", linewidth=.5,
                 yerr=errors, error_kw={"elinewidth": .8, "capsize": 2, "capthick": .8})
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions, LABELS)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E6E7E9", linewidth=.7)
    axis.set_xlim(-.6, len(SCENARIOS) - .4)
    axis.set_ylim(bottom=0)
    if coverage:
        axis.set_ylim(0, 112)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.text(4, 8, "0/10", ha="center", fontsize=9, color="#333333",
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})


def interventions():
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    grouped(axes[0, 0], "forecast_mae_common", "A  Common-target forecast error", "MAE",
            methods=["original_reliability", "revised_residual_fusion", "baseline_EF", "baseline_DS"])
    grouped(axes[0, 1], "raw_alarm_fraction", "B  Nominal raw alarms", "Alarm windows (%)", scale=100)
    grouped(axes[0, 2], "admission_fraction", "C  Nominal gated admissions", "Admissible windows (%)", scale=100)
    grouped(axes[1, 0], "raw_events_detected_total", "D  Raw event coverage", "Event trials covered (%)", coverage=True)
    grouped(axes[1, 1], "admitted_events_detected_total", "E  Gated event coverage", "Event trials admitted (%)", coverage=True)
    grouped(axes[1, 2], "fallback_windows", "F  Explicit abstention", "Fallback windows / seed")
    axes[0, 1].text(.03, .98, "Conflict remains a failure case", transform=axes[0, 1].transAxes,
                    ha="left", va="top", fontsize=9, color="#444444")
    axes[1, 0].text(.03, .98, "All loss: every method misses both events", transform=axes[1, 0].transAxes,
                    ha="left", va="top", fontsize=8.5, color="#444444")
    legend = [Patch(facecolor=COLORS[method], label=NAMES[method])
              for method in ["original_reliability", "revised_residual_fusion", "baseline_EF", "baseline_DS"]]
    figure.legend(handles=legend, ncol=4, loc="upper center", bbox_to_anchor=(.5, .99), frameon=False)
    figure.suptitle("Frozen observer-quality interventions on executed synthetic NoRT logs", fontsize=16, y=1.035, fontweight="bold")
    figure.text(.02, .015,
                 "Five paired seeds; bars show means and SD. Coverage counts two distinct injected events repeated across five seeds (10 trials).\n"
                 "MAE uses 58 common forecast origins per seed, or 46 under all-source loss; omitted windows are not zero-error successes.\n"
                 "Svc loss: service-feedback outage; Op noise: corrupted operating telemetry; Conflict: corrupted workload reports.\n"
                 "These are diagnostic replay outputs, not deployments or service gains. Dempster supports are not calibrated class posteriors.",
                 ha="left", va="bottom", fontsize=9, color="#40444A")
    figure.subplots_adjust(top=.87, bottom=.15, wspace=.27, hspace=.40)
    save(figure, "quality_interventions")


def ablations():
    components = ["N", "A", "V", "E/R", "Cap", "D", "Missing", "U gate", "Inertia"]
    mappings = {
        "original": ["original_without_qN", "original_without_qA", "original_without_qV", "original_without_qE",
                     "original_without_cap", "original_without_disagreement", "original_without_missing_penalty", "original_without_U_gate", None],
        "revised": ["revised_without_qN", "revised_without_qA", "revised_without_qV", "revised_without_residual_consistency",
                    "revised_without_cap", "revised_without_disagreement", "revised_without_missing_penalty", "revised_without_U_gate", "revised_without_inertia"],
    }
    row_labels, errors, alarms, admitted = [], [], [], []
    for family, baseline in [("original", "original_reliability"), ("revised", "revised_residual_fusion")]:
        for scenario, label in zip(SCENARIOS, ["clean", "service outage", "operating noise", "conflict", "all missing"]):
            base = INDEX[scenario, baseline]
            row_labels.append(("Original" if family == "original" else "Revised") + " / " + label)
            delta_error, delta_alarm, delta_admitted = [], [], []
            for method in mappings[family]:
                if method is None:
                    delta_error.append(np.nan); delta_alarm.append(np.nan); delta_admitted.append(np.nan)
                    continue
                item = INDEX[scenario, method]
                delta_error.append((item["forecast_mae_common_mean"] - base["forecast_mae_common_mean"]) * 1000)
                delta_alarm.append((item["raw_alarm_fraction_mean"] - base["raw_alarm_fraction_mean"]) * 100)
                delta_admitted.append((item["admitted_events_detected_total"] - base["admitted_events_detected_total"]) / base["event_trials_total"] * 100)
            errors.append(delta_error); alarms.append(delta_alarm); admitted.append(delta_admitted)
    figure, axes = plt.subplots(1, 3, figsize=(18.7, 7.4))
    matrices = [np.asarray(errors), np.asarray(alarms), np.asarray(admitted)]
    titles = ["A  Forecast MAE change (x 0.001)", "B  Raw-alarm change (percentage points)", "C  Event-admission change (percentage points)"]
    for index, (axis, matrix, title) in enumerate(zip(axes, matrices, titles)):
        maximum = max(float(np.nanmax(np.abs(matrix))), .01)
        cmap = plt.get_cmap("RdBu_r" if index < 2 else "RdBu").copy()
        cmap.set_bad("#E5E7EB")
        heat = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap,
                           norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum))
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=12)
        axis.set_xticks(np.arange(len(components)), components, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(row_labels)), row_labels if index == 0 else [""] * len(row_labels))
        axis.tick_params(length=0)
        axis.axhline(4.5, color="#1F2937", linewidth=1.5)
        axis.set_xticks(np.arange(-.5, len(components), 1), minor=True)
        axis.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=.8)
        axis.tick_params(which="minor", length=0)
        for row in range(len(row_labels)):
            for column in range(len(components)):
                value = matrix[row, column]
                displayed = 0.0 if np.isfinite(value) and abs(value) < (.005 if index == 0 else .05) else value
                text = "NA" if not np.isfinite(value) else (f"{displayed:.2f}" if index == 0 else f"{displayed:.1f}")
                color = "white" if np.isfinite(value) and abs(value) > .6 * maximum else "#222222"
                axis.text(column, row, text, ha="center", va="center", fontsize=7.5, color=color)
        colorbar = figure.colorbar(heat, ax=axis, fraction=.046, pad=.025)
        colorbar.ax.tick_params(labelsize=8)
    figure.suptitle("Removing a component: complete ablation results, including null and negative effects", fontsize=15, y=.995, fontweight="bold")
    figure.text(.01, .012,
                 "Deltas are ablated minus full method, computed on the same forecast windows and paired seeds. Blue indicates improvement; red indicates harm.\n"
                 "N: support; A: freshness; V: bootstrap variability; E/R: original delayed consistency / revised residual consistency; D: disagreement.\n"
                 "Inertia is absent from the original method. A small or zero delta does not demonstrate component necessity.\n"
                 "Higher event admission can accompany more false admissions; see the intervention figure. All-source loss still misses every event.",
                 ha="left", va="bottom", fontsize=9, color="#40444A")
    figure.subplots_adjust(left=.155, right=.985, top=.86, bottom=.20, wspace=.22)
    save(figure, "component_ablations")


if __name__ == "__main__":
    style()
    interventions()
    ablations()
    print("Saved quality_interventions and component_ablations as PNG, PDF and SVG.")
