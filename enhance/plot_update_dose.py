"""Publication exports from every retained contextual-bandit update-budget run."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results/update_necessity/budget_sensitivity"
OUTPUT = ROOT / "figures"


def main():
    analysis = json.loads((SOURCE / "analysis.json").read_text())
    raw = json.loads((SOURCE / "all_budget_seed_results.json").read_text())
    groups = {(r["backbone"], r["scenario"], r["method"], r["budget_updates"]): r for r in analysis["groups"]}
    colors = {"periodic": "#0072B2", "observed_loss": "#D55E00"}
    method_names = {"periodic": "Periodic", "observed_loss": "Observed loss"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 11, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.1))
    budgets = [8, 32, 128]
    horizontal = np.arange(3)
    panels = "abcdef"

    def values(backbone, scenario, method, budget, metric):
        selected = sorted((r for r in raw if (r["backbone"], r["scenario"], r["method"], r["budget_updates"])
                           == (backbone, scenario, method, budget)), key=lambda r: r["seed"])
        if len(selected) != 5:
            raise AssertionError("Plot must include all five seeds at every displayed condition")
        return np.asarray([r[metric] for r in selected])

    for row, backbone in enumerate(["dqn", "ppo"]):
        for column in range(3):
            ax = axes[row, column]
            ax.set_xlim(-0.18, 2.18)
            ax.set_xticks(horizontal, [str(budget) for budget in budgets])
            ax.set_xlabel("Gradient updates / deployment-delay slots")
            ax.grid(axis="y", color="#E3E5E7", linewidth=0.65)
            ax.set_axisbelow(True)
            ax.text(-0.13, 1.04, f"({panels[row*3+column]})", transform=ax.transAxes,
                    fontsize=10, fontweight="bold")
        for column, scenario in [(0, "relation_shift"), (2, "no_shift")]:
            ax = axes[row, column]
            for method in ["periodic", "observed_loss"]:
                for metric, linestyle, marker, offset in [
                        ("post_boundary_accuracy", "--", "o", -0.025),
                        ("last_512_accuracy", "-", "s", 0.025)]:
                    means = [100 * groups[backbone, scenario, method, budget][metric]["mean"] for budget in budgets]
                    ax.plot(horizontal, means, color=colors[method], linestyle=linestyle,
                            marker=marker, markersize=4.4, linewidth=1.45,
                            markerfacecolor="white" if metric == "last_512_accuracy" else colors[method],
                            label=f"{method_names[method]} / {'post' if metric=='post_boundary_accuracy' else 'final'}")
                    for position, budget in enumerate(budgets):
                        all_seeds = 100 * values(backbone, scenario, method, budget, metric)
                        jitter = np.linspace(-0.065, 0.065, len(all_seeds)) + offset
                        ax.scatter(position + jitter, all_seeds, s=11, color=colors[method],
                                   marker=marker, alpha=0.40, edgecolors="none", zorder=2)
            # Both no-update reference metrics are retained; gray band shows all
            # seed values for the post-boundary reference, not a confidence band.
            reference = values(backbone, scenario, "no_rt", 8, "post_boundary_accuracy") * 100
            tail_reference = values(backbone, scenario, "no_rt", 8, "last_512_accuracy") * 100
            ax.axhspan(reference.min(), reference.max(), color="#7E8790", alpha=0.12)
            ax.axhline(reference.mean(), color="#4D5156", linestyle=":", linewidth=1.2)
            ax.axhline(tail_reference.mean(), color="#4D5156", linestyle="-.", linewidth=0.9)
            ax.set_ylabel("Correct actions (%)")
            ax.set_title(f"{backbone.upper()}: {'hidden relation shift' if scenario=='relation_shift' else 'no-shift negative control'}")
            if scenario == "relation_shift":
                ax.set_ylim(-2, 103)
            elif backbone == "dqn":
                ax.set_ylim(89, 100)
            else:
                ax.set_ylim(65, 102)
            ax.legend(frameon=False, fontsize=7.2, loc="upper left" if scenario=="relation_shift" else "lower left",
                      handlelength=2.4, labelspacing=0.25)
        cost_ax = axes[row, 1]
        for method in ["periodic", "observed_loss"]:
            for scenario, linestyle, marker in [("relation_shift", "-", "o"), ("no_shift", "--", "^")]:
                means = [groups[backbone, scenario, method, budget]["normalized_training_cost"]["mean"] for budget in budgets]
                cost_ax.plot(horizontal, means, color=colors[method], linestyle=linestyle,
                             marker=marker, markersize=4.7, linewidth=1.45,
                             markerfacecolor="white" if scenario=="no_shift" else colors[method],
                             label=f"{method_names[method]} / {'shift' if scenario=='relation_shift' else 'no shift'}")
                for position, budget in enumerate(budgets):
                    all_seeds = values(backbone, scenario, method, budget, "normalized_training_cost")
                    cost_ax.scatter(position + np.linspace(-0.065, 0.065, 5), all_seeds,
                                    s=10, marker=marker, color=colors[method], alpha=0.4, edgecolors="none")
        cost_ax.axhline(0, color="#4D5156", linestyle=":", linewidth=1.1)
        cost_ax.set_ylim(-0.07, 1.67)
        cost_ax.set_ylabel("Actual worker updates / 1,000")
        cost_ax.set_title(f"{backbone.upper()}: paid training budget")
        cost_ax.legend(frameon=False, fontsize=7.2, loc="upper left", handlelength=2.4, labelspacing=0.25)
    fig.suptitle("Training-budget diagnostic: all doses and failure outcomes retained", y=0.985,
                 fontsize=13, fontweight="medium")
    fig.text(0.5, 0.947, "32/128-dose extension is post-hoc; 8-dose runs and all shared no-update baselines are unchanged",
             ha="center", fontsize=9, color="#474C52")
    handles = [Line2D([], [], color="#0072B2", marker="o", linestyle="--", label="Post-boundary mean"),
               Line2D([], [], color="#0072B2", marker="s", markerfacecolor="white", label="Final-512 mean"),
               Line2D([], [], color="#777777", marker="o", linestyle="None", markersize=4, alpha=.5, label="All 5 seed outcomes"),
               Line2D([], [], color="#4D5156", linestyle=":", label="No-update post-boundary baseline"),
               Line2D([], [], color="#4D5156", linestyle="-.", label="No-update final-512 baseline")]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.004))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.885, bottom=0.12, hspace=0.42, wspace=0.27)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUTPUT / f"dose_response.{suffix}", dpi=240 if suffix=="png" else None,
                    bbox_inches="tight", pad_inches=0.1)
    caption = """All retained training-budget doses in the separate synthetic contextual-bandit diagnostic. Lines show means and translucent points show every one of the five paired seeds; the gray band spans the no-update post-boundary seed range. Correct-action panels retain both complete post-boundary and final-512 accuracy. Training cost is actual worker gradient updates divided by 1000, including computed pending workers, and each dose has an equal-number service-slot deployment delay. Negative controls are included. The 32/128-step extension is post-hoc; it does not replace the original eight-step failures or the queue-service negative results. The no-shift accuracy panels use different vertical ranges to expose degradation.\n"""
    (OUTPUT / "dose_response.caption.md").write_text(caption)
    print("Saved dose_response.png, .pdf, .svg and caption.")


if __name__ == "__main__":
    main()
