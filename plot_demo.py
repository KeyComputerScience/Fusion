"""White-background black-ink figure; only SYNTHETIC functional traces."""
import argparse
import csv
from pathlib import Path
import os
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="example_results/decisions.csv")
    parser.add_argument("--output", default="example_results/synthetic_trace")
    parser.add_argument("--font-dir", help="Folder containing licensed times.ttf and timesbd.ttf")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fusion-matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if args.font_dir:
        font_manager.fontManager.addfont(str(Path(args.font_dir) / "times.ttf"))
        font_manager.fontManager.addfont(str(Path(args.font_dir) / "timesbd.ttf"))
    # Fail clearly instead of silently rendering a substitute face.
    font_manager.findfont(font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10,
                         "text.color": "black", "axes.labelcolor": "black",
                         "xtick.color": "black", "ytick.color": "black",
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})
    with open(args.input, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    slots = [int(row["slot"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True, layout="constrained")
    axes[0].plot(slots, [float(row["gamma"]) for row in rows], color="black", label="Fused drift", linewidth=1.2)
    axes[0].plot(slots, [float(row["uncertainty"]) for row in rows], color="black", linestyle="--", label="Disagreement / missing sources", linewidth=1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(-0.03, 1.08)
    axes[0].legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
                   loc="upper right", fontsize=8)
    for name, style in zip(("workload", "operating", "performance"), ("-", "--", ":")):
        axes[1].plot(slots, [float(row[f"weight_{name}"]) for row in rows], color="black", linestyle=style, label=name.capitalize(), linewidth=1.0)
    axes[1].set_ylabel("Fusion weight")
    axes[1].set_ylim(-0.03, 1.08)
    axes[1].legend(frameon=False, loc="upper right", ncol=3, fontsize=8)
    profile_index = [int(row["training_profile"].replace("tr", "")) if row["training_profile"] else float("nan") for row in rows]
    axes[2].step(slots, profile_index, where="post", color="black", linewidth=1.0)
    axes[2].set_yticks(range(5), ["tr0", "tr1", "tr2", "tr3", "tr4"])
    axes[2].set_ylabel("Recommended profile")
    axes[2].set_xlabel("Global slot (calibration prefix excluded)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    fig.suptitle("Synthetic functional trace: fusion and logged-replay recommendations", fontsize=11)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=600)
    plt.close(fig)
    for extension in ("pdf", "png", "svg"):
        if output.with_suffix(f".{extension}").stat().st_size == 0:
            raise RuntimeError(f"Empty {extension} export")
    print(f"Synthetic trace written to {output}.pdf/.png/.svg")


if __name__ == "__main__":
    main()
