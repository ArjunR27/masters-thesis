"""
Box plots of per-question answer_correctness for all 14 systems.

Usage:
    python ragas_evaluation/analysis/plot_score_distributions.py

Outputs:
    ragas_evaluation/analysis/outputs/score_distributions.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
PER_Q = HERE.parent.parent / "ragas_evaluation/outputs/per_question_all_systems.csv"
OUT = HERE / "outputs" / "score_distributions.png"


def short_name(s: str) -> str:
    return (
        s.replace("treeseg_leaf", "T-Leaf")
         .replace("treeseg_summary_tree", "T-SumTree")
         .replace("baseline_raw_", "Raw-")
         .replace("baseline_utt_", "Utt-")
         .replace("_0ov", "-0ov")
         .replace("_10ov", "-10ov")
    )


def system_color(s: str) -> str:
    if "treeseg" in s:
        return "#2a78d6"
    if "raw" in s:
        return "#e34948"
    return "#1baf7a"


def main() -> None:
    df = pd.read_csv(PER_Q)

    # Sort systems by median answer_correctness descending
    medians = df.groupby("system")["answer_correctness"].median().sort_values(ascending=True)
    systems_ordered = medians.index.tolist()
    labels = [short_name(s) for s in systems_ordered]
    colors = [system_color(s) for s in systems_ordered]

    data = [df[df["system"] == s]["answer_correctness"].dropna().values for s in systems_ordered]

    fig, ax = plt.subplots(figsize=(8, 9))
    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        notch=False,
        widths=0.55,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker="o", markersize=3, alpha=0.4, linestyle="none"),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for flier, color in zip(bp["fliers"], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Answer Correctness (per question)", fontsize=9)
    ax.set_title("Answer Correctness Distribution by System\n(n=150 questions each)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.05, 1.05)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#2a78d6", alpha=0.8, label="TreeSeg"),
        Patch(color="#e34948", alpha=0.8, label="Raw baseline"),
        Patch(color="#1baf7a", alpha=0.8, label="Utt baseline"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
