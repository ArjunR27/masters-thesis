"""
Line charts showing the effect of chunk size on answer quality and retrieval.

Usage:
    python ragas_evaluation/analysis/plot_chunk_size_ablation.py

Outputs:
    ragas_evaluation/analysis/outputs/chunk_size_ablation.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = Path(__file__).resolve().parent
UNIFIED = HERE / "outputs" / "unified_table.csv"
OUT = HERE / "outputs" / "chunk_size_ablation.png"

LINES = [
    ("baseline_raw", 0,  "Raw  0% overlap",  "#e34948", "o",  "-"),
    ("baseline_raw", 10, "Raw 10% overlap",  "#e34948", "s", "--"),
    ("baseline_utt", 0,  "Utt  0% overlap",  "#1baf7a", "o",  "-"),
    ("baseline_utt", 10, "Utt 10% overlap",  "#1baf7a", "s", "--"),
]

CHUNK_SIZES = [128, 256, 512]


def get_value(df, strategy_prefix: str, overlap: int, chunk_size: int, col: str):
    ov_suffix = f"_{overlap}ov"
    system = f"{strategy_prefix}_{chunk_size}{ov_suffix}"
    row = df[df["system"] == system]
    if row.empty:
        return None
    return float(row.iloc[0][col])


def draw_panel(ax, df, y_col, y_label, title):
    for prefix, overlap, label, color, marker, ls in LINES:
        ys = [get_value(df, prefix, overlap, cs, y_col) for cs in CHUNK_SIZES]
        valid = [(cs, y) for cs, y in zip(CHUNK_SIZES, ys) if y is not None]
        if not valid:
            continue
        xs, ys_clean = zip(*valid)
        ax.plot(xs, ys_clean, color=color, marker=marker, linestyle=ls,
                linewidth=1.8, markersize=7, label=label, zorder=3)

    ax.set_xticks(CHUNK_SIZES)
    ax.set_xticklabels([str(c) for c in CHUNK_SIZES], fontsize=8)
    ax.set_xlabel("Chunk size (tokens)", fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="semibold", pad=6)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)


def main() -> None:
    df = pd.read_csv(UNIFIED)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Effect of Chunk Size on Performance\n(baseline systems only)",
                 fontsize=13, fontweight="bold", y=1.01)

    draw_panel(axes[0], df, "answer_correctness", "Answer Correctness",
               "Answer Correctness vs Chunk Size")
    draw_panel(axes[1], df, "ndcg@5", "nDCG@5",
               "nDCG@5 vs Chunk Size")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
