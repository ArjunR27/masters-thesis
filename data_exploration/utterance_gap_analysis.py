import argparse
import csv
import math
import os
import matplotlib.pyplot as plt
import numpy as np
 
def iter_transcript_paths(target):
    target = os.path.abspath(target)
    if os.path.isfile(target):
        yield target
    else:
        for root, _, files in os.walk(target):
            for name in files:
                if name.endswith("_transcripts.csv"):
                    yield os.path.join(root, name)

def collect_gaps(csv_path):
    gaps = []
    prev_end = None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = float(row.get("Start"))
            end = float(row.get("End"))
            if prev_end is not None:
                gaps.append(start-prev_end)
            prev_end = end
    return gaps

def plot_gap_histogram(gaps):
    g = np.array(gaps, dtype=float)
    g = g[g > 0.0]

    plt.figure(figsize=(8, 4))
    plt.hist(g, bins='fd', edgecolor='black', alpha=0.7)
    plt.xlabel("Inter-word gap (seconds)")
    plt.ylabel("Count")
    plt.title("Histogram of Inter-word Gaps")
    plt.tight_layout()
    plt.show()

def plot_gap_histogram_log(gaps):
    g = np.array(gaps, dtype=float)
    g = g[g > 0.0]

    plt.figure(figsize=(10, 5))
    plt.hist(g, bins="fd", edgecolor="black", alpha=0.7)
    plt.xscale("log")
    
    thresholds = [0.5, 0.8, 1.0, 1.5, 2.0]
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for thresh, color in zip(thresholds, colors):
        pct = (g <= thresh).sum() / len(g) * 100
        plt.axvline(thresh, color=color, linestyle='--', linewidth=2, alpha=0.7,
                    label=f'{thresh}s ({pct:.1f}%)')
    
    plt.xlabel("Inter-word gap (seconds, log scale)")
    plt.ylabel("Count")
    plt.title("Histogram of Inter-word Gaps (Log Scale)")
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    all_gaps = []
    seen_files = 0
    for path in iter_transcript_paths('./lpm_data'):
        gaps = collect_gaps(path)
        if gaps:
            all_gaps.extend(gaps)
            seen_files += 1
    
    plot_gap_histogram(all_gaps)
    plot_gap_histogram_log(all_gaps)

if __name__ == "__main__":
    main()