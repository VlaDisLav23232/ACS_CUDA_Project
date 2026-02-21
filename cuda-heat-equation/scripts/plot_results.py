#!/usr/bin/env python3
"""quick plots from benchmark csv"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # max absolute error vs grid size
    ax = axes[0, 0]
    for v in df["variant"].unique():
        sub = df[df["variant"] == v]
        grouped = sub.groupby("grid_size")["max_abs_error"].mean()
        ax.semilogy(grouped.index, grouped.values, "o-", label=v)
    ax.set_xlabel("grid size N")
    ax.set_ylabel("max absolute error")
    ax.set_title("accuracy vs grid size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # L2 error vs grid size
    ax = axes[0, 1]
    for v in df["variant"].unique():
        sub = df[df["variant"] == v]
        grouped = sub.groupby("grid_size")["l2_error"].mean()
        ax.semilogy(grouped.index, grouped.values, "o-", label=v)
    ax.set_xlabel("grid size N")
    ax.set_ylabel("relative L2 error")
    ax.set_title("L2 error vs grid size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # runtime vs grid size (for fixed timesteps)
    ax = axes[1, 0]
    t_max = df["timesteps"].max()
    sub_t = df[df["timesteps"] == t_max]
    for v in sub_t["variant"].unique():
        sub = sub_t[sub_t["variant"] == v]
        grouped = sub.groupby("grid_size")["elapsed_ms"].mean()
        ax.plot(grouped.index, grouped.values, "o-", label=v)
    ax.set_xlabel("grid size N")
    ax.set_ylabel("elapsed (ms)")
    ax.set_title(f"runtime vs grid size (T={t_max})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # bandwidth vs grid size
    ax = axes[1, 1]
    gpu_variants = [v for v in df["variant"].unique() if v != "cpu_fp64"]
    for v in gpu_variants:
        sub = df[(df["variant"] == v) & (df["timesteps"] == t_max)]
        grouped = sub.groupby("grid_size")["bandwidth_gbs"].mean()
        ax.plot(grouped.index, grouped.values, "o-", label=v)
    ax.axhline(y=192, color="red", linestyle="--", alpha=0.5, label="theoretical peak")
    ax.set_xlabel("grid size N")
    ax.set_ylabel("effective bandwidth (GB/s)")
    ax.set_title("memory bandwidth utilization")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = csv_path.replace(".csv", ".png")
    plt.savefig(out, dpi=150)
    print(f"saved plot to {out}")
    plt.show()

if __name__ == "__main__":
    main()
