#!/usr/bin/env python3
"""benchmark visualization for 2D/3D heat stencil with variable reach"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

COLORS = {
    "cpu_fp64":           "#333333",
    "cuda_fp32":          "#2196F3",
    "cuda_fp16_naive":    "#FF9800",
    "cuda_fp16_kahan":    "#4CAF50",
    "cpu_fp64_3d":        "#333333",
    "cuda_fp32_3d":       "#2196F3",
    "cuda_fp16_naive_3d": "#FF9800",
    "cuda_fp16_kahan_3d": "#4CAF50",
}

LABELS = {
    "cpu_fp64":           "CPU fp64",
    "cuda_fp32":          "CUDA fp32",
    "cuda_fp16_naive":    "CUDA fp16 naive",
    "cuda_fp16_kahan":    "CUDA fp16+Kahan",
    "cpu_fp64_3d":        "CPU fp64",
    "cuda_fp32_3d":       "CUDA fp32",
    "cuda_fp16_naive_3d": "CUDA fp16 naive",
    "cuda_fp16_kahan_3d": "CUDA fp16+Kahan",
}

PEAK_BW = 192.0


def style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)


def plot_2d(df, outdir):
    d2 = df[df["dim"] == 2].copy()
    gpu_variants = [v for v in d2["variant"].unique() if "cpu" not in v]
    reaches = sorted(d2["reach"].unique())

    # figure 1: accuracy vs grid size (one subplot per reach)
    fig, axes = plt.subplots(1, len(reaches), figsize=(5 * len(reaches), 4.5), sharey=True)
    if len(reaches) == 1:
        axes = [axes]
    for ax, R in zip(axes, reaches):
        for v in d2["variant"].unique():
            if v == "cpu_fp64":
                continue
            sub = d2[(d2["variant"] == v) & (d2["reach"] == R)]
            if sub.empty or sub["max_abs_error"].max() == 0:
                continue
            ax.semilogy(sub["grid_size"], sub["max_abs_error"], "o-",
                        color=COLORS.get(v, "gray"), label=LABELS.get(v, v), markersize=5)
        style_ax(ax, "N", "max |error|" if R == reaches[0] else "", f"2D reach={R}")
    fig.suptitle("2D accuracy vs grid size", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{outdir}/2d_accuracy.png", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/2d_accuracy.png")

    # figure 2: bandwidth vs grid size (one subplot per reach)
    fig, axes = plt.subplots(1, len(reaches), figsize=(5 * len(reaches), 4.5), sharey=True)
    if len(reaches) == 1:
        axes = [axes]
    for ax, R in zip(axes, reaches):
        for v in gpu_variants:
            sub = d2[(d2["variant"] == v) & (d2["reach"] == R)]
            if sub.empty:
                continue
            ax.plot(sub["grid_size"], sub["bandwidth_gbs"], "o-",
                    color=COLORS.get(v, "gray"), label=LABELS.get(v, v), markersize=5)
        ax.axhline(y=PEAK_BW, color="red", linestyle="--", alpha=0.4, label=f"peak {PEAK_BW} GB/s")
        style_ax(ax, "N", "bandwidth (GB/s)" if R == reaches[0] else "", f"2D reach={R}")
    fig.suptitle("2D effective bandwidth vs grid size", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{outdir}/2d_bandwidth.png", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/2d_bandwidth.png")

    # figure 3: speedup vs CPU
    fig, axes = plt.subplots(1, len(reaches), figsize=(5 * len(reaches), 4.5), sharey=True)
    if len(reaches) == 1:
        axes = [axes]
    for ax, R in zip(axes, reaches):
        cpu = d2[(d2["variant"] == "cpu_fp64") & (d2["reach"] == R)].set_index("grid_size")["elapsed_ms"]
        for v in gpu_variants:
            sub = d2[(d2["variant"] == v) & (d2["reach"] == R)].set_index("grid_size")["elapsed_ms"]
            common = cpu.index.intersection(sub.index)
            if len(common) == 0:
                continue
            speedup = cpu.loc[common] / sub.loc[common]
            ax.plot(common, speedup, "o-", color=COLORS.get(v, "gray"),
                    label=LABELS.get(v, v), markersize=5)
        ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
        style_ax(ax, "N", "speedup vs CPU" if R == reaches[0] else "", f"2D reach={R}")
    fig.suptitle("2D GPU speedup over CPU", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{outdir}/2d_speedup.png", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/2d_speedup.png")


def plot_3d(df, outdir):
    d3 = df[df["dim"] == 3].copy()
    gpu_variants = [v for v in d3["variant"].unique() if "cpu" not in v]
    reaches = sorted(d3["reach"].unique())

    # figure 4: 3D accuracy
    fig, axes = plt.subplots(1, len(reaches), figsize=(5 * len(reaches), 4.5), sharey=True)
    if len(reaches) == 1:
        axes = [axes]
    for ax, R in zip(axes, reaches):
        for v in d3["variant"].unique():
            if "cpu" in v:
                continue
            sub = d3[(d3["variant"] == v) & (d3["reach"] == R)]
            if sub.empty or sub["max_abs_error"].max() == 0:
                continue
            ax.semilogy(sub["grid_size"], sub["max_abs_error"], "s-",
                        color=COLORS.get(v, "gray"), label=LABELS.get(v, v), markersize=5)
        style_ax(ax, "N", "max |error|" if R == reaches[0] else "", f"3D reach={R}")
    fig.suptitle("3D accuracy vs grid size", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{outdir}/3d_accuracy.png", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/3d_accuracy.png")

    # figure 5: 3D bandwidth
    fig, axes = plt.subplots(1, len(reaches), figsize=(5 * len(reaches), 4.5), sharey=True)
    if len(reaches) == 1:
        axes = [axes]
    for ax, R in zip(axes, reaches):
        for v in gpu_variants:
            sub = d3[(d3["variant"] == v) & (d3["reach"] == R)]
            if sub.empty:
                continue
            ax.plot(sub["grid_size"], sub["bandwidth_gbs"], "s-",
                    color=COLORS.get(v, "gray"), label=LABELS.get(v, v), markersize=5)
        ax.axhline(y=PEAK_BW, color="red", linestyle="--", alpha=0.4, label=f"peak {PEAK_BW} GB/s")
        style_ax(ax, "N", "bandwidth (GB/s)" if R == reaches[0] else "", f"3D reach={R}")
    fig.suptitle("3D effective bandwidth vs grid size", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{outdir}/3d_bandwidth.png", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/3d_bandwidth.png")

    # figure 6: 3D speedup
    fig, axes = plt.subplots(1, len(reaches), figsize=(5 * len(reaches), 4.5), sharey=True)
    if len(reaches) == 1:
        axes = [axes]
    for ax, R in zip(axes, reaches):
        cpu_v = "cpu_fp64_3d"
        cpu = d3[(d3["variant"] == cpu_v) & (d3["reach"] == R)].set_index("grid_size")["elapsed_ms"]
        for v in gpu_variants:
            sub = d3[(d3["variant"] == v) & (d3["reach"] == R)].set_index("grid_size")["elapsed_ms"]
            common = cpu.index.intersection(sub.index)
            if len(common) == 0:
                continue
            speedup = cpu.loc[common] / sub.loc[common]
            ax.plot(common, speedup, "s-", color=COLORS.get(v, "gray"),
                    label=LABELS.get(v, v), markersize=5)
        ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
        style_ax(ax, "N", "speedup vs CPU" if R == reaches[0] else "", f"3D reach={R}")
    fig.suptitle("3D GPU speedup over CPU", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{outdir}/3d_speedup.png", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/3d_speedup.png")


def plot_combined_summary(df, outdir):
    """single summary figure: 2x2 grid with the most important comparisons"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # top-left: 2D bandwidth for all reaches, fp32
    ax = axes[0, 0]
    d2 = df[df["dim"] == 2]
    for R in sorted(d2["reach"].unique()):
        sub = d2[(d2["variant"] == "cuda_fp32") & (d2["reach"] == R)]
        ax.plot(sub["grid_size"], sub["bandwidth_gbs"], "o-", label=f"R={R}", markersize=5)
    ax.axhline(y=PEAK_BW, color="red", linestyle="--", alpha=0.4, label=f"peak {PEAK_BW}")
    style_ax(ax, "N", "bandwidth (GB/s)", "2D fp32 bandwidth by reach")

    # top-right: 3D bandwidth for all reaches, fp32
    ax = axes[0, 1]
    d3 = df[df["dim"] == 3]
    for R in sorted(d3["reach"].unique()):
        sub = d3[(d3["variant"] == "cuda_fp32_3d") & (d3["reach"] == R)]
        ax.plot(sub["grid_size"], sub["bandwidth_gbs"], "s-", label=f"R={R}", markersize=5)
    ax.axhline(y=PEAK_BW, color="red", linestyle="--", alpha=0.4, label=f"peak {PEAK_BW}")
    style_ax(ax, "N", "bandwidth (GB/s)", "3D fp32 bandwidth by reach")

    # bottom-left: Kahan vs naive accuracy improvement (2D, largest grid per reach)
    ax = axes[1, 0]
    reaches = sorted(d2["reach"].unique())
    x = np.arange(len(reaches))
    width = 0.35
    naive_err = []
    kahan_err = []
    for R in reaches:
        sub_n = d2[(d2["variant"] == "cuda_fp16_naive") & (d2["reach"] == R)]
        sub_k = d2[(d2["variant"] == "cuda_fp16_kahan") & (d2["reach"] == R)]
        naive_err.append(sub_n["max_abs_error"].iloc[-1] if len(sub_n) > 0 else 0)
        kahan_err.append(sub_k["max_abs_error"].iloc[-1] if len(sub_k) > 0 else 0)
    bars1 = ax.bar(x - width/2, naive_err, width, label="fp16 naive", color=COLORS["cuda_fp16_naive"])
    bars2 = ax.bar(x + width/2, kahan_err, width, label="fp16+Kahan", color=COLORS["cuda_fp16_kahan"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"R={R}" for R in reaches])
    style_ax(ax, "stencil reach", "max |error| (N=512)", "2D Kahan vs naive accuracy")

    # bottom-right: GPU speedup over CPU, 3D N=128 across reaches
    ax = axes[1, 1]
    variants_3d = ["cuda_fp32_3d", "cuda_fp16_naive_3d", "cuda_fp16_kahan_3d"]
    x = np.arange(len(reaches))
    width = 0.25
    for i, v in enumerate(variants_3d):
        speedups = []
        for R in reaches:
            cpu_t = d3[(d3["variant"] == "cpu_fp64_3d") & (d3["reach"] == R) & (d3["grid_size"] == 128)]
            gpu_t = d3[(d3["variant"] == v) & (d3["reach"] == R) & (d3["grid_size"] == 128)]
            if len(cpu_t) > 0 and len(gpu_t) > 0:
                speedups.append(cpu_t["elapsed_ms"].iloc[0] / gpu_t["elapsed_ms"].iloc[0])
            else:
                speedups.append(0)
        ax.bar(x + (i - 1) * width, speedups, width, label=LABELS.get(v, v), color=COLORS.get(v, "gray"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"R={R}" for R in reaches])
    style_ax(ax, "stencil reach", "speedup vs CPU", "3D GPU speedup (N=128)")

    fig.suptitle("Heat stencil benchmark summary (GTX 1650 Max-Q)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{outdir}/summary.png", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/summary.png")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmarks.csv"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "results"

    df = pd.read_csv(csv_path)
    print(f"loaded {len(df)} rows from {csv_path}")
    print(f"dims: {sorted(df['dim'].unique())}, reaches: {sorted(df['reach'].unique())}")

    plt.style.use("seaborn-v0_8-whitegrid")

    print("generating 2D plots...")
    plot_2d(df, outdir)
    print("generating 3D plots...")
    plot_3d(df, outdir)
    print("generating summary...")
    plot_combined_summary(df, outdir)
    print("done")


if __name__ == "__main__":
    main()
