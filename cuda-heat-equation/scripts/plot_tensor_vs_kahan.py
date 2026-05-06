#!/usr/bin/env python3
"""
plot_tensor_vs_kahan.py — Compare all 2D stencil variants.

Reads the CSV produced by the heat_stencil benchmark and generates:
  Figure 1 (tensor_vs_kahan.png):  Performance — 3 subplots
    1. Execution Time (ms)
    2. Effective Bandwidth (GB/s)
    3. Throughput (MPoints/s)

  Figure 2 (accuracy_comparison.png): Accuracy — 2 subplots
    1. Max Absolute Error (log scale)
    2. Relative L2 Error (log scale)
    comparing ALL GPU variants (fp32, fp16_naive, fp16_kahan, fp16_tensor_core)

Usage:
    python3 plot_tensor_vs_kahan.py [csv_path]

Defaults:
    csv_path = ../results/benchmarks.csv
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 11


def annotate_bars(ax, bars, fmt='.1f'):
    """Place value labels on top of bar chart bars."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f'{h:{fmt}}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=7)


def plot_performance(df, common, kahan_data, tensor_data, reach_str, output_dir):
    """Generate the 3-panel performance comparison (Kahan vs Tensor Core)."""
    kahan_time  = [kahan_data.loc[s, 'elapsed_ms']          for s in common]
    tensor_time = [tensor_data.loc[s, 'elapsed_ms']         for s in common]
    kahan_bw    = [kahan_data.loc[s, 'bandwidth_gbs']        for s in common]
    tensor_bw   = [tensor_data.loc[s, 'bandwidth_gbs']       for s in common]
    kahan_mpts  = [kahan_data.loc[s, 'megapoints_per_sec']   for s in common]
    tensor_mpts = [tensor_data.loc[s, 'megapoints_per_sec']  for s in common]

    c_kahan  = '#4CAF50'
    c_tensor = '#FF6F00'
    x = np.arange(len(common))
    width = 0.35

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # -- execution time --
    b1 = ax1.bar(x - width/2, kahan_time,  width, label='FP16 + Kahan',     color=c_kahan,  edgecolor='white')
    b2 = ax1.bar(x + width/2, tensor_time, width, label='FP16 Tensor Core', color=c_tensor, edgecolor='white')
    ax1.set_xlabel('Grid Size N (N×N)')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Execution Time')
    ax1.set_xticks(x); ax1.set_xticklabels([str(s) for s in common])
    ax1.legend(); ax1.grid(axis='y', alpha=0.3)
    annotate_bars(ax1, b1); annotate_bars(ax1, b2)

    # -- bandwidth --
    b3 = ax2.bar(x - width/2, kahan_bw,  width, label='FP16 + Kahan',     color=c_kahan,  edgecolor='white')
    b4 = ax2.bar(x + width/2, tensor_bw, width, label='FP16 Tensor Core', color=c_tensor, edgecolor='white')
    ax2.set_xlabel('Grid Size N (N×N)')
    ax2.set_ylabel('Effective Bandwidth (GB/s)')
    ax2.set_title('Effective Bandwidth')
    ax2.set_xticks(x); ax2.set_xticklabels([str(s) for s in common])
    ax2.legend(); ax2.grid(axis='y', alpha=0.3)
    annotate_bars(ax2, b3); annotate_bars(ax2, b4)

    # -- throughput --
    b5 = ax3.bar(x - width/2, kahan_mpts,  width, label='FP16 + Kahan',     color=c_kahan,  edgecolor='white')
    b6 = ax3.bar(x + width/2, tensor_mpts, width, label='FP16 Tensor Core', color=c_tensor, edgecolor='white')
    ax3.set_xlabel('Grid Size N (N×N)')
    ax3.set_ylabel('Throughput (MPoints/s)')
    ax3.set_title('Throughput')
    ax3.set_xticks(x); ax3.set_xticklabels([str(s) for s in common])
    ax3.legend(); ax3.grid(axis='y', alpha=0.3)
    annotate_bars(ax3, b5); annotate_bars(ax3, b6)

    fig.suptitle(f'Tensor Core WMMA vs FP16+Kahan — 2D Heat Stencil{reach_str}',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, 'tensor_vs_kahan.png')
    fig.savefig(path, bbox_inches='tight')
    print(f'Saved {path}')


def plot_accuracy(df_all, output_dir, reach_str):
    """Generate accuracy comparison across ALL GPU variants."""

    # variants we want to compare (order matters for legend)
    variant_meta = {
        'cuda_fp32':               ('FP32',              '#2196F3', 'o'),
        'cuda_fp16_naive':         ('FP16 Naive',        '#FF9800', '^'),
        'cuda_fp16_kahan':         ('FP16 + Kahan',      '#4CAF50', 'D'),
        'cuda_fp16_tensor_core':   ('FP16 Tensor Core',  '#FF6F00', 's'),
    }

    # filter to 2D GPU variants
    mask = df_all['variant'].isin(variant_meta.keys())
    if 'dim' in df_all.columns:
        mask &= (df_all['dim'] == 2)
    df = df_all[mask].copy()

    if df.empty:
        print("No GPU variant data found for accuracy plot, skipping.")
        return

    # keep latest run per (variant, grid_size)
    if 'reach' in df.columns:
        group_cols = ['variant', 'grid_size', 'reach']
    else:
        group_cols = ['variant', 'grid_size']
    df = df.sort_values('timestamp').groupby(group_cols).last().reset_index()

    sizes = sorted(df['grid_size'].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for variant, (label, color, marker) in variant_meta.items():
        vdf = df[df['variant'] == variant].set_index('grid_size')
        common_sizes = sorted(set(sizes) & set(vdf.index))
        if not common_sizes:
            continue

        max_err = [float(vdf.loc[s, 'max_abs_error']) for s in common_sizes]
        l2_err  = [float(vdf.loc[s, 'l2_error'])      for s in common_sizes]

        # replace exact zeros with tiny value for log scale
        max_err = [v if v > 0 else 1e-16 for v in max_err]
        l2_err  = [v if v > 0 else 1e-16 for v in l2_err]

        ax1.semilogy(common_sizes, max_err, color=color, marker=marker,
                     markersize=7, linewidth=2, label=label)
        ax2.semilogy(common_sizes, l2_err, color=color, marker=marker,
                     markersize=7, linewidth=2, label=label)

    ax1.set_xlabel('Grid Size N (N×N)')
    ax1.set_ylabel('Max Absolute Error (log scale)')
    ax1.set_title('Max Absolute Error vs CPU FP64 Reference')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xticks(sizes)

    ax2.set_xlabel('Grid Size N (N×N)')
    ax2.set_ylabel('Relative L2 Error (log scale)')
    ax2.set_title('Relative L2 Error vs CPU FP64 Reference')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xticks(sizes)

    fig.suptitle(f'Accuracy Comparison — 2D Heat Stencil{reach_str}',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, 'accuracy_comparison.png')
    fig.savefig(path, bbox_inches='tight')
    print(f'Saved {path}')


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else '../results/benchmarks.csv'
    output_dir = os.path.dirname(csv_path) or '.'

    # ---- load data -----------------------------------------------------------
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # ---- determine reach for title -------------------------------------------
    reach_str = ''
    if 'reach' in df.columns:
        gpu_df = df[df['variant'].str.startswith('cuda')]
        if not gpu_df.empty:
            reaches = gpu_df['reach'].dropna().unique()
            reach_str = f', Reach={",".join(str(int(r)) for r in sorted(reaches))}'

    # ---- performance plot (Kahan vs Tensor Core) -----------------------------
    perf_variants = ['cuda_fp16_kahan', 'cuda_fp16_tensor_core']
    perf_mask = df['variant'].isin(perf_variants)
    if 'dim' in df.columns:
        perf_mask &= (df['dim'] == 2)
    perf_df = df[perf_mask].copy()

    if not perf_df.empty:
        if 'reach' in perf_df.columns:
            group_cols = ['variant', 'grid_size', 'reach']
        else:
            group_cols = ['variant', 'grid_size']
        perf_df = perf_df.sort_values('timestamp').groupby(group_cols).last().reset_index()

        kahan_data  = perf_df[perf_df['variant'] == 'cuda_fp16_kahan'].set_index('grid_size')
        tensor_data = perf_df[perf_df['variant'] == 'cuda_fp16_tensor_core'].set_index('grid_size')
        common = sorted(set(kahan_data.index) & set(tensor_data.index))

        if common:
            plot_performance(perf_df, common, kahan_data, tensor_data, reach_str, output_dir)
        else:
            print("No common grid sizes for Kahan vs Tensor Core, skipping performance plot.")
    else:
        print("No Kahan/Tensor Core data found, skipping performance plot.")

    # ---- accuracy plot (all GPU variants) ------------------------------------
    plot_accuracy(df, output_dir, reach_str)


if __name__ == '__main__':
    main()
