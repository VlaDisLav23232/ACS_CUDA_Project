#!/usr/bin/env python3
"""
plot_tensor_vs_kahan.py — Compare Tensor Core (WMMA) vs FP16+Kahan stencil.

Reads the CSV produced by the heat_stencil benchmark and generates grouped
bar charts for:
  1. Execution Time (ms)
  2. Effective Bandwidth (GB/s)

Usage:
    python3 plot_tensor_vs_kahan.py [csv_path] [output_path]

Defaults:
    csv_path    = ../results/benchmarks.csv
    output_path = ../results/tensor_vs_kahan.png
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 11

def main():
    csv_path    = sys.argv[1] if len(sys.argv) > 1 else '../results/benchmarks.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else '../results/tensor_vs_kahan.png'

    # ---- load data -----------------------------------------------------------
    df = pd.read_csv(csv_path)

    # normalise column names (strip whitespace if any)
    df.columns = df.columns.str.strip()

    # filter to 2D and the two variants of interest
    variants = ['cuda_fp16_kahan', 'cuda_fp16_tensor_core']
    mask = df['variant'].isin(variants)
    if 'dim' in df.columns:
        mask &= (df['dim'] == 2)
    df = df[mask].copy()

    if df.empty:
        print(f"No data found for variants {variants} in {csv_path}")
        print("Available variants:", df['variant'].unique() if not df.empty else "(empty)")
        sys.exit(1)

    # keep only the latest run per (variant, grid_size, reach) group
    if 'reach' in df.columns:
        group_cols = ['variant', 'grid_size', 'reach']
    else:
        group_cols = ['variant', 'grid_size']
    df = df.sort_values('timestamp').groupby(group_cols).last().reset_index()

    # ---- pivot by grid size --------------------------------------------------
    kahan_data  = df[df['variant'] == 'cuda_fp16_kahan'].set_index('grid_size')
    tensor_data = df[df['variant'] == 'cuda_fp16_tensor_core'].set_index('grid_size')

    # align to common grid sizes
    common = sorted(set(kahan_data.index) & set(tensor_data.index))
    if not common:
        print("No common grid sizes found between the two variants.")
        print(f"  Kahan sizes:  {sorted(kahan_data.index.tolist())}")
        print(f"  Tensor sizes: {sorted(tensor_data.index.tolist())}")
        sys.exit(1)

    kahan_time  = [kahan_data.loc[s, 'elapsed_ms']          for s in common]
    tensor_time = [tensor_data.loc[s, 'elapsed_ms']         for s in common]
    kahan_bw    = [kahan_data.loc[s, 'bandwidth_gbs']        for s in common]
    tensor_bw   = [tensor_data.loc[s, 'bandwidth_gbs']       for s in common]
    kahan_mpts  = [kahan_data.loc[s, 'megapoints_per_sec']   for s in common]
    tensor_mpts = [tensor_data.loc[s, 'megapoints_per_sec']  for s in common]

    # ---- colours -------------------------------------------------------------
    c_kahan  = '#4CAF50'
    c_tensor = '#FF6F00'

    x = np.arange(len(common))
    width = 0.35

    # ---- helper: annotate bars -----------------------------------------------
    def annotate_bars(ax, bars, fmt='.1f'):
        for bar in bars:
            ax.annotate(f'{bar.get_height():{fmt}}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=8)

    # ---- figure: three subplots ----------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # -- subplot 1: execution time --
    bars1 = ax1.bar(x - width/2, kahan_time,  width, label='FP16 + Kahan',      color=c_kahan,  edgecolor='white')
    bars2 = ax1.bar(x + width/2, tensor_time, width, label='FP16 Tensor Core',  color=c_tensor, edgecolor='white')
    ax1.set_xlabel('Grid Size N (N×N)')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Execution Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in common])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    annotate_bars(ax1, bars1)
    annotate_bars(ax1, bars2)

    # -- subplot 2: effective bandwidth --
    bars3 = ax2.bar(x - width/2, kahan_bw,  width, label='FP16 + Kahan',      color=c_kahan,  edgecolor='white')
    bars4 = ax2.bar(x + width/2, tensor_bw, width, label='FP16 Tensor Core',  color=c_tensor, edgecolor='white')
    ax2.set_xlabel('Grid Size N (N×N)')
    ax2.set_ylabel('Effective Bandwidth (GB/s)')
    ax2.set_title('Effective Bandwidth')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in common])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    annotate_bars(ax2, bars3)
    annotate_bars(ax2, bars4)

    # -- subplot 3: megapoints per second --
    bars5 = ax3.bar(x - width/2, kahan_mpts,  width, label='FP16 + Kahan',      color=c_kahan,  edgecolor='white')
    bars6 = ax3.bar(x + width/2, tensor_mpts, width, label='FP16 Tensor Core',  color=c_tensor, edgecolor='white')
    ax3.set_xlabel('Grid Size N (N×N)')
    ax3.set_ylabel('Throughput (MPoints/s)')
    ax3.set_title('Throughput')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(s) for s in common])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    annotate_bars(ax3, bars5)
    annotate_bars(ax3, bars6)

    # determine reach from data for the title
    reach_str = ''
    if 'reach' in df.columns:
        reaches = df['reach'].unique()
        reach_str = f', Reach={",".join(str(int(r)) for r in sorted(reaches))}'

    fig.suptitle(f'Tensor Core WMMA vs FP16+Kahan — 2D Heat Stencil{reach_str}',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    print(f'Saved {output_path}')

if __name__ == '__main__':
    main()

