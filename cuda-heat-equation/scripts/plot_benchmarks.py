#!/usr/bin/env python3
"""Generate benchmark plots for the heat stencil project."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 11

# DATA: reach=1, timesteps=200, 3D
sizes_r1 = [32, 64, 96, 128, 160, 192]

data_r1 = {
    'CPU fp64':          [602.1,  599.2,  587.3,  550.2,  566.4,  567.7],
    'GPU fp32 naive':    [3046.6, 5663.8, 5815.6, 7605.1, 7598.5, 6447.3],
    'GPU fp16 naive':    [4210.5, 6575.6, 7215.7, 8422.1, 9155.7, 8360.4],
    'GPU fp16 Kahan':    [2263.8, 3210.3, 3555.4, 4389.2, 4670.3, 4371.2],
    'GPU fp32 2.5D':     [2854.2, 4957.8, 5048.4, 6616.5, 6518.5, 6649.1],
    'GPU fp16 Kahan 2.5D': [2265.5, 3704.5, 3938.7, 5174.3, 5275.7, 5287.8],
}

errors_r1 = {
    'GPU fp32 naive':    [3.37e-3, 2.86e-6, 7.63e-6, 2.29e-5, 3.05e-5, 3.05e-5],
    'GPU fp16 naive':    [3.38e-3, 2.20e-2, 1.29e-1, 3.86e-1, 4.36e-1, 9.59e-1],
    'GPU fp16 Kahan':    [3.42e-3, 2.86e-6, 7.63e-6, 2.29e-5, 3.05e-5, 3.05e-5],
    'GPU fp16 Kahan 2.5D': [3.42e-3, 2.86e-6, 7.63e-6, 2.29e-5, 3.05e-5, 3.05e-5],
}

# DATA: reach=4, timesteps=100, 3D
sizes_r4 = [32, 64, 96, 128, 160]

data_r4 = {
    'CPU fp64':          [321.4,  196.1,  191.3,  23.1,   145.6],
    'GPU fp32 naive':    [1636.8, 2499.5, 2044.0, 2246.0, 2156.1],
    'GPU fp16 Kahan':    [1079.0, 1281.8, 1183.7, 1358.3, 1442.4],
    'GPU fp32 2.5D':     [1746.8, 2987.5, 2484.5, 3252.9, 3210.0],
    'GPU fp16 Kahan 2.5D': [1198.5, 1863.9, 1763.1, 2015.3, 1795.0],
}

# Colors
colors = {
    'CPU fp64': '#888888',
    'GPU fp32 naive': '#2196F3',
    'GPU fp16 naive': '#FF9800',
    'GPU fp16 Kahan': '#4CAF50',
    'GPU fp32 2.5D': '#9C27B0',
    'GPU fp16 Kahan 2.5D': '#E91E63',
}

styles = {
    'CPU fp64': '--',
    'GPU fp32 naive': '-',
    'GPU fp16 naive': ':',
    'GPU fp16 Kahan': '-',
    'GPU fp32 2.5D': '-',
    'GPU fp16 Kahan 2.5D': '-',
}

markers = {
    'CPU fp64': 's',
    'GPU fp32 naive': 'o',
    'GPU fp16 naive': '^',
    'GPU fp16 Kahan': 'D',
    'GPU fp32 2.5D': 'v',
    'GPU fp16 Kahan 2.5D': 'P',
}

# FIGURE 1: Throughput vs Grid Size (reach=1)
fig, ax = plt.subplots(figsize=(10, 6))
for name, vals in data_r1.items():
    ax.plot(sizes_r1, vals, color=colors[name], linestyle=styles[name],
            marker=markers[name], markersize=7, linewidth=2, label=name)

ax.set_xlabel('Grid Size N (NxNxN)')
ax.set_ylabel('Throughput (Mpoints/s)')
ax.set_title('3D Heat Stencil Throughput — Reach=1 (7pt stencil), GTX 1650')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xticks(sizes_r1)
fig.tight_layout()
fig.savefig('../results/throughput_r1.png')
print('Saved throughput_r1.png')

# FIGURE 2: Throughput vs Grid Size (reach=4)
fig, ax = plt.subplots(figsize=(10, 6))
for name, vals in data_r4.items():
    ax.plot(sizes_r4, vals, color=colors[name], linestyle=styles[name],
            marker=markers[name], markersize=7, linewidth=2, label=name)

ax.set_xlabel('Grid Size N (NxNxN)')
ax.set_ylabel('Throughput (Mpoints/s)')
ax.set_title('3D Heat Stencil Throughput — Reach=4 (25pt stencil), GTX 1650')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xticks(sizes_r4)
fig.tight_layout()
fig.savefig('../results/throughput_r4.png')
print('Saved throughput_r4.png')

# FIGURE 3: 2.5D Speedup (reach=1 vs reach=4)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Reach=1 speedup
sp_fp32_r1 = [data_r1['GPU fp32 2.5D'][i] / data_r1['GPU fp32 naive'][i] for i in range(len(sizes_r1))]
sp_kahan_r1 = [data_r1['GPU fp16 Kahan 2.5D'][i] / data_r1['GPU fp16 Kahan'][i] for i in range(len(sizes_r1))]

ax1.plot(sizes_r1, sp_fp32_r1, 'o-', color='#2196F3', linewidth=2, label='fp32 2.5D / fp32 naive')
ax1.plot(sizes_r1, sp_kahan_r1, 'D-', color='#E91E63', linewidth=2, label='Kahan 2.5D / Kahan naive')
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Grid Size N')
ax1.set_ylabel('Speedup (2.5D / naive)')
ax1.set_title('Reach=1 (7pt stencil)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(sizes_r1)

# Reach=4 speedup
sp_fp32_r4 = [data_r4['GPU fp32 2.5D'][i] / data_r4['GPU fp32 naive'][i] for i in range(len(sizes_r4))]
sp_kahan_r4 = [data_r4['GPU fp16 Kahan 2.5D'][i] / data_r4['GPU fp16 Kahan'][i] for i in range(len(sizes_r4))]

ax2.plot(sizes_r4, sp_fp32_r4, 'o-', color='#2196F3', linewidth=2, label='fp32 2.5D / fp32 naive')
ax2.plot(sizes_r4, sp_kahan_r4, 'D-', color='#E91E63', linewidth=2, label='Kahan 2.5D / Kahan naive')
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Grid Size N')
ax2.set_ylabel('Speedup (2.5D / naive)')
ax2.set_title('Reach=4 (25pt stencil)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(sizes_r4)

fig.suptitle('2.5D Blocking Speedup over Naive — GTX 1650', fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig('../results/speedup_25d.png', bbox_inches='tight')
print('Saved speedup_25d.png')

# FIGURE 4: Error comparison (reach=1)
fig, ax = plt.subplots(figsize=(10, 6))
for name, vals in errors_r1.items():
    ax.semilogy(sizes_r1, vals, color=colors[name], linestyle=styles[name],
                marker=markers[name], markersize=7, linewidth=2, label=name)

ax.set_xlabel('Grid Size N (NxNxN)')
ax.set_ylabel('Max Absolute Error (log scale)')
ax.set_title('Accuracy Comparison — Reach=1, 3D, GTX 1650')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3, which='both')
ax.set_xticks(sizes_r1)
fig.tight_layout()
fig.savefig('../results/error_comparison.png')
print('Saved error_comparison.png')

print('\nAll plots saved to results/')
