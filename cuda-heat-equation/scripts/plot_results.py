#!/usr/bin/env python3
"""benchmark visualization for 2D/3D heat stencil with variable reach"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from adjustText import adjust_text

COLORS = {
    "cpu_fp64":           "#333333",
    "cuda_fp32":          "#2F6DB3",
    "cuda_fp16_naive":    "#D97706",
    "cuda_fp16_kahan":    "#F59E0B",
    "cuda_fp16_kahan_tiled": "#A16207",
    "cuda_fp16_neumaier": "#B45309",
    "cuda_fp16_twosum":   "#92400E",
    "cuda_cfp16_naive":   "#0F766E",
    "cuda_cfp16_kahan":   "#10B981",
    "cuda_cfp16_kahan_tiled": "#7A7A7A",
    "cpu_fp64_3d":        "#333333",
    "cuda_fp32_3d":       "#2F6DB3",
    "cuda_fp16_naive_3d": "#D97706",
    "cuda_fp16_kahan_3d": "#F59E0B",
}

LABELS = {
    "cpu_fp64":           "CPU fp64",
    "cuda_fp32":          "CUDA fp32",
    "cuda_fp16_naive":    "CUDA fp16 naive",
    "cuda_fp16_kahan":    "CUDA fp16+Kahan",
    "cuda_fp16_kahan_tiled": "CUDA fp16+Kahan tiled",
    "cuda_fp16_neumaier": "CUDA fp16+Neumaier",
    "cuda_fp16_twosum":   "CUDA fp16+TwoSum",
    "cuda_cfp16_naive":   "CUDA cfp16 naive",
    "cuda_cfp16_kahan":   "CUDA cfp16+Kahan",
    "cuda_cfp16_kahan_tiled": "CUDA cfp16+Kahan tiled",
    "cpu_fp64_3d":        "CPU fp64",
    "cuda_fp32_3d":       "CUDA fp32",
    "cuda_fp16_naive_3d": "CUDA fp16 naive",
    "cuda_fp16_kahan_3d": "CUDA fp16+Kahan",
}

PEAK_BW = 192.0

VARIANT_ORDER = [
    "cpu_fp64",
    "cuda_fp32",
    "cuda_fp16_naive",
    "cuda_fp16_kahan",
    "cuda_fp16_kahan_tiled",
    "cuda_fp16_neumaier",
    "cuda_fp16_twosum",
    "cuda_cfp16_naive",
    "cuda_cfp16_kahan",
    "cuda_cfp16_kahan_tiled",
    "cpu_fp64_3d",
    "cuda_fp32_3d",
    "cuda_fp16_naive_3d",
    "cuda_fp16_kahan_3d",
]

MARKERS = {
    "cpu_fp64": "X",
    "cuda_fp32": "o",
    "cuda_fp16_naive": "s",
    "cuda_fp16_kahan": "^",
    "cuda_fp16_kahan_tiled": "h",
    "cuda_fp16_neumaier": "h",
    "cuda_fp16_twosum":   "p",
    "cuda_cfp16_naive": "P",
    "cuda_cfp16_kahan": "v",
    "cuda_cfp16_kahan_tiled": "o",
    "cpu_fp64_3d": "X",
    "cuda_fp32_3d": "o",
    "cuda_fp16_naive_3d": "s",
    "cuda_fp16_kahan_3d": "^",
}

LINESTYLES = {
    "cpu_fp64": "-",
    "cuda_fp32": "-",
    "cuda_fp16_naive": "-",
    "cuda_fp16_kahan": "--",
    "cuda_fp16_kahan_tiled": "-.",
    "cuda_fp16_neumaier": "-.",
    "cuda_fp16_twosum":   (0, (3, 1, 1, 1)),
    "cuda_cfp16_naive": "-",
    "cuda_cfp16_kahan": "--",
    "cuda_cfp16_kahan_tiled": "-.",
    "cpu_fp64_3d": "-",
    "cuda_fp32_3d": "-",
    "cuda_fp16_naive_3d": "--",
    "cuda_fp16_kahan_3d": "-.",
}

METHOD_LABELS = {
    "cuda_fp32": "fp32",
    "cuda_fp16_naive": "naive",
    "cuda_fp16_kahan": "Kahan",
    "cuda_fp16_kahan_tiled": "Kahan (tiled)",
    "cuda_fp16_neumaier": "Neumaier",
    "cuda_fp16_twosum": "TwoSum",
    "cuda_cfp16_naive": "naive",
    "cuda_cfp16_kahan": "Kahan",
    "cuda_cfp16_kahan_tiled": "Kahan (tiled)",
    "cuda_fp32_3d": "fp32",
    "cuda_fp16_naive_3d": "naive",
    "cuda_fp16_kahan_3d": "Kahan",
}

TRADEOFF_LABELS = {
    "cuda_fp32": "fp32",
    "cuda_fp32_3d": "fp32",
    "cuda_fp16_naive": "fp16 naive",
    "cuda_fp16_kahan": "fp16 Kahan",
    "cuda_fp16_kahan_tiled": "fp16 Kahan-tiled",
    "cuda_fp16_neumaier": "fp16 Neumaier",
    "cuda_fp16_twosum": "fp16 TwoSum",
    "cuda_fp16_naive_3d": "fp16 naive",
    "cuda_fp16_kahan_3d": "fp16 Kahan",
    "cuda_cfp16_naive": "cfp16 naive",
    "cuda_cfp16_kahan": "cfp16 Kahan",
    "cuda_cfp16_kahan_tiled": "cfp16 Kahan-tiled",
}

TRADEOFF_LABEL_OFFSETS = {
    "cuda_fp32": (7, 4),
    "cuda_fp32_3d": (7, 4),
    "cuda_fp16_naive": (8, 2),
    "cuda_fp16_naive_3d": (8, 2),
    "cuda_fp16_kahan": (8, 6),
    "cuda_fp16_kahan_tiled": (8, 6),
    "cuda_fp16_kahan_3d": (8, 6),
    "cuda_fp16_neumaier": (8, -8),
    "cuda_fp16_twosum": (8, -2),
    "cuda_cfp16_naive": (8, -10),
    "cuda_cfp16_kahan": (8, 4),
    "cuda_cfp16_kahan_tiled": (8, 6),
}

PLOT_EXCLUDED_VARIANTS = {
    "cuda_fp16_kahan_reg",
    "cuda_cfp16_kahan_reg",
}

ACCURACY_FILTERED_VARIANTS_2D = {
    "cuda_fp32",
    "cuda_fp16_naive",
    "cuda_cfp16_naive",
    "cuda_fp16_kahan",
    "cuda_fp16_neumaier",
    "cuda_fp16_twosum",
    "cuda_cfp16_kahan",
}

TRADEOFF_INCLUDED_VARIANTS = {
    "cuda_fp32",
    "cuda_cfp16_kahan_tiled",
    "cuda_fp16_neumaier",
    "cuda_fp16_twosum",
}

BANDWIDTH_FAMILIES = [
    ("fp32", ["cuda_fp32", "cuda_fp32_3d"]),
    ("fp16", ["cuda_fp16_naive", "cuda_fp16_kahan", "cuda_fp16_kahan_tiled",
              "cuda_fp16_neumaier", "cuda_fp16_twosum",
              "cuda_fp16_naive_3d", "cuda_fp16_kahan_3d"]),
    ("cfp16", ["cuda_cfp16_naive", "cuda_cfp16_kahan", "cuda_cfp16_kahan_tiled"]),
]

def style_ax(ax, xlabel, ylabel, title, show_legend=True):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if show_legend:
        ax.legend(fontsize=8, loc="best", frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.25)


def add_figure_hint(fig, hint_text, y=0.94):
    fig.text(0.5, y, hint_text, ha="center", va="top", fontsize=9, color="#555555")


def prepare_plot_df(df):
    key_cols = ["variant", "dim", "reach", "grid_size"]
    work = df[~df["variant"].isin(PLOT_EXCLUDED_VARIANTS)].copy()
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
        work = work.sort_values(key_cols + ["timestamp", "timesteps"])
    else:
        work = work.sort_values(key_cols + ["timesteps"])
    return work.groupby(key_cols, as_index=False).tail(1).copy()


def variant_sort_key(variant):
    if variant in VARIANT_ORDER:
        return (0, VARIANT_ORDER.index(variant))
    return (1, variant)


def plot_variant(ax, x, y, variant, label=None, color=None, marker=None, linestyle=None):
    ax.plot(
        x, y,
        color=color or COLORS.get(variant, "gray"),
        label=label or LABELS.get(variant, variant),
        marker=MARKERS.get(variant, "o") if marker is None else marker,
        linestyle=linestyle or LINESTYLES.get(variant, "-"),
        markersize=4,
        linewidth=1.8,
        alpha=0.9,
        markerfacecolor="white",
        markeredgewidth=1.2,
    )


def plot_accuracy_grid(df_dim, outdir, dim_label, filename, variants=None, title_suffix=""):
    reaches = sorted(df_dim["reach"].unique())
    fig, axes = plt.subplots(1, len(reaches), figsize=(5 * len(reaches), 4.5), sharey=True)
    if len(reaches) == 1:
        axes = [axes]

    allowed_variants = None if variants is None else set(variants)
    plot_variants = sorted(df_dim["variant"].unique(), key=variant_sort_key)

    for ax, reach in zip(axes, reaches):
        for variant in plot_variants:
            if "cpu" in variant:
                continue
            if allowed_variants is not None and variant not in allowed_variants:
                continue
            sub = df_dim[(df_dim["variant"] == variant) & (df_dim["reach"] == reach)]
            if sub.empty or sub["max_abs_error"].max() == 0:
                continue
            sub = sub.sort_values("grid_size")
            plot_variant(ax, sub["grid_size"], sub["max_abs_error"], variant)

        ax.set_xscale("log")
        ax.set_yscale("log")
        style_ax(ax, "N", "max |error|" if reach == reaches[0] else "", f"{dim_label} reach={reach}")

    title = f"{dim_label} accuracy vs grid size"
    if title_suffix:
        title = f"{title} {title_suffix}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    add_figure_hint(fig, "Lower is better")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{outdir}/{filename}", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/{filename}")


def family_variants_for_dim(df_dim, family_name):
    family_variants = []
    family_set = None
    for name, variants in BANDWIDTH_FAMILIES:
        if name == family_name:
            family_set = set(variants)
            break
    if family_set is None:
        return family_variants
    for variant in sorted(df_dim["variant"].unique(), key=variant_sort_key):
        if variant in family_set:
            family_variants.append(variant)
    return family_variants


def plot_bandwidth_family_grid(df_dim, outdir, dim_label, filename):
    reaches = sorted(df_dim["reach"].unique())
    fig, axes = plt.subplots(
        len(BANDWIDTH_FAMILIES),
        len(reaches),
        figsize=(4.2 * len(reaches), 3.2 * len(BANDWIDTH_FAMILIES)),
        sharex=False,
        sharey=True,
    )
    axes = np.array(axes, dtype=object).reshape(len(BANDWIDTH_FAMILIES), len(reaches))

    peak_color = "#C97E7E"
    peak_alpha = 0.35

    for row_idx, (family_name, _) in enumerate(BANDWIDTH_FAMILIES):
        family_variants = family_variants_for_dim(df_dim, family_name)
        for col_idx, reach in enumerate(reaches):
            ax = axes[row_idx, col_idx]
            plotted = False
            for variant in family_variants:
                sub = df_dim[(df_dim["variant"] == variant) & (df_dim["reach"] == reach)]
                if sub.empty:
                    continue
                sub = sub.sort_values("grid_size")
                plot_variant(
                    ax,
                    sub["grid_size"],
                    sub["bandwidth_gbs"],
                    variant,
                    label=METHOD_LABELS.get(variant, LABELS.get(variant, variant)),
                    marker="",
                )
                plotted = True

            ax.axhline(y=PEAK_BW, color=peak_color, linestyle="--", linewidth=0.9, alpha=peak_alpha, zorder=0)
            if plotted:
                style_ax(
                    ax,
                    "N" if row_idx == len(BANDWIDTH_FAMILIES) - 1 else "",
                    "bandwidth (GB/s)" if col_idx == 0 else "",
                    f"{family_name}, reach={reach}",
                )
            else:
                ax.set_title(f"{family_name}, reach={reach}", fontsize=11, fontweight="bold")
                ax.grid(True, alpha=0.25)
                ax.set_ylabel("bandwidth (GB/s)" if col_idx == 0 else "", fontsize=10)
                ax.set_xlabel("N" if row_idx == len(BANDWIDTH_FAMILIES) - 1 else "", fontsize=10)
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10, color="#666666",
                        transform=ax.transAxes)

    fig.suptitle(f"{dim_label} effective bandwidth by family and reach", fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.985, 0.985, f"peak bandwidth = {PEAK_BW} GB/s", ha="right", va="top",
             fontsize=9, color=peak_color)
    add_figure_hint(fig, "Higher is better")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{outdir}/{filename}", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/{filename}")


def pick_tradeoff_grid_size(df_dim, preferred=512):
    grid_sizes = sorted(df_dim["grid_size"].unique())
    if preferred in grid_sizes:
        return preferred
    return grid_sizes[-1]


def compute_mp_per_sec(row):
    points_per_step = float(row["grid_size"]) ** int(row["dim"])
    total_points = points_per_step * float(row["timesteps"])
    elapsed_sec = float(row["elapsed_ms"]) / 1000.0
    if elapsed_sec <= 0.0:
        return np.nan
    return total_points / elapsed_sec / 1.0e6


def plot_accuracy_bandwidth_tradeoff(df_dim, outdir, dim_label, filename):
    reaches = sorted(df_dim["reach"].unique())
    grid_size = pick_tradeoff_grid_size(df_dim)
    plot_df = df_dim[
        (df_dim["grid_size"] == grid_size)
        & (~df_dim["variant"].str.contains("cpu"))
        & (df_dim["variant"].isin(TRADEOFF_INCLUDED_VARIANTS))
    ].copy()
    if plot_df.empty:
        print(f"  skipped {filename}: no GPU rows found at N={grid_size}")
        return

    fig, axes = plt.subplots(1, len(reaches), figsize=(4.6 * len(reaches), 4.0), sharey=True)
    if len(reaches) == 1:
        axes = [axes]

    for ax, reach in zip(axes, reaches):
        sub = plot_df[plot_df["reach"] == reach].copy()
        if sub.empty:
            ax.set_title(f"{dim_label} reach={reach}", fontsize=11, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.grid(True, alpha=0.25)
            continue

        sub = sub.sort_values("max_abs_error")
        texts = []
        for _, row in sub.iterrows():
            variant = row["variant"]
            x = row["max_abs_error"]
            y = row["bandwidth_gbs"] / PEAK_BW
            ax.scatter(
                x,
                y,
                s=70,
                color=COLORS.get(variant, "gray"),
                marker=MARKERS.get(variant, "o"),
                alpha=0.95,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            dx, dy = TRADEOFF_LABEL_OFFSETS.get(variant, (6, 4))
            text = ax.text(
                x * (1.0 + dx * 0.002),
                y + dy * 0.002,
                TRADEOFF_LABELS.get(variant, METHOD_LABELS.get(variant, LABELS.get(variant, variant))),
                fontsize=8,
                color=COLORS.get(variant, "gray"),
            )
            texts.append(text)

        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.7),
            expand=(1.2, 1.4),
            force_text=(0.5, 0.8),
            force_static=(0.4, 0.7),
            only_move={"points": "y", "text": "xy"},
        )

        ax.set_xscale("log")
        style_ax(
            ax,
            "max |error|",
            "bandwidth / peak" if reach == reaches[0] else "",
            f"{dim_label} reach={reach}",
            show_legend=False,
        )

    fig.suptitle(
        f"{dim_label} accuracy-performance tradeoff at N={grid_size}",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    add_figure_hint(fig, "Lower error is better; higher bandwidth / peak is better")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(f"{outdir}/{filename}", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/{filename}")


def plot_accuracy_throughput_tradeoff(df_dim, outdir, dim_label, filename):
    reaches = sorted(df_dim["reach"].unique())
    grid_size = pick_tradeoff_grid_size(df_dim)
    plot_df = df_dim[
        (df_dim["grid_size"] == grid_size)
        & (~df_dim["variant"].str.contains("cpu"))
        & (df_dim["variant"].isin(TRADEOFF_INCLUDED_VARIANTS))
    ].copy()
    if plot_df.empty:
        print(f"  skipped {filename}: no GPU rows found at N={grid_size}")
        return

    fig, axes = plt.subplots(1, len(reaches), figsize=(4.6 * len(reaches), 4.0), sharey=True)
    if len(reaches) == 1:
        axes = [axes]

    for ax, reach in zip(axes, reaches):
        sub = plot_df[plot_df["reach"] == reach].copy()
        if sub.empty:
            ax.set_title(f"{dim_label} reach={reach}", fontsize=11, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.grid(True, alpha=0.25)
            continue

        sub = sub.sort_values("max_abs_error")
        texts = []
        for _, row in sub.iterrows():
            variant = row["variant"]
            x = row["max_abs_error"]
            y = compute_mp_per_sec(row)
            ax.scatter(
                x,
                y,
                s=70,
                color=COLORS.get(variant, "gray"),
                marker=MARKERS.get(variant, "o"),
                alpha=0.95,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            dx, dy = TRADEOFF_LABEL_OFFSETS.get(variant, (6, 4))
            text = ax.text(
                x * (1.0 + dx * 0.002),
                y + dy * 0.002,
                TRADEOFF_LABELS.get(variant, METHOD_LABELS.get(variant, LABELS.get(variant, variant))),
                fontsize=8,
                color=COLORS.get(variant, "gray"),
            )
            texts.append(text)

        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.7),
            expand=(1.2, 1.4),
            force_text=(0.5, 0.8),
            force_static=(0.4, 0.7),
            only_move={"points": "y", "text": "xy"},
        )

        ax.set_xscale("log")
        style_ax(
            ax,
            "max |error|",
            "throughput (MP/s)" if reach == reaches[0] else "",
            f"{dim_label} reach={reach}",
            show_legend=False,
        )

    fig.suptitle(
        f"{dim_label} accuracy-throughput tradeoff at N={grid_size}",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    add_figure_hint(fig, "Lower error is better; higher MP/s is better")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(f"{outdir}/{filename}", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/{filename}")


def ratio_label(numerator_variant, denominator_variant):
    numerator = METHOD_LABELS.get(numerator_variant, LABELS.get(numerator_variant, numerator_variant))
    denominator = METHOD_LABELS.get(denominator_variant, LABELS.get(denominator_variant, denominator_variant))
    return f"{numerator} / {denominator}"


def build_ratio_series(df_dim, reach, base_variant, compare_variant, value_col, invert=False):
    base = df_dim[(df_dim["variant"] == base_variant) & (df_dim["reach"] == reach)].set_index("grid_size")[value_col]
    comp = df_dim[(df_dim["variant"] == compare_variant) & (df_dim["reach"] == reach)].set_index("grid_size")[value_col]
    common = base.index.intersection(comp.index)
    if len(common) == 0:
        return None, None
    common = common.sort_values()
    if invert:
        ratio = base.loc[common] / comp.loc[common]
    else:
        ratio = comp.loc[common] / base.loc[common]
    return common, ratio


def family_naive_comparisons(df_dim):
    families = []
    available_variants = set(df_dim["variant"].unique())
    for family_name in ["fp16", "cfp16"]:
        family_variants = family_variants_for_dim(df_dim, family_name)
        base_variant = next((v for v in family_variants if "naive" in v), None)
        if base_variant is None or base_variant not in available_variants:
            continue
        compare_variants = [v for v in family_variants if v != base_variant and v in available_variants]
        if compare_variants:
            families.append((family_name, base_variant, compare_variants))
    return families


def plot_kahan_benefit(df_dim, outdir, dim_label, metric_col, ylabel, filename, invert=False):
    reaches = sorted(df_dim["reach"].unique())
    families = family_naive_comparisons(df_dim)
    if not families:
        print(f"  skipped {filename}: no mitigation/naive comparisons found")
        return

    fig, axes = plt.subplots(
        len(families),
        len(reaches),
        figsize=(4.4 * len(reaches), 3.3 * len(families)),
        sharex=False,
        sharey=True,
    )
    axes = np.array(axes, dtype=object).reshape(len(families), len(reaches))

    for row_idx, (family_name, base_variant, compare_variants) in enumerate(families):
        for col_idx, reach in enumerate(reaches):
            ax = axes[row_idx, col_idx]
            plotted = False
            for compare_variant in compare_variants:
                x, ratio = build_ratio_series(
                    df_dim,
                    reach,
                    base_variant,
                    compare_variant,
                    metric_col,
                    invert=invert,
                )
                if x is None:
                    continue
                plot_variant(
                    ax,
                    x,
                    ratio,
                    compare_variant,
                    label=ratio_label(base_variant if invert else compare_variant,
                                      compare_variant if invert else base_variant),
                    marker=None,
                )
                plotted = True

            ax.axhline(y=1.0, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7, zorder=0)
            if plotted:
                style_ax(
                    ax,
                    "N" if row_idx == len(families) - 1 else "",
                    ylabel if col_idx == 0 else "",
                    f"{family_name}, reach={reach}",
                )
            else:
                ax.set_title(f"{family_name}, reach={reach}", fontsize=11, fontweight="bold")
                ax.grid(True, alpha=0.25)
                ax.set_ylabel(ylabel if col_idx == 0 else "", fontsize=10)
                ax.set_xlabel("N" if row_idx == len(families) - 1 else "", fontsize=10)
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10, color="#666666",
                        transform=ax.transAxes)

    fig.suptitle(f"{dim_label} {ylabel} vs naive baseline", fontsize=13, fontweight="bold", y=0.98)
    add_figure_hint(fig, "Higher is better")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{outdir}/{filename}", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/{filename}")


def plot_error_vs_fp32(df_dim, outdir, dim_label, filename):
    reaches = sorted(df_dim["reach"].unique())
    variants = sorted([v for v in df_dim["variant"].unique() if "cpu" not in v], key=variant_sort_key)
    fp32_variant = "cuda_fp32" if dim_label == "2D" else "cuda_fp32_3d"
    families = []
    for family_name in ["fp16", "cfp16"]:
        compare_variants = [v for v in family_variants_for_dim(df_dim, family_name) if v != fp32_variant]
        if compare_variants:
            families.append((family_name, compare_variants))
    if fp32_variant not in variants or not families:
        print(f"  skipped {filename}: no fp32 comparison set found")
        return

    fig, axes = plt.subplots(
        len(families),
        len(reaches),
        figsize=(4.4 * len(reaches), 3.3 * len(families)),
        sharex=False,
        sharey=True,
    )
    axes = np.array(axes, dtype=object).reshape(len(families), len(reaches))

    for row_idx, (family_name, compare_variants) in enumerate(families):
        for col_idx, reach in enumerate(reaches):
            ax = axes[row_idx, col_idx]
            plotted = False
            for variant in compare_variants:
                x, ratio = build_ratio_series(
                    df_dim,
                    reach,
                    fp32_variant,
                    variant,
                    "max_abs_error",
                    invert=True,
                )
                if x is None:
                    continue
                plot_variant(
                    ax,
                    x,
                    ratio,
                    variant,
                    label=ratio_label(fp32_variant, variant),
                    marker=None,
                )
                plotted = True

            ax.axhline(y=1.0, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7, zorder=0)
            if plotted:
                ax.set_xscale("log")
                ax.set_yscale("log")
                style_ax(
                    ax,
                    "N" if row_idx == len(families) - 1 else "",
                    "error_fp32 / error_variant" if col_idx == 0 else "",
                    f"{family_name}, reach={reach}",
                )
            else:
                ax.set_title(f"{family_name}, reach={reach}", fontsize=11, fontweight="bold")
                ax.grid(True, alpha=0.25)
                ax.set_xlabel("N" if row_idx == len(families) - 1 else "", fontsize=10)
                ax.set_ylabel("error_fp32 / error_variant" if col_idx == 0 else "", fontsize=10)
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10, color="#666666",
                        transform=ax.transAxes)

    fig.suptitle(f"{dim_label} error ratio vs fp32 baseline", fontsize=13, fontweight="bold", y=0.98)
    add_figure_hint(fig, "Higher is better")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{outdir}/{filename}", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/{filename}")


def plot_2d(df, outdir):
    d2 = df[df["dim"] == 2].copy()
    if d2.empty:
        print("  skipped 2D plots: no 2D rows found")
        return
    gpu_variants = sorted([v for v in d2["variant"].unique() if "cpu" not in v], key=variant_sort_key)
    reaches = sorted(d2["reach"].unique())

    # figure 1: accuracy vs grid size (one subplot per reach)
    plot_accuracy_grid(d2, outdir, "2D", "2d_accuracy.png")
    plot_accuracy_grid(
        d2,
        outdir,
        "2D",
        "2d_accuracy_filtered.png",
        variants=ACCURACY_FILTERED_VARIANTS_2D,
        title_suffix="(selected variants)",
    )

    # figure 2: bandwidth by precision family and reach
    plot_bandwidth_family_grid(d2, outdir, "2D", "2d_bandwidth.png")

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
            common = common.sort_values()
            speedup = cpu.loc[common] / sub.loc[common]
            plot_variant(ax, common, speedup, v)
        ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
        style_ax(ax, "N", "speedup vs CPU" if R == reaches[0] else "", f"2D reach={R}")
    fig.suptitle("2D GPU speedup over CPU", fontsize=13, fontweight="bold", y=1.02)
    add_figure_hint(fig, "Higher is better")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{outdir}/2d_speedup.png", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/2d_speedup.png")

    # figure 4: accuracy-performance tradeoff at fixed N
    plot_accuracy_bandwidth_tradeoff(d2, outdir, "2D", "2d_accuracy_bandwidth_tradeoff.png")
    plot_accuracy_throughput_tradeoff(d2, outdir, "2D", "2d_accuracy_throughput_tradeoff.png")

    # figure 5: normalized Kahan benefit
    plot_kahan_benefit(
        d2,
        outdir,
        "2D",
        "max_abs_error",
        "error improvement (error_naive / error_variant)",
        "2d_kahan_error_improvement.png",
        invert=True,
    )
    plot_kahan_benefit(
        d2,
        outdir,
        "2D",
        "bandwidth_gbs",
        "bandwidth ratio (bandwidth_variant / bandwidth_naive)",
        "2d_kahan_bandwidth_ratio.png",
        invert=False,
    )
    plot_error_vs_fp32(d2, outdir, "2D", "2d_error_fp32_over_variant.png")


def plot_3d(df, outdir):
    d3 = df[df["dim"] == 3].copy()
    if d3.empty:
        print("  skipped 3D plots: no 3D rows found")
        return
    gpu_variants = sorted([v for v in d3["variant"].unique() if "cpu" not in v], key=variant_sort_key)
    reaches = sorted(d3["reach"].unique())

    # figure 4: 3D accuracy
    plot_accuracy_grid(d3, outdir, "3D", "3d_accuracy.png")

    # figure 5: bandwidth by precision family and reach
    plot_bandwidth_family_grid(d3, outdir, "3D", "3d_bandwidth.png")

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
            common = common.sort_values()
            speedup = cpu.loc[common] / sub.loc[common]
            plot_variant(ax, common, speedup, v)
        ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
        style_ax(ax, "N", "speedup vs CPU" if R == reaches[0] else "", f"3D reach={R}")
    fig.suptitle("3D GPU speedup over CPU", fontsize=13, fontweight="bold", y=1.02)
    add_figure_hint(fig, "Higher is better")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{outdir}/3d_speedup.png", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/3d_speedup.png")

    # figure 7: accuracy-performance tradeoff at fixed N
    plot_accuracy_bandwidth_tradeoff(d3, outdir, "3D", "3d_accuracy_bandwidth_tradeoff.png")
    plot_accuracy_throughput_tradeoff(d3, outdir, "3D", "3d_accuracy_throughput_tradeoff.png")

    # figure 8: normalized Kahan benefit
    plot_kahan_benefit(
        d3,
        outdir,
        "3D",
        "max_abs_error",
        "error improvement (error_naive / error_variant)",
        "3d_kahan_error_improvement.png",
        invert=True,
    )
    plot_kahan_benefit(
        d3,
        outdir,
        "3D",
        "bandwidth_gbs",
        "bandwidth ratio (bandwidth_variant / bandwidth_naive)",
        "3d_kahan_bandwidth_ratio.png",
        invert=False,
    )
    plot_error_vs_fp32(d3, outdir, "3D", "3d_error_fp32_over_variant.png")


def plot_combined_summary(df, outdir):
    """single summary figure: 2x2 grid with the most important comparisons"""
    if df.empty:
        print("  skipped summary: no rows found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # top-left: 2D bandwidth for all reaches, fp32
    ax = axes[0, 0]
    d2 = df[df["dim"] == 2]
    if not d2.empty:
        for R in sorted(d2["reach"].unique()):
            sub = d2[(d2["variant"] == "cuda_fp32") & (d2["reach"] == R)]
            if not sub.empty:
                ax.plot(sub["grid_size"], sub["bandwidth_gbs"], "o-", label=f"R={R}", markersize=5)
        ax.axhline(y=PEAK_BW, color="red", linestyle="--", alpha=0.4, label=f"peak {PEAK_BW}")
        style_ax(ax, "N", "bandwidth (GB/s)", "2D fp32 bandwidth by reach")
    else:
        ax.set_title("2D fp32 bandwidth by reach")
        ax.text(0.5, 0.5, "No 2D data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    # top-right: 3D bandwidth for all reaches, fp32
    ax = axes[0, 1]
    d3 = df[df["dim"] == 3]
    if not d3.empty:
        for R in sorted(d3["reach"].unique()):
            sub = d3[(d3["variant"] == "cuda_fp32_3d") & (d3["reach"] == R)]
            if not sub.empty:
                ax.plot(sub["grid_size"], sub["bandwidth_gbs"], "s-", label=f"R={R}", markersize=5)
        ax.axhline(y=PEAK_BW, color="red", linestyle="--", alpha=0.4, label=f"peak {PEAK_BW}")
        style_ax(ax, "N", "bandwidth (GB/s)", "3D fp32 bandwidth by reach")
    else:
        ax.set_title("3D fp32 bandwidth by reach")
        ax.text(0.5, 0.5, "No 3D data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    # bottom-left: Kahan vs naive vs tiled accuracy (2D, largest grid per reach)
    ax = axes[1, 0]
    if not d2.empty:
        reaches = sorted(d2["reach"].unique())
        x = np.arange(len(reaches))
        width = 0.2
        naive_err = []
        kahan_err = []
        kahan_tiled_err = []
        for R in reaches:
            sub_n = d2[(d2["variant"] == "cuda_fp16_naive") & (d2["reach"] == R)]
            sub_k = d2[(d2["variant"] == "cuda_fp16_kahan") & (d2["reach"] == R)]
            sub_kt = d2[(d2["variant"] == "cuda_fp16_kahan_tiled") & (d2["reach"] == R)]
            naive_err.append(sub_n["max_abs_error"].iloc[-1] if len(sub_n) > 0 else 0)
            kahan_err.append(sub_k["max_abs_error"].iloc[-1] if len(sub_k) > 0 else 0)
            kahan_tiled_err.append(sub_kt["max_abs_error"].iloc[-1] if len(sub_kt) > 0 else 0)
        ax.bar(x - width, naive_err, width, label="fp16 naive", color=COLORS["cuda_fp16_naive"])
        ax.bar(x, kahan_err, width, label="fp16+Kahan", color=COLORS["cuda_fp16_kahan"])
        ax.bar(x + width, kahan_tiled_err, width, label="fp16+Kahan tiled", color=COLORS["cuda_fp16_kahan_tiled"])
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"R={R}" for R in reaches])
        style_ax(ax, "stencil reach", "max |error| (largest N)", "2D Kahan variants accuracy")
    else:
        ax.set_title("2D Kahan variants accuracy")
        ax.text(0.5, 0.5, "No 2D data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    # bottom-right: GPU speedup over CPU, 3D N=128 across reaches
    ax = axes[1, 1]
    if not d3.empty:
        reaches = sorted(d3["reach"].unique())
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
    else:
        ax.set_title("3D GPU speedup (N=128)")
        ax.text(0.5, 0.5, "No 3D data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    fig.suptitle("Heat stencil benchmark summary (GTX 1650 Max-Q)", fontsize=14, fontweight="bold")
    add_figure_hint(fig, "Panel-specific: lower error is better; higher bandwidth and speedup are better")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{outdir}/summary.png", dpi=150, bbox_inches="tight")
    print(f"  saved {outdir}/summary.png")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmarks.csv"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "results"

    df = pd.read_csv(csv_path)
    df = prepare_plot_df(df)
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
