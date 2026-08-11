"""Visualization utilities for stability analysis.

This module provides plotting functions for:
- Metric aggregation with bootstrap confidence intervals
- Correlation heatmaps
- Mean-variance scatter plots for stability analysis
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stability.utils import bootstrap_mean_ci

logger = logging.getLogger(__name__)


def plot_metric_aggregate(
    df_stats: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
    n_boot: int = 1000,
    ci: float = 95.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Return a plot of the aggregate metric with confidence intervals.

    Args:
        df_stats: DataFrame with 'temperature' column and metric column
        metric: Name of the metric column to aggregate
        rng: NumPy random generator for bootstrap sampling
        n_boot: Number of bootstrap samples (default 1000)
        ci: Confidence interval percentage (default 95.0)

    Returns:
        Tuple of (Figure, Axes) with the aggregated metric plot

    """
    df = df_stats.copy()
    temps = sorted(df["temperature"].unique())
    agg_rows = []
    for t in temps:
        vals = df.loc[df["temperature"] == t, metric].to_numpy()
        mean_val, low, high = bootstrap_mean_ci(vals, rng, n_boot, ci)
        agg_rows.append(
            {
                "temperature": t,
                "mean": mean_val,
                "low": low,
                "high": high,
                "n": int(np.isfinite(vals).sum()),
            },
        )
    agg = pd.DataFrame(agg_rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(agg["temperature"], agg["mean"], marker="o", label=f"mean {metric}")
    ax.fill_between(
        agg["temperature"],
        agg["low"],
        agg["high"],
        alpha=0.2,
        label=f"{ci}% CI",
    )
    ax.set(xlabel="temperature", ylabel=metric)
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    sns.despine()
    return fig, ax


def plot_metric_correlation_heatmap(
    df_stats: pd.DataFrame,
    metrics: list[str],
    **kwargs: dict,
) -> tuple[plt.Figure, plt.Axes] | None:
    """Return a heatmap showing Spearman correlation among metrics.

    Args:
        df_stats: DataFrame containing metric columns
        metrics: List of metric names to correlate
        **kwargs: Additional arguments passed to seaborn.heatmap

    Returns:
        Tuple of (Figure, Axes) with the correlation heatmap, or None if no
        valid metrics are found in the DataFrame. Returns None when all
        specified metrics are missing from df_stats columns.

    """
    df = df_stats.copy()
    cols = [m for m in metrics if m in df.columns]
    if not cols:
        logger.warning("No metric columns found to correlate")
        return None
    corr = df[cols].corr(method="spearman", min_periods=1)
    fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 2, 0.6 * len(cols) + 2))
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        **kwargs,
    )
    ax.set_title("Spearman correlation among metrics")
    return fig, ax


def plot_mean_variance_scatter(
    df_stats: pd.DataFrame,
    metric_base: str,
    group_by: str | None = None,
    *,
    use_std: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot mean vs variability scatter for stability analysis.

    This visualization helps identify the sweet spot where a metric achieves
    both high performance (high mean) AND stability (low variability). The ideal
    region is the bottom-right: high mean with low standard deviation, indicating
    consistent good performance across samples.

    Args:
        df_stats: DataFrame containing experiment metrics with statistical moments
        metric_base: Base metric name (e.g., "ndcg@5", "precision@10")
        group_by: Optional column name to group and color-code points
        use_std: If True, plot standard deviation; if False, plot variance

    Returns:
        Tuple of (Figure, Axes) objects

    Raises:
        ValueError: If required columns not found in DataFrame

    Notes:
        - Requires columns: {metric_base}, {metric_base}_std
        - Bottom-right quadrant (high mean, low std) = IDEAL (good & stable)
        - Top-right quadrant (high mean, high std) = risky (good but unreliable)
        - Bottom-left quadrant (low mean, low std) = consistent but poor
        - Top-left quadrant (low mean, high std) = worst (poor & unreliable)

    """
    df = df_stats.copy()

    # Validate required columns exist
    mean_col = metric_base
    std_col = f"{metric_base}_std"

    if mean_col not in df.columns or std_col not in df.columns:
        msg = f"Required columns not found. Need: '{mean_col}' and '{std_col}'"
        raise ValueError(msg)

    # extract data
    mean_vals = df[mean_col].to_numpy()
    std_vals = df[std_col].to_numpy()
    variability_vals = std_vals if use_std else std_vals**2

    # create figure
    fig, ax = plt.subplots()

    # plot with optional grouping
    if group_by and group_by in df.columns:
        groups = df[group_by].unique()
        cmap = plt.colormaps.get_cmap("viridis").resampled(len(groups))

        for idx, group_val in enumerate(sorted(groups)):
            mask = df[group_by] == group_val
            ax.scatter(
                mean_vals[mask],
                variability_vals[mask],
                label=f"{group_by}={group_val:.1f}"
                if isinstance(group_val, (int, float))
                else f"{group_by}={group_val}",
                alpha=0.7,
                s=100,
                color=cmap(idx),
                edgecolors="black",
                linewidths=0.5,
            )
        ax.legend(loc="best", framealpha=0.9)
    else:
        ax.scatter(
            mean_vals,
            variability_vals,
            s=100,
            color="steelblue",
            edgecolors="black",
            linewidths=0.5,
        )

    # labels and formatting
    var_label = "Standard Deviation" if use_std else "Variance"
    ax.set_xlabel(f"{metric_base} (Mean)", fontsize=12)
    ax.set_ylabel(f"{metric_base} ({var_label})", fontsize=12)
    ax.set_title(
        f"Mean-Variance Analysis: {metric_base}",
        fontsize=14,
        fontweight="bold",
    )

    # add reference lines at median
    if len(mean_vals) > 0:
        median_mean = np.nanmedian(mean_vals)
        median_var = np.nanmedian(variability_vals)
        ax.axvline(median_mean, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(median_var, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # annotate ideal region (bottom-right)
    ax.text(
        0.95,
        0.05,
        "IDEAL:\nHigh mean\nLow variability",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.5},
    )

    sns.despine()
    plt.tight_layout()

    return fig, ax
