"""Tests for visuals module."""

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from stability.visuals import (
    plot_mean_variance_scatter,
    plot_metric_aggregate,
    plot_metric_correlation_heatmap,
)


class TestPlotMetricAggregate:
    """Tests for plot_metric_aggregate function."""

    def test_plot_metric_aggregate_basic(self) -> None:
        """Test basic plotting with multiple temperature values."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "temperature": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
                "ndcg@5": [0.1, 0.12, 0.15, 0.14, 0.18, 0.19],
            }
        )

        fig, ax = plot_metric_aggregate(df, "ndcg@5", rng, n_boot=100)

        try:
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, plt.Axes)
            # Check that the plot has content
            lines = ax.get_lines()
            assert len(lines) > 0
            # Check labels
            assert ax.get_xlabel() == "temperature"
            assert ax.get_ylabel() == "ndcg@5"
        finally:
            plt.close(fig)

    def test_plot_metric_aggregate_single_temperature(self) -> None:
        """Test edge case with single temperature value."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "temperature": [0.0, 0.0, 0.0],
                "precision@10": [0.2, 0.25, 0.22],
            }
        )

        fig, ax = plot_metric_aggregate(df, "precision@10", rng, n_boot=100)

        try:
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, plt.Axes)
            # Should still render without errors
            lines = ax.get_lines()
            assert len(lines) > 0
        finally:
            plt.close(fig)

    def test_plot_metric_aggregate_with_many_temperatures(self) -> None:
        """Test with many temperature values."""
        rng = np.random.default_rng(42)
        temperatures = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        data = []
        for t in temperatures:
            for _ in range(5):  # 5 samples per temperature
                data.append(
                    {
                        "temperature": t,
                        "hit_rate@5": 0.1 + t * 0.3 + rng.normal(0, 0.05),
                    }
                )
        df = pd.DataFrame(data)

        fig, ax = plot_metric_aggregate(df, "hit_rate@5", rng, n_boot=100)

        try:
            assert isinstance(fig, plt.Figure)
            # Check that we have lines for each unique temperature
            lines = ax.get_lines()
            assert len(lines) > 0
        finally:
            plt.close(fig)

    def test_plot_metric_aggregate_custom_ci(self) -> None:
        """Test with custom confidence interval."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "temperature": [0.0, 0.0, 1.0, 1.0],
                "recall@3": [0.3, 0.35, 0.4, 0.45],
            }
        )

        fig, ax = plot_metric_aggregate(df, "recall@3", rng, n_boot=100, ci=90.0)

        try:
            assert isinstance(fig, plt.Figure)
            # Check legend contains CI label
            legend = ax.get_legend()
            assert legend is not None
            texts = [t.get_text() for t in legend.get_texts()]
            assert any("90.0% CI" in t for t in texts)
        finally:
            plt.close(fig)


class TestPlotMetricCorrelationHeatmap:
    """Tests for plot_metric_correlation_heatmap function."""

    def test_plot_metric_correlation_heatmap_basic(self) -> None:
        """Test basic heatmap with multiple metric columns."""
        df = pd.DataFrame(
            {
                "ndcg@5": [0.1, 0.2, 0.3, 0.4, 0.5],
                "precision@5": [0.15, 0.25, 0.35, 0.45, 0.55],
                "recall@5": [0.2, 0.3, 0.4, 0.5, 0.6],
                "other_column": [1, 2, 3, 4, 5],
            }
        )
        metrics = ["ndcg@5", "precision@5", "recall@5"]

        result = plot_metric_correlation_heatmap(df, metrics)

        try:
            assert result is not None
            fig, ax = result
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, plt.Axes)
            # Check title
            assert ax.get_title() == "Spearman correlation among metrics"
        finally:
            if result is not None:
                plt.close(result[0])

    def test_plot_metric_correlation_heatmap_empty_metrics(self) -> None:
        """Test with empty metrics list - should return None."""
        df = pd.DataFrame(
            {
                "ndcg@5": [0.1, 0.2, 0.3],
                "precision@5": [0.15, 0.25, 0.35],
            }
        )
        metrics: list[str] = []

        result = plot_metric_correlation_heatmap(df, metrics)

        assert result is None

    def test_plot_metric_correlation_heatmap_missing_columns(self) -> None:
        """Test with metrics not in DataFrame - should return None."""
        df = pd.DataFrame(
            {
                "ndcg@5": [0.1, 0.2, 0.3],
                "precision@5": [0.15, 0.25, 0.35],
            }
        )
        metrics = ["hit_rate@10", "f1@10", "mrr@10"]  # None of these exist

        result = plot_metric_correlation_heatmap(df, metrics)

        assert result is None

    def test_plot_metric_correlation_heatmap_partial_columns(self) -> None:
        """Test with some metrics in DataFrame and some missing."""
        df = pd.DataFrame(
            {
                "ndcg@5": [0.1, 0.2, 0.3, 0.4, 0.5],
                "precision@5": [0.15, 0.25, 0.35, 0.45, 0.55],
            }
        )
        metrics = ["ndcg@5", "precision@5", "missing_metric"]

        result = plot_metric_correlation_heatmap(df, metrics)

        try:
            # Should still work with the valid columns
            assert result is not None
            fig, ax = result
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, plt.Axes)
        finally:
            if result is not None:
                plt.close(result[0])

    def test_plot_metric_correlation_heatmap_single_metric(self) -> None:
        """Test with single metric - should still produce heatmap."""
        df = pd.DataFrame(
            {
                "ndcg@5": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        metrics = ["ndcg@5"]

        result = plot_metric_correlation_heatmap(df, metrics)

        try:
            assert result is not None
            fig, _ax = result
            assert isinstance(fig, plt.Figure)
        finally:
            if result is not None:
                plt.close(result[0])


class TestPlotMeanVarianceScatter:
    """Tests for plot_mean_variance_scatter function."""

    def test_plot_mean_variance_scatter_basic(self) -> None:
        """Test basic scatter plot with metric and metric_std columns."""
        df = pd.DataFrame(
            {
                "ndcg@5": [0.1, 0.2, 0.3, 0.4, 0.5],
                "ndcg@5_std": [0.01, 0.02, 0.015, 0.025, 0.02],
            }
        )

        fig, ax = plot_mean_variance_scatter(df, "ndcg@5")

        try:
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, plt.Axes)
            # Check labels
            assert "ndcg@5" in ax.get_xlabel()
            assert "ndcg@5" in ax.get_ylabel()
            assert "Standard Deviation" in ax.get_ylabel()
            # Check title
            assert "ndcg@5" in ax.get_title()
        finally:
            plt.close(fig)

    def test_plot_mean_variance_scatter_with_groupby(self) -> None:
        """Test scatter plot with group_by functionality."""
        df = pd.DataFrame(
            {
                "ndcg@5": [0.1, 0.2, 0.3, 0.4, 0.5, 0.15],
                "ndcg@5_std": [0.01, 0.02, 0.015, 0.025, 0.02, 0.012],
                "temperature": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
            }
        )

        fig, ax = plot_mean_variance_scatter(df, "ndcg@5", group_by="temperature")

        try:
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, plt.Axes)
            # Check that legend exists (groups should create legend entries)
            legend = ax.get_legend()
            assert legend is not None
            # Check that legend has correct number of entries (3 temperatures)
            texts = [t.get_text() for t in legend.get_texts()]
            assert len(texts) == 3
        finally:
            plt.close(fig)

    def test_plot_mean_variance_scatter_missing_columns(self) -> None:
        """Test ValueError when required columns are missing."""
        df = pd.DataFrame(
            {
                "ndcg@5": [0.1, 0.2, 0.3],
                # Missing ndcg@5_std column
            }
        )

        with pytest.raises(ValueError, match="Required columns not found"):
            plot_mean_variance_scatter(df, "ndcg@5")

    def test_plot_mean_variance_scatter_use_variance(self) -> None:
        """Test use_std=False parameter to plot variance instead of std."""
        df = pd.DataFrame(
            {
                "precision@10": [0.2, 0.3, 0.4, 0.5],
                "precision@10_std": [0.02, 0.03, 0.025, 0.035],
            }
        )

        fig, ax = plot_mean_variance_scatter(df, "precision@10", use_std=False)

        try:
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, plt.Axes)
            # Check that ylabel shows Variance instead of Standard Deviation
            assert "Variance" in ax.get_ylabel()
            assert "Standard Deviation" not in ax.get_ylabel()
        finally:
            plt.close(fig)
