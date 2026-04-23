#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "typer",
#   "matplotlib",
#   "pandas",
# ]
# ///
"""Plot GPU vs CPU SuperBloom comparison from a CSV file.

Reads a two-row CSV (implementation=gpu / cpu_rust) produced by
compare_gpu_vs_cpu.py and generates a throughput comparison bar chart.

Usage:
    python scripts/plot_gpu_vs_cpu.py build/gpu_vs_cpu_comparison.csv
    python scripts/plot_gpu_vs_cpu.py build/gpu_vs_cpu_comparison.csv -o comparison.pdf
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from plot_utils import (
    AXIS_LABEL_FONT_SIZE,
    BAR_FONT_SIZE,
    FILTER_STYLES,
    GRID_ALPHA,
    TITLE_FONT_SIZE,
    save_figure,
)

app = typer.Typer(add_completion=False)


def generate_plot(
    gpu_medians: dict[str, float],
    cpu_medians: dict[str, float],
    output_path: Path,
    config: Optional[dict[str, int]] = None,
) -> None:
    """Generate a comparison bar chart from GPU and CPU median metrics.

    Args:
        gpu_medians: Metric name -> median value for GPU.
        cpu_medians: Metric name -> median value for CPU.
        output_path: Where to write the PDF.
        config: Optional dict with keys k, s, m, n_hashes for the title.
    """
    import matplotlib.pyplot as plt

    metrics = [
        ("index_kmers_per_s", "Index Throughput", 1e9, "B k-mers/s"),
        ("query_kmers_per_s", "Query Throughput", 1e9, "B k-mers/s"),
    ]

    available = [
        (k, label, scale, unit)
        for k, label, scale, unit in metrics
        if k in gpu_medians and k in cpu_medians
    ]

    if not available:
        typer.secho("No common metrics to plot.", fg=typer.colors.YELLOW, err=True)
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    gpu_style = FILTER_STYLES["superbloom"]
    cpu_color = "#E8871E"

    for ax, (key, label, scale, unit) in zip(axes, available):
        gpu_val = gpu_medians[key] / scale
        cpu_val = cpu_medians[key] / scale

        bars = ax.bar(
            ["GPU", "CPU"],
            [gpu_val, cpu_val],
            color=[gpu_style["color"], cpu_color],
            edgecolor="black",
            linewidth=0.5,
        )

        for bar, val in zip(bars, [gpu_val, cpu_val]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3g}",
                ha="center",
                va="bottom",
                fontsize=BAR_FONT_SIZE,
                fontweight="bold",
            )

        # ax.set_yscale("log")
        ax.set_ylabel(
            f"{label} [{unit}]", fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold"
        )
        ax.set_title(label, fontsize=TITLE_FONT_SIZE, fontweight="bold")
        ax.grid(True, axis="y", ls="--", alpha=GRID_ALPHA)

    if config:
        config_str = f"K={config['k']} S={config['s']} M={config['m']} H={config['n_hashes']}"
    else:
        config_str = ""
    suptitle = "GPU vs CPU SuperBloom" + (f" ({config_str})" if config_str else "")
    fig.suptitle(
        suptitle,
        fontsize=TITLE_FONT_SIZE + 2,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, output_path)


@app.command()
def main(
    csv_path: Path = typer.Argument(..., help="Input CSV produced by compare_gpu_vs_cpu.py"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output PDF path (default: <csv_stem>.pdf)"
    ),
) -> None:
    """Generate GPU vs CPU comparison plot from a CSV file."""
    df = pd.read_csv(csv_path)

    gpu_row = df[df["implementation"] == "gpu"]
    cpu_row = df[df["implementation"] == "cpu_rust"]

    if gpu_row.empty or cpu_row.empty:
        typer.secho(
            "CSV must contain rows with implementation='gpu' and 'cpu_rust'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    gpu_medians = gpu_row.iloc[0].drop("implementation").dropna().to_dict()
    cpu_medians = cpu_row.iloc[0].drop("implementation").dropna().to_dict()

    if output is None:
        output = csv_path.with_suffix(".pdf").absolute()

    generate_plot(gpu_medians, cpu_medians, output)


if __name__ == "__main__":
    app()
