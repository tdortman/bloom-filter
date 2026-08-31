#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
"""Plot Shapley-attributed design contributions from benchmark CSV."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot Shapley-attributed design contributions")

_COMPONENTS = {
    "Insert": (
        (1, "Sectorized shards", "#7B4397"),
        (2, "Shared-memory tiling", "#F18F01"),
        (4, "Segmented warp reduction", "#6A994E"),
    ),
    "Query": (
        (1, "Sectorized shards", "#7B4397"),
        (2, "Shared-memory tiling", "#F18F01"),
    ),
}
_VARIANTS = {
    "Insert": {
        0: "BaselineInsert",
        1: "SectorizedInsert",
        2: "TiledInsert",
        3: "SectorizedTiledInsert",
        4: "SegmentedInsert",
        5: "SectorizedSegmentedInsert",
        6: "TiledSegmentedInsert",
        7: "FullInsert",
    },
    "Query": {
        0: "BaselineQuery",
        1: "SectorizedQuery",
        2: "TiledQuery",
        3: "FullQuery",
    },
}
_BENCHMARK_TO_OPERATION = {
    benchmark: operation
    for operation, variants in _VARIANTS.items()
    for benchmark in variants.values()
}
_TARGET_SIZES = (1 << 22, 1 << 28)


def load_benchmark_csv(csv_path: Path) -> pd.DataFrame:
    """Load a benchmark CSV after its optional preamble."""
    lines = csv_path.read_text().splitlines()
    header_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if line.lstrip("\ufeff \t").startswith("name,")
        ),
        None,
    )
    if header_idx is None:
        raise typer.BadParameter(f"No Google Benchmark CSV header found in {csv_path}")

    lines[header_idx] = lines[header_idx].lstrip("\ufeff \t")
    data = io.StringIO("\n".join(lines[header_idx:]))
    header = pd.read_csv(data, nrows=0)
    data.seek(0)
    return pd.read_csv(data, usecols=list(header.columns))


def parse_benchmark_row(name: str) -> tuple[str, str, int] | None:
    """Parse an ablation benchmark name into operation, variant, and size."""
    parts = str(name).strip().strip('"').split("/")
    if len(parts) < 3 or parts[0] != "DesignAblationFixture":
        return None

    benchmark = parts[1]
    operation = _BENCHMARK_TO_OPERATION.get(benchmark)
    if operation is None or not parts[2].isdigit():
        return None
    return operation, benchmark, int(parts[2])


def load_throughput_series(csv_path: Path) -> pd.DataFrame:
    """Load positive median items-per-second rows for the ablation fixture."""
    df = load_benchmark_csv(csv_path)
    if "name" not in df or "items_per_second" not in df:
        raise typer.BadParameter("CSV must contain name and items_per_second columns")

    median_rows = df[df["name"].astype(str).str.endswith("_median", na=False)]
    rows: list[dict[str, object]] = []
    for _, row in median_rows.iterrows():
        parsed = parse_benchmark_row(row["name"])
        if parsed is None:
            continue
        operation, benchmark, num_symbols = parsed
        throughput = pd.to_numeric(row["items_per_second"], errors="coerce")
        if pd.isna(throughput) or float(throughput) <= 0:
            continue
        rows.append(
            {
                "operation": operation,
                "benchmark": benchmark,
                "num_symbols": num_symbols,
                "throughput": float(throughput),
            }
        )

    if not rows:
        raise typer.BadParameter(f"No design-ablation median rows found in {csv_path}")
    return pd.DataFrame(rows)


def compute_shapley_contributions(
    data: pd.DataFrame, operation: str, num_symbols: int
) -> tuple[float, list[float]]:
    """Split full speedup into order-independent component effects."""
    variants = _VARIANTS[operation]
    point = data[
        (data["operation"] == operation) & (data["num_symbols"] == num_symbols)
    ].set_index("benchmark")["throughput"]
    missing = [
        benchmark for benchmark in variants.values() if benchmark not in point.index
    ]
    if missing:
        raise typer.BadParameter(
            f"Missing {operation} subset rows at 2^{num_symbols.bit_length() - 1} symbols"
        )

    baseline = float(point.loc[variants[0]])
    relative = {
        mask: float(point.loc[benchmark]) / baseline
        for mask, benchmark in variants.items()
    }
    component_count = len(_COMPONENTS[operation])
    denominator = math.factorial(component_count)
    contributions: list[float] = []
    for bit, _, _ in _COMPONENTS[operation]:
        contribution = 0.0
        for mask in variants:
            if mask & bit:
                continue
            subset_size = mask.bit_count()
            weight = (
                math.factorial(subset_size)
                * math.factorial(component_count - subset_size - 1)
                / denominator
            )
            contribution += weight * (relative[mask | bit] - relative[mask])
        contributions.append(contribution)

    full_speedup = relative[(1 << component_count) - 1]
    if not math.isclose(1.0 + sum(contributions), full_speedup, rel_tol=1e-9):
        raise RuntimeError("Shapley contributions do not conserve full speedup")
    return full_speedup, contributions


def plot_speedups(data: pd.DataFrame, output_pdf: Path) -> None:
    """Plot Shapley-attributed design contributions in one chart."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
    positions = (0.0, 0.65, 1.65, 2.3)
    entries = [
        (operation, size, compute_shapley_contributions(data, operation, size))
        for operation in _COMPONENTS
        for size in _TARGET_SIZES
    ]
    text_colors = ("white", "black", "white")
    legend_labels: set[str] = set()
    y_min = 0.0
    y_max = 1.0

    for position, (operation, _, (full_speedup, contributions)) in zip(
        positions, entries
    ):
        ax.bar(
            position,
            1.0,
            width=0.5,
            color="#BDBDBD",
            edgecolor="black",
            linewidth=pu.BAR_EDGE_WIDTH,
            label=pu.paper_text("Baseline") if not legend_labels else None,
            zorder=3,
        )
        legend_labels.add("Baseline")
        ax.text(
            position,
            0.5,
            "1.00$\\times$",
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            zorder=4,
        )
        positive_bottom = 1.0
        negative_bottom = 1.0
        for (_, label, color), contribution, text_color in zip(
            _COMPONENTS[operation], contributions, text_colors
        ):
            if contribution >= 0.0:
                bottom = positive_bottom
                positive_bottom += contribution
            else:
                bottom = negative_bottom
                negative_bottom += contribution
            legend_label = pu.paper_text(label)
            ax.bar(
                position,
                contribution,
                bottom=bottom,
                width=0.5,
                color=color,
                edgecolor="black",
                linewidth=pu.BAR_EDGE_WIDTH,
                label=legend_label if legend_label not in legend_labels else None,
                zorder=3,
            )
            legend_labels.add(legend_label)
            if abs(contribution) >= 0.08:
                ax.text(
                    position,
                    bottom + contribution / 2,
                    f"{contribution:+.2f}$\\times$",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=text_color,
                    fontweight="bold",
                    zorder=4,
                )

        ax.hlines(
            full_speedup,
            position - 0.28,
            position + 0.28,
            color="black",
            linewidth=1.2,
            zorder=5,
        )
        ax.text(
            position,
            max(full_speedup, positive_bottom) + 0.05,
            rf"$\mathbf{{{full_speedup:.2f}\times}}$",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        y_min = min(y_min, negative_bottom, full_speedup)
        y_max = max(y_max, positive_bottom, full_speedup)

    tick_labels = [
        rf"\textbf{{{operation}}}" + "\n" + rf"$2^{{{size.bit_length() - 1}}}$"
        for operation, size, _ in entries
    ]
    ax.axhline(1.0, color="#333333", linewidth=pu.REFERENCE_LINE_WIDTH, zorder=2)
    ax.set_xticks(positions, tick_labels, fontsize=7)
    ax.set_ylabel(
        pu.paper_text("Throughput relative to baseline", bold=True), fontsize=8
    )
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="y", ls="--", alpha=pu.GRID_ALPHA, zorder=0)
    margin = max(0.2, (y_max - y_min) * 0.08)
    ax.set_ylim(min(0.0, y_min - margin), y_max + margin)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=7,
        framealpha=pu.LEGEND_FRAME_ALPHA,
        handlelength=1.8,
        columnspacing=0.8,
    )
    pu.save_figure(
        fig,
        output_pdf,
        f"Design ablation speedup plot saved to {output_pdf}",
    )


@app.command()
def main(
    csv_path: Annotated[Path, typer.Argument(help="Design-ablation benchmark CSV")],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for plots (default: build/)",
        ),
    ] = None,
) -> None:
    """Plot Shapley-attributed contributions to full throughput."""
    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))
    data = load_throughput_series(csv_path)
    plot_speedups(data, output_dir / "design_ablation_speedup.pdf")


if __name__ == "__main__":
    app()
