#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

app = typer.Typer(help="Compare filters for throughput and memory usage")

OPERATION_MARKERS = {
    "Insert": "o",
    "Query": "s",
    "Delete": "^",
}


@app.command()
def main(
    csv_file: Path = typer.Argument(
        "-",
        help="Path to CSV file, or '-' to read from stdin (default: stdin)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
):
    """
    Generate throughput and memory comparison plots from benchmark CSV results.

    Plots throughput [B elem/s] and memory usage [MiB] vs input size for
    various benchmarks in a single figure with two y-axes.

    Examples:
        cat results.csv | compare_filters.py
        compare_filters.py < results.csv
        compare_filters.py results.csv
        compare_filters.py results.csv -o custom/dir
    """
    try:
        if str(csv_file) == "-":
            import sys

            df = pd.read_csv(sys.stdin)
        else:
            df = pd.read_csv(csv_file)
    except Exception as e:
        typer.secho(f"Error parsing CSV: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    throughput_data = defaultdict(dict)
    memory_by_filter = defaultdict(dict)

    for _, row in df.iterrows():
        name = row["name"]
        if "/" not in name:
            continue

        parsed = pu.parse_fixture_benchmark_name(name)
        if parsed is None:
            continue
        filter_key, operation, size = parsed
        # Fixture names may embed tuning params (e.g. superbloom_k31_...);
        # normalize to the canonical filter family for styling/grouping.
        filter_key = filter_key.split("_", 1)[0]
        benchmark_key = (filter_key, operation)

        # Only process median records
        if "_median" not in name:
            continue

        if operation not in OPERATION_MARKERS:
            continue

        try:
            items_per_second = row.get("items_per_second")
            if pd.notna(items_per_second):
                throughput_beps = pu.to_billion_elems_per_sec(items_per_second)
                throughput_data[benchmark_key][size] = throughput_beps

            memory_bytes = row.get("memory_bytes")
            if pd.notna(memory_bytes):
                memory_mib = float(memory_bytes) / (1024 * 1024)
                existing = memory_by_filter[filter_key].get(size, 0.0)
                memory_by_filter[filter_key][size] = max(existing, memory_mib)
        except (ValueError, KeyError):
            continue

    if not throughput_data:
        typer.secho("No throughput data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not memory_by_filter:
        typer.secho("No memory_bytes data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    fig, ax_throughput = plt.subplots(figsize=(12, 8.5))
    ax_throughput.set_facecolor("white")
    fig.subplots_adjust(top=0.78)

    def get_last_throughput(bench_key):
        sizes = sorted(throughput_data[bench_key].keys())
        if sizes:
            return throughput_data[bench_key][sizes[-1]]
        return 0

    benchmark_keys = sorted(
        throughput_data.keys(), key=get_last_throughput, reverse=True
    )

    filter_handles: dict[str, Line2D] = {}
    seen_operations = set()

    for bench_key in benchmark_keys:
        filter_key, operation = bench_key

        style = pu.get_filter_style(filter_key)
        color = style["color"]
        marker = OPERATION_MARKERS.get(operation, style["marker"])
        filter_label = pu.get_filter_display_name(filter_key)
        seen_operations.add(operation)

        throughput_sizes = sorted(throughput_data[bench_key].keys())
        throughput_points = [
            (size, throughput_data[bench_key][size])
            for size in throughput_sizes
            if throughput_data[bench_key][size] > 0
        ]
        throughput_values = [value for _, value in throughput_points]
        throughput_sizes = [size for size, _ in throughput_points]

        if throughput_sizes:
            ax_throughput.plot(
                throughput_sizes,
                throughput_values,
                linestyle="-",
                marker=marker,
                color=color,
                linewidth=pu.LINE_WIDTH,
                markersize=pu.MARKER_SIZE,
            )

        if filter_label not in filter_handles:
            filter_handles[filter_label] = Line2D(
                [0],
                [0],
                color=color,
                linestyle="-",
                linewidth=pu.LINE_WIDTH,
                label=filter_label,
            )


    ax_throughput.set_xlabel(
        "Input Size [bases]", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax_throughput.set_ylabel(
        pu.THROUGHPUT_LABEL, fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax_throughput.set_xscale("log", base=2)
    ax_throughput.yaxis.set_major_locator(MultipleLocator(10))
    # ax_throughput.set_yscale("log")
    ax_throughput.grid(True, which="major", ls="--", alpha=pu.GRID_ALPHA)
    # Keep both y-axis spines visible and distinct.
    ax_throughput.spines["left"].set_visible(True)
    ax_throughput.spines["right"].set_visible(False)

    ax_throughput.tick_params(axis="y", which="both", direction="out")

    # Keep a compact filter legend plus explicit operation marker legend.
    filter_legend = ax_throughput.legend(
        handles=list(filter_handles.values()),
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=max(1, min(2, len(filter_handles))),
        framealpha=pu.LEGEND_FRAME_ALPHA,
    )
    filter_legend.set_clip_on(False)
    ax_throughput.add_artist(filter_legend)

    operation_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=OPERATION_MARKERS[op],
            linestyle="None",
            markersize=pu.MARKER_SIZE,
            label=op,
        )
        for op in ("Insert", "Query", "Delete")
        if op in seen_operations
    ]
    op_legend = None
    if operation_handles:
        op_legend = ax_throughput.legend(
            handles=operation_handles,
            fontsize=pu.LEGEND_FONT_SIZE,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.02),
            ncol=max(1, len(operation_handles)),
            framealpha=pu.LEGEND_FRAME_ALPHA,
        )
        op_legend.set_clip_on(False)

    output_file = output_dir / "benchmark_throughput_memory.pdf"
    extra_artists = [filter_legend]
    if op_legend is not None:
        extra_artists.append(op_legend)
    plt.savefig(
        output_file,
        bbox_extra_artists=extra_artists,
        bbox_inches="tight",
        transparent=True,
        format="pdf",
        dpi=600,
    )
    typer.secho(
        f"Throughput+memory plot saved to {output_file}", fg=typer.colors.GREEN
    )


if __name__ == "__main__":
    app()
