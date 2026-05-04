#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot FPR benchmark results")


SUPERBLOOM_FIXTURE_PATTERN = re.compile(
    r"^SuperBloomFixture(?P<s>\d+)?$", re.IGNORECASE
)
SUPERBLOOM_CONFIG_FIXTURE_PATTERN = re.compile(
    r"^SuperBloom_K(?P<k>\d+)_S(?P<s>\d+)_M(?P<m>\d+)_H(?P<h>\d+)_Fixture$",
    re.IGNORECASE,
)
SUPERBLOOM_CPU_CONFIG_FIXTURE_PATTERN = re.compile(
    r"^SuperBloomCpu_K(?P<k>\d+)_S(?P<s>\d+)_M(?P<m>\d+)_H(?P<h>\d+)_Fixture$",
    re.IGNORECASE,
)
CUCO_FIXTURE_PATTERN = re.compile(r"^CucoBloomFixture$", re.IGNORECASE)

SUPERBLOOM_VARIANT_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
SUPERBLOOM_VARIANT_LINESTYLES = ["-", "--", "-.", ":"]


def parse_superbloom_variant(fixture_name: str, row: pd.Series) -> Optional[int]:
    """Parse SuperBloom variant ``s`` value from fixture name or CSV counters."""
    match = SUPERBLOOM_FIXTURE_PATTERN.match(fixture_name)
    if match is not None:
        fixture_suffix = match.group("s")
        if fixture_suffix is not None:
            return int(fixture_suffix)

        row_s = row.get("s")
        if pd.notna(row_s):
            try:
                return int(float(row_s))
            except (TypeError, ValueError):
                return None

        return None

    config_match = SUPERBLOOM_CONFIG_FIXTURE_PATTERN.match(fixture_name)
    if config_match is not None:
        return int(config_match.group("s"))

    cpu_match = SUPERBLOOM_CPU_CONFIG_FIXTURE_PATTERN.match(fixture_name)
    if cpu_match is not None:
        return int(cpu_match.group("s"))

    return None


def extract_filter_series(name: str, row: pd.Series) -> Optional[tuple[str, str, str]]:
    """Extract filter series key, base style key, and display label."""
    stripped_name = str(name).strip('"')
    parts = stripped_name.split("/")

    if len(parts) >= 2:
        fixture_name = parts[0]
        operation = parts[1]
        superbloom_variant = parse_superbloom_variant(fixture_name, row)
        is_cpu = SUPERBLOOM_CPU_CONFIG_FIXTURE_PATTERN.match(fixture_name) is not None
        is_gpu = (
            SUPERBLOOM_CONFIG_FIXTURE_PATTERN.match(fixture_name) is not None
            or SUPERBLOOM_FIXTURE_PATTERN.match(fixture_name) is not None
        )
        is_cuco = CUCO_FIXTURE_PATTERN.match(fixture_name) is not None

        if is_gpu or is_cpu or is_cuco:
            if operation.upper() != "FPR":
                return None

            if is_cpu and superbloom_variant is not None:
                series_key = f"superbloom_cpu_s{superbloom_variant}"
                display_name = f"{pu.get_filter_display_name('superbloom_cpu')} (s={superbloom_variant})"
                return series_key, "superbloom_cpu", display_name

            if is_gpu and superbloom_variant is not None:
                series_key = f"superbloom_s{superbloom_variant}"
                display_name = f"{pu.get_filter_display_name('superbloom')} (s={superbloom_variant})"
                return series_key, "superbloom", display_name

            if is_cuco:
                return (
                    "cucobloom",
                    "cucobloom",
                    pu.get_filter_display_name("cucobloom"),
                )

    return None


def get_plot_style(filter_type: str, base_filter: str) -> dict[str, str]:
    """Get style for a filter series, including SuperBloom variant styling."""
    style = dict(
        pu.FILTER_STYLES.get(
            filter_type,
            pu.FILTER_STYLES.get(base_filter, {}),
        )
    )

    if base_filter in ("superbloom", "superbloom_cpu") and re.search(r"_s(\d+)$", filter_type):
        variant_match = re.search(r"_s(\d+)$", filter_type)
        if variant_match is not None:
            variant_index = int(variant_match.group(1))
            style["marker"] = SUPERBLOOM_VARIANT_MARKERS[
                variant_index % len(SUPERBLOOM_VARIANT_MARKERS)
            ]
            style["linestyle"] = SUPERBLOOM_VARIANT_LINESTYLES[
                variant_index % len(SUPERBLOOM_VARIANT_LINESTYLES)
            ]

    return style


def sort_filters_by_descending_fpr(
    fpr_data: dict[str, dict[float, float]],
    filter_display_names: dict[str, str],
) -> list[str]:
    """Sort filters by highest observed FPR (descending)."""

    def fpr_sort_key(filter_type: str) -> tuple[float, str]:
        values = fpr_data.get(filter_type, {}).values()
        max_fpr = max(values) if values else float("-inf")
        display_name = filter_display_names.get(
            filter_type,
            pu.get_filter_display_name(filter_type),
        )
        return (-max_fpr, display_name)

    return sorted(fpr_data.keys(), key=fpr_sort_key)


@app.command()
def main(
    csv_file: Path = typer.Argument(
        ...,
        help="Path to the CSV file containing benchmark results",
    ),
    output_dir: Path = typer.Option(
        Path("./build"),
        help="Directory to save output plots",
    ),
):
    """
    Parse FPR benchmark CSV results and generate plots.
    """
    if not csv_file.exists():
        typer.secho(f"CSV file not found: {csv_file}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.secho(f"Reading CSV from: {csv_file}", fg=typer.colors.CYAN)

    df = pd.read_csv(csv_file)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    # Dictionary structure: filter_type -> {memory_size: metric_value}
    fpr_data = defaultdict(dict)
    bits_per_item_data = defaultdict(dict)
    filter_display_names: dict[str, str] = {}
    filter_base_types: dict[str, str] = {}

    for _, row in df.iterrows():
        name = row["name"]
        filter_series = extract_filter_series(name, row)

        if filter_series is None:
            continue

        filter_type, base_filter_type, display_name = filter_series
        filter_display_names[filter_type] = display_name
        filter_base_types[filter_type] = base_filter_type

        memory_bytes = row.get("memory_bytes")
        fpr_percentage = row.get("fpr_percentage")
        bits_per_item = row.get("bits_per_item")

        if pd.notna(memory_bytes):
            if pd.notna(fpr_percentage):
                fpr_data[filter_type][memory_bytes] = fpr_percentage
            if pd.notna(bits_per_item):
                bits_per_item_data[filter_type][memory_bytes] = bits_per_item

    if not fpr_data:
        typer.secho("No FPR data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    fpr_sorted_filters = sort_filters_by_descending_fpr(
        dict(fpr_data),
        filter_display_names,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: FPR vs Memory Size
    fig, ax = plt.subplots(figsize=(12, 8))

    for filter_type in fpr_sorted_filters:
        memory_sizes = sorted(fpr_data[filter_type].keys())
        fpr_values = [fpr_data[filter_type][mem] for mem in memory_sizes]

        style = get_plot_style(
            filter_type,
            filter_base_types.get(filter_type, filter_type),
        )
        ax.plot(
            memory_sizes,
            fpr_values,
            label=filter_display_names.get(
                filter_type,
                pu.get_filter_display_name(filter_type),
            ),
            linewidth=pu.LINE_WIDTH,
            markersize=pu.MARKER_SIZE,
            **style,  # type: ignore
        )

    ax.set_xlabel(
        "Sequence length [bases]", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax.set_ylabel(
        "False Positive Rate [%]", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=pu.AXIS_LABEL_FONT_SIZE)
    ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    axes_box = ax.get_position()
    legend_center_x = (axes_box.x0 + axes_box.x1) / 2
    legend_y = axes_box.y1 + 0.01
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="lower center",
        bbox_to_anchor=(legend_center_x, legend_y),
        ncol=max(1, min(4, len(labels))),
        framealpha=pu.LEGEND_FRAME_ALPHA,
    )

    output_file = output_dir / "fpr_vs_memory.pdf"
    plt.savefig(
        output_file,
        bbox_inches="tight",
        transparent=True,
        format="pdf",
        dpi=600,
    )
    typer.secho(
        f"FPR vs memory plot saved to {output_file.absolute()}",
        fg=typer.colors.GREEN,
    )
    plt.close()

    # Plot 2: Bits per Item vs Memory Size
    fig, ax = plt.subplots(figsize=(12, 8))

    bits_filter_order = [
        f for f in fpr_sorted_filters if f in bits_per_item_data
    ] + sorted(set(bits_per_item_data.keys()) - set(fpr_sorted_filters))

    for filter_type in bits_filter_order:
        memory_sizes = sorted(bits_per_item_data[filter_type].keys())
        bits_values = [bits_per_item_data[filter_type][mem] for mem in memory_sizes]

        style = get_plot_style(
            filter_type,
            filter_base_types.get(filter_type, filter_type),
        )
        ax.plot(
            memory_sizes,
            bits_values,
            label=filter_display_names.get(
                filter_type,
                pu.get_filter_display_name(filter_type),
            ),
            linewidth=pu.LINE_WIDTH,
            markersize=pu.MARKER_SIZE,
            **style,  # type: ignore
        )

    ax.set_xlabel(
        "Memory Size [bytes]", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax.set_ylabel("Bits per Item", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.tick_params(axis="both", labelsize=pu.AXIS_LABEL_FONT_SIZE)
    ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    axes_box = ax.get_position()
    legend_center_x = (axes_box.x0 + axes_box.x1) / 2
    legend_y = axes_box.y1 + 0.01
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="lower center",
        bbox_to_anchor=(legend_center_x, legend_y),
        ncol=max(1, min(4, len(labels))),
        framealpha=pu.LEGEND_FRAME_ALPHA,
    )

    output_file = output_dir / "bits_per_item_vs_memory.pdf"
    plt.savefig(
        output_file,
        bbox_inches="tight",
        transparent=True,
        format="pdf",
        dpi=600,
    )
    typer.secho(
        f"Bits per item plot saved to {output_file.absolute()}",
        fg=typer.colors.GREEN,
    )
    plt.close()

    typer.secho("\nAll plots generated successfully!", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
