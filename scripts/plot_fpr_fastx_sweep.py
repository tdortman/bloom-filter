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

app = typer.Typer(help="Plot FPR FASTX sweep benchmark results")

SUPERBLOOM_CONFIG_FIXTURE_PATTERN = re.compile(
    r"^SuperBloom_K(?P<k>\d+)_S(?P<s>\d+)_M(?P<m>\d+)_H(?P<h>\d+)_FprFastxFixture$",
    re.IGNORECASE,
)
SUPERBLOOM_CPU_FIXTURE_PATTERN = re.compile(
    r"^SuperBloomCpuFprFastxFixture$", re.IGNORECASE
)
CUCO_FIXTURE_PATTERN = re.compile(r"^CucoBloomFprFastxFixture$", re.IGNORECASE)

SUPERBLOOM_VARIANT_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*", "p", "H"]
SUPERBLOOM_VARIANT_LINESTYLES = ["-", "--", "-.", ":"]


def extract_filter_series(name: str) -> Optional[tuple[str, str, str]]:
    """Extract filter series key, base style key, and display label."""
    stripped_name = str(name).strip('"')
    parts = stripped_name.split("/")

    if len(parts) >= 2:
        fixture_name = parts[0]
        operation = parts[1]

        config_match = SUPERBLOOM_CONFIG_FIXTURE_PATTERN.match(fixture_name)
        if config_match is not None:
            if operation.upper() != "FPR":
                return None
            s = int(config_match.group("s"))
            series_key = f"superbloom_s{s}"
            display_name = f"{pu.get_filter_display_name('superbloom')} (s={s})"
            return series_key, "superbloom", display_name

        if CUCO_FIXTURE_PATTERN.match(fixture_name) is not None:
            if operation.upper() != "FPR":
                return None
            return (
                "cucobloom",
                "cucobloom",
                pu.get_filter_display_name("cucobloom"),
            )

        if SUPERBLOOM_CPU_FIXTURE_PATTERN.match(fixture_name) is not None:
            if operation.upper() != "FPR":
                return None
            return (
                "superbloom_cpu",
                "superbloom_cpu",
                f"{pu.get_filter_display_name('superbloom_cpu')} (s=27)",
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

    if base_filter == "superbloom" and filter_type.startswith("superbloom_s"):
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
    Parse FPR FASTX sweep benchmark CSV results and generate plots.
    """
    if not csv_file.exists():
        typer.secho(f"CSV file not found: {csv_file}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.secho(f"Reading CSV from: {csv_file}", fg=typer.colors.CYAN)

    df = pd.read_csv(csv_file)

    # Dictionary structure: filter_type -> {filter_bits: false_positives}
    fp_data = defaultdict(dict)
    filter_display_names: dict[str, str] = {}
    filter_base_types: dict[str, str] = {}

    for _, row in df.iterrows():
        name = row["name"]
        filter_series = extract_filter_series(name)

        if filter_series is None:
            continue

        filter_type, base_filter_type, display_name = filter_series
        filter_display_names[filter_type] = display_name
        filter_base_types[filter_type] = base_filter_type

        filter_bits = row.get("filter_bits")
        false_positives = row.get("false_positives")

        if pd.notna(filter_bits) and pd.notna(false_positives):
            fp_data[filter_type][filter_bits] = false_positives

    if not fp_data:
        typer.secho("No false-positive data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    # Sort filters: Cuco first, then SuperBloom by descending s
    def sort_key(filter_type: str) -> tuple:
        if filter_type == "cucobloom":
            return (0, "")
        if filter_type == "superbloom_cpu":
            return (1, "")
        match = re.search(r"_s(\d+)$", filter_type)
        s = int(match.group(1)) if match else 0
        return (2, -s)

    sorted_filters = sorted(fp_data.keys(), key=sort_key)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    for filter_type in sorted_filters:
        filter_bits_list = sorted(fp_data[filter_type].keys())
        fp_values = [fp_data[filter_type][fb] for fb in filter_bits_list]

        style = get_plot_style(
            filter_type,
            filter_base_types.get(filter_type, filter_type),
        )
        ax.plot(
            filter_bits_list,
            fp_values,
            label=filter_display_names.get(
                filter_type,
                pu.get_filter_display_name(filter_type),
            ),
            linewidth=pu.LINE_WIDTH,
            markersize=pu.MARKER_SIZE,
            **style,  # type: ignore
        )

    ax.set_xlabel(
        "Filter Size [bits]", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax.set_ylabel(
        "False Positives", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
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

    output_file = output_dir / "fpr_fastx_sweep.pdf"
    plt.savefig(
        output_file,
        bbox_inches="tight",
        transparent=True,
        format="pdf",
        dpi=600,
    )
    typer.secho(
        f"FPR FASTX sweep plot saved to {output_file.absolute()}",
        fg=typer.colors.GREEN,
    )
    plt.close()

    typer.secho("\nPlot generated successfully!", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
