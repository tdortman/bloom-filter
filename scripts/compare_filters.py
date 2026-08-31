#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
"""Clustered-bar Insert/Query throughput (GKmer/s).

Small workload: C. elegans reference genome (left subplot).
Large workload: human T2T-CHM13 reference genome (right subplot).

Each subplot combines three memory-system CSVs (GH200 (HBM3) + RTX PRO 6000 (GDDR7)
paired bars; Super Bloom CPU from W9-3595X (DDR5) as a single dashed-edge series).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer
from matplotlib.axes import Axes
from matplotlib.patches import Patch

app = typer.Typer(
    help="Plot gpu-filter-comparison Insert/Query throughput"
)

# Left-to-right bar order; short legend keys.
FILTER_GROUP_ORDER: list[str] = [
    "cusbf",
    "cucobloom",
    "cuckoogpu",
    "cuckoogpu_dedup",
    "gqf",
    "tcf",
    "superbloom_cpu",
]

FILTER_LEGEND_LABELS: dict[str, str] = {
    "cusbf": "cuSBF",
    "cuckoogpu": "Cuckoo-GPU",
    "cuckoogpu_dedup": "Cuckoo-GPU (dedup.)",
    "tcf": "TCF",
    "gqf": "GQF",
    "cucobloom": "GBBF",
    "superbloom_cpu": "Super Bloom",
}

# (label, hatch_pattern, alpha)
BAR_OPERATIONS = [
    ("Insert", "//", 1.0),
    ("Query", None, 1.0),
]

_PAIRED_BAR_WIDTH = 0.145
_PAIRED_OP_STRIDE = 0.30
_PAIRED_MEM_OFFSET = 0.075
_PAIRED_FILTER_SPACING = 1.36
_GDDR7_PAIRED_ALPHA_SCALE = 0.55
_GDDR7_PAIRED_ALPHA_FLOOR = 0.35
_DDR5_ALPHA = 0.75
_DDR5_EDGE_COLOR = "#1F2937"
_HBM3_LEGEND_LABEL = "GH200 (HBM3)"
_GDDR7_LEGEND_LABEL = "RTX PRO 6000 (GDDR7)"
_DDR5_LEGEND_LABEL = "W9-3595X (DDR5)"
_X_AXIS_MARGIN_LEFT = _PAIRED_BAR_WIDTH
_X_AXIS_MARGIN_RIGHT = _PAIRED_BAR_WIDTH
_SUBPLOT_FIGSIZE = (7.2, 3.0)
_SUBPLOT_LEFT_MARGIN = 0.16
_SUBPLOT_RIGHT_MARGIN = 0.99
_YLIM_TOP_FACTOR = 1.45
_OUTPUT_BASENAME = "benchmark_throughput_comparison"
_DDR5_CPU_FILTER = "superbloom_cpu"
_SUBPLOT_BOTTOM_MARGIN = 0.06
_SUBPLOT_TOP_MARGIN = 0.94


def normalize_filter_key(fixture_base: str) -> str:
    """Map fixture base names to canonical filter keys."""
    key = fixture_base.lower()
    if key.startswith("cuckoogpudeduplicated"):
        return "cuckoogpu_dedup"
    if key.startswith("cuckoogpu"):
        return "cuckoogpu"
    if key.startswith("cucobloom"):
        return "cucobloom"
    if key == "gqf" or key.startswith("gqffixture"):
        return "gqf"
    if key == "tcf" or key.startswith("tcffixture"):
        return "tcf"
    if key.startswith("cusbf"):
        return "cusbf"
    if key.startswith("superbloom"):
        return "superbloom_cpu"
    return key


def sort_groups(
    throughput_data: dict[str, dict[str, float]],
    preferred_order: list[str],
) -> list[str]:
    ordered = [g for g in preferred_order if g in throughput_data]
    extras = sorted(
        set(throughput_data.keys()) - set(ordered),
        key=lambda g: -max(throughput_data[g].values(), default=0.0),
    )
    return ordered + extras


def load_throughput_data(csv_path: Path) -> dict[str, dict[str, float]]:
    """Load Insert/Query throughput [GKmer/s] from a gpu-filter-comparison CSV."""
    df = pu.load_csv(csv_path)
    df = df[df["name"].str.endswith("_median", na=False)]

    throughput_data: dict[str, dict[str, float]] = defaultdict(dict)

    for _, row in df.iterrows():
        parsed = pu.parse_fixture_benchmark_name(row["name"])
        if parsed is None:
            continue

        fixture_base, operation, _size = parsed
        if operation not in {"Insert", "Query"}:
            continue

        items_per_second = row.get("items_per_second")
        if pd.isna(items_per_second):
            continue

        filter_key = normalize_filter_key(fixture_base)
        throughput_data[filter_key][operation] = pu.to_gkmers_per_sec(items_per_second)

    return dict(throughput_data)


def apply_ddr5_override(
    primary_data: dict[str, dict[str, float]],
    secondary_data: dict[str, dict[str, float]],
    ddr5_data: dict[str, dict[str, float]],
    filter_name: str,
) -> bool:
    """Replace one filter in *primary_data* with DDR5 results; drop from *secondary_data*."""
    override_values = ddr5_data.get(filter_name)
    if not override_values:
        return False
    primary_data[filter_name] = override_values
    secondary_data.pop(filter_name, None)
    return True


def expand_throughput_ylim(ylim: tuple[float, float]) -> tuple[float, float]:
    ymin, ymax = ylim
    if ymax <= 0:
        return ylim
    return ymin, ymax * _YLIM_TOP_FACTOR


def relabel_chart_data(
    throughput_data: dict[str, dict[str, float]],
    groups: list[str],
) -> dict[str, dict[str, float]]:
    categories = [op for op, _, _ in BAR_OPERATIONS]
    return {
        group: {cat: throughput_data.get(group, {}).get(cat, 0.0) for cat in categories}
        for group in groups
    }


def plot_bar_on_axis(
    ax: Axes,
    data: dict[str, dict[str, float]],
    filter_order: list[str],
    show_ylabel: bool,
    bg_data: dict[str, dict[str, float]] | None = None,
    single_source_filters: set[str] | None = None,
    ddr5_filters: set[str] | None = None,
) -> list[Patch]:
    """Plot clustered bars; return legend patches (filters + operations + memory)."""
    n_ops = len(BAR_OPERATIONS)
    has_bg = bg_data is not None and len(bg_data) > 0
    single_source_filters = single_source_filters or set()
    ddr5_filters = ddr5_filters or set()
    has_ddr5 = bool(ddr5_filters)

    if has_bg:
        bg_data_asserted = bg_data or {}
        for filter_idx, filter_name in enumerate(filter_order):
            filter_center = filter_idx * _PAIRED_FILTER_SPACING
            hbm_data = data.get(filter_name, {})
            gddr_data = bg_data_asserted.get(filter_name, {})
            single_source = filter_name in single_source_filters
            ddr5_source = filter_name in ddr5_filters

            for op_idx, (op_label, hatch, alpha) in enumerate(BAR_OPERATIONS):
                cluster_center = (
                    filter_center + (op_idx - (n_ops - 1) / 2) * _PAIRED_OP_STRIDE
                )
                hbm_tp = hbm_data.get(op_label, 0)
                gddr_tp = gddr_data.get(op_label, 0)

                if hbm_tp > 0:
                    hbm_x = (
                        cluster_center
                        if single_source
                        else cluster_center - _PAIRED_MEM_OFFSET
                    )
                    hbm_alpha = _DDR5_ALPHA if ddr5_source else alpha
                    hbm_edgecolor = _DDR5_EDGE_COLOR if ddr5_source else "black"
                    hbm_linestyle = "--" if ddr5_source else "-"
                    ax.bar(
                        hbm_x,
                        hbm_tp,
                        _PAIRED_BAR_WIDTH,
                        color=pu.FILTER_COLORS.get(filter_name, "#333333"),
                        edgecolor=hbm_edgecolor,
                        linewidth=pu.BAR_EDGE_WIDTH,
                        hatch=hatch,
                        linestyle=hbm_linestyle,
                        alpha=hbm_alpha,
                        zorder=3,
                    )

                if gddr_tp > 0 and not single_source:
                    gddr_alpha = max(
                        _GDDR7_PAIRED_ALPHA_FLOOR, alpha * _GDDR7_PAIRED_ALPHA_SCALE
                    )
                    ax.bar(
                        cluster_center + _PAIRED_MEM_OFFSET,
                        gddr_tp,
                        _PAIRED_BAR_WIDTH,
                        color=pu.FILTER_COLORS.get(filter_name, "#333333"),
                        edgecolor="#666666",
                        linewidth=pu.BAR_EDGE_WIDTH,
                        hatch=hatch,
                        alpha=gddr_alpha,
                        zorder=2,
                    )
    else:
        for filter_idx, filter_name in enumerate(filter_order):
            filter_data = data.get(filter_name, {})
            ddr5_source = filter_name in ddr5_filters
            for op_idx, (op_label, hatch, alpha) in enumerate(BAR_OPERATIONS):
                tp = filter_data.get(op_label, 0)
                if tp <= 0:
                    continue
                x = filter_idx + (op_idx - (n_ops - 1) / 2) * _PAIRED_OP_STRIDE
                fg_alpha = _DDR5_ALPHA if ddr5_source else alpha
                fg_edgecolor = _DDR5_EDGE_COLOR if ddr5_source else "black"
                fg_linestyle = "--" if ddr5_source else "-"
                ax.bar(
                    x,
                    tp,
                    _PAIRED_BAR_WIDTH,
                    color=pu.FILTER_COLORS.get(filter_name, "#333333"),
                    edgecolor=fg_edgecolor,
                    linewidth=pu.BAR_EDGE_WIDTH,
                    hatch=hatch,
                    linestyle=fg_linestyle,
                    alpha=fg_alpha,
                    zorder=3,
                )

    pu.format_axis(
        ax,
        "",
        pu.THROUGHPUT_LABEL if show_ylabel else "",
        xscale=None,
        yscale="log",
    )
    ylabel = pu.paper_text(pu.THROUGHPUT_LABEL, bold=True)
    if show_ylabel:
        ax.set_ylabel(
            ylabel,
            fontsize=pu.AXIS_LABEL_FONT_SIZE,
            labelpad=10,
        )
    else:
        # Keep both cropped subplot PDFs the same size under bbox_inches="tight".
        # A transparent label still contributes to the bounding box without showing.
        ax.set_ylabel(
            ylabel,
            fontsize=pu.AXIS_LABEL_FONT_SIZE,
            labelpad=10,
            color=(0, 0, 0, 0),
        )
    ax.set_xticks([])
    if filter_order:
        if has_bg:
            pair_half_span = (
                ((n_ops - 1) / 2) * _PAIRED_OP_STRIDE
                + _PAIRED_MEM_OFFSET
                + (_PAIRED_BAR_WIDTH / 2)
            )
            x_min = -pair_half_span - _X_AXIS_MARGIN_LEFT
            x_max = (
                (len(filter_order) - 1) * _PAIRED_FILTER_SPACING
                + pair_half_span
                + _X_AXIS_MARGIN_RIGHT
            )
        else:
            pair_half_span = ((n_ops - 1) / 2) * _PAIRED_OP_STRIDE + (
                _PAIRED_BAR_WIDTH / 2
            )
            x_min = -pair_half_span - _X_AXIS_MARGIN_LEFT
            x_max = (
                (len(filter_order) - 1) * _PAIRED_FILTER_SPACING
                + pair_half_span
                + _X_AXIS_MARGIN_RIGHT
            )
        ax.set_xlim(x_min, x_max)

    legend_elements: list[Patch] = [
        Patch(
            facecolor=pu.FILTER_COLORS.get(name, "#333333"),
            edgecolor="black",
            linewidth=pu.BAR_EDGE_WIDTH,
            label=FILTER_LEGEND_LABELS.get(name, name),
        )
        for name in filter_order
    ]
    for op_label, hatch, alpha in BAR_OPERATIONS:
        legend_elements.append(
            Patch(
                facecolor="gray",
                edgecolor="black",
                linewidth=pu.BAR_EDGE_WIDTH,
                hatch=hatch,
                alpha=alpha,
                label=op_label,
            )
        )

    if has_bg or has_ddr5:
        legend_elements.append(
            Patch(
                facecolor="gray",
                edgecolor="black",
                linewidth=pu.BAR_EDGE_WIDTH,
                alpha=1.0,
                label=_HBM3_LEGEND_LABEL,
            )
        )
        if has_bg:
            gddr_alpha = max(_GDDR7_PAIRED_ALPHA_FLOOR, _GDDR7_PAIRED_ALPHA_SCALE)
            legend_elements.append(
                Patch(
                    facecolor="gray",
                    edgecolor="#666666",
                    linewidth=pu.BAR_EDGE_WIDTH,
                    alpha=gddr_alpha,
                    label=_GDDR7_LEGEND_LABEL,
                )
            )
        if has_ddr5:
            legend_elements.append(
                Patch(
                    facecolor="gray",
                    edgecolor=_DDR5_EDGE_COLOR,
                    linewidth=pu.BAR_EDGE_WIDTH,
                    linestyle="--",
                    alpha=_DDR5_ALPHA,
                    label=_DDR5_LEGEND_LABEL,
                )
            )

    return legend_elements


def save_bar_subplot(
    output_path: Path,
    chart_data: dict[str, dict[str, float]],
    filter_order: list[str],
    show_ylabel: bool,
    bg_data: dict[str, dict[str, float]] | None,
    single_source_filters: set[str],
    ddr5_filters: set[str],
    ylim: tuple[float, float],
    message: str,
) -> list[Patch]:
    fig, ax = plt.subplots(1, 1, figsize=_SUBPLOT_FIGSIZE)
    legend_patches = plot_bar_on_axis(
        ax,
        chart_data,
        filter_order,
        show_ylabel=show_ylabel,
        bg_data=bg_data,
        single_source_filters=single_source_filters,
        ddr5_filters=ddr5_filters,
    )
    ax.set_ylim(*expand_throughput_ylim(ylim))
    fig.subplots_adjust(
        left=_SUBPLOT_LEFT_MARGIN,
        right=_SUBPLOT_RIGHT_MARGIN,
        bottom=_SUBPLOT_BOTTOM_MARGIN,
        top=_SUBPLOT_TOP_MARGIN,
    )
    pu.save_figure(fig, output_path, message)
    return legend_patches


def save_bar_legend_figure(
    output_path: Path,
    filter_handles: list[Patch],
    op_handles: list[Patch],
    mem_handles: list[Patch],
    fig_width: float,
    message: str,
) -> None:
    n_rows = 2 + int(bool(mem_handles))
    legend_fig_height = 0.55 + 0.42 * n_rows
    fig = plt.figure(figsize=(fig_width, legend_fig_height))

    legend_y_top = 0.9
    legend_row_step = 0.22

    if filter_handles:
        fig.legend(
            handles=filter_handles,
            fontsize=pu.LEGEND_FONT_SIZE,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y_top),
            ncol=len(filter_handles),
            framealpha=pu.LEGEND_FRAME_ALPHA,
        )
    if mem_handles:
        fig.legend(
            handles=mem_handles,
            fontsize=pu.LEGEND_FONT_SIZE,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y_top - legend_row_step),
            ncol=len(mem_handles),
            framealpha=pu.LEGEND_FRAME_ALPHA,
        )
    if op_handles:
        op_legend_y = legend_y_top - (2 * legend_row_step if mem_handles else legend_row_step)
        fig.legend(
            handles=op_handles,
            fontsize=pu.LEGEND_FONT_SIZE,
            loc="upper center",
            bbox_to_anchor=(0.5, op_legend_y),
            ncol=len(op_handles),
            framealpha=pu.LEGEND_FRAME_ALPHA,
        )

    pu.save_figure(fig, output_path, message)


def split_bar_legend_elements(
    legend_elements: list[Patch], filter_order: list[str]
) -> tuple[list[Patch], list[Patch], list[Patch]]:
    n_filters = len(filter_order)
    n_ops = len(BAR_OPERATIONS)
    filter_handles = legend_elements[:n_filters]
    op_handles = legend_elements[n_filters : n_filters + n_ops]
    mem_handles = legend_elements[n_filters + n_ops :]
    return filter_handles, op_handles, mem_handles


def build_memory_legend_handles(has_bg: bool, has_ddr5: bool) -> list[Patch]:
    mem_handles: list[Patch] = []
    if has_bg or has_ddr5:
        mem_handles.append(
            Patch(
                facecolor="gray",
                edgecolor="black",
                linewidth=pu.BAR_EDGE_WIDTH,
                alpha=1.0,
                label=_HBM3_LEGEND_LABEL,
            )
        )
    if has_bg:
        gddr_alpha = max(_GDDR7_PAIRED_ALPHA_FLOOR, _GDDR7_PAIRED_ALPHA_SCALE)
        mem_handles.append(
            Patch(
                facecolor="gray",
                edgecolor="#666666",
                linewidth=pu.BAR_EDGE_WIDTH,
                alpha=gddr_alpha,
                label=_GDDR7_LEGEND_LABEL,
            )
        )
    if has_ddr5:
        mem_handles.append(
            Patch(
                facecolor="gray",
                edgecolor=_DDR5_EDGE_COLOR,
                linewidth=pu.BAR_EDGE_WIDTH,
                linestyle="--",
                alpha=_DDR5_ALPHA,
                label=_DDR5_LEGEND_LABEL,
            )
        )
    return mem_handles


def prepare_panel(
    hbm3_csv: Path,
    gddr7_csv: Path,
    ddr5_csv: Path,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    set[str],
    set[str],
]:
    hbm3_data = load_throughput_data(hbm3_csv)
    gddr7_data = load_throughput_data(gddr7_csv)
    ddr5_data = load_throughput_data(ddr5_csv)

    single_source: set[str] = set()
    ddr5_filters: set[str] = set()

    if apply_ddr5_override(hbm3_data, gddr7_data, ddr5_data, _DDR5_CPU_FILTER):
        single_source.add(_DDR5_CPU_FILTER)
        ddr5_filters.add(_DDR5_CPU_FILTER)
    elif _DDR5_CPU_FILTER in ddr5_data:
        typer.secho(
            f"Warning: {_DDR5_CPU_FILTER} present in DDR5 CSV but not applied to panel "
            f"({hbm3_csv.name})",
            fg=typer.colors.YELLOW,
            err=True,
        )

    return hbm3_data, gddr7_data, single_source, ddr5_filters


@app.command()
def main(
    csv_gddr7_small: Annotated[
        Path,
        typer.Argument(
            help="RTX PRO 6000 (GDDR7) throughput CSV for C. elegans (small filter)"
        ),
    ],
    csv_gddr7_large: Annotated[
        Path,
        typer.Argument(
            help="RTX PRO 6000 (GDDR7) throughput CSV for human T2T-CHM13 (large filter)"
        ),
    ],
    csv_hbm3_small: Annotated[
        Path,
        typer.Argument(
            help="GH200 (HBM3) throughput CSV for C. elegans (small filter)"
        ),
    ],
    csv_hbm3_large: Annotated[
        Path,
        typer.Argument(
            help="GH200 (HBM3) throughput CSV for human T2T-CHM13 (large filter)"
        ),
    ],
    csv_ddr5_small: Annotated[
        Path,
        typer.Argument(
            help="W9-3595X (DDR5) throughput CSV for C. elegans (Super Bloom CPU)"
        ),
    ],
    csv_ddr5_large: Annotated[
        Path,
        typer.Argument(
            help="W9-3595X (DDR5) throughput CSV for human T2T-CHM13 (Super Bloom CPU)"
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for plots (default: build/)",
        ),
    ] = None,
):
    """
    Plot Insert/Query throughput [GKmer/s].

    Combines six benchmark CSVs (RTX PRO 6000/GH200/W9-3595X × small/large) into
    two subplot PDFs plus a legend PDF.
    """
    hbm3_small, gddr7_small, single_left, ddr5_left = prepare_panel(
        csv_hbm3_small, csv_gddr7_small, csv_ddr5_small
    )
    hbm3_large, gddr7_large, single_right, ddr5_right = prepare_panel(
        csv_hbm3_large, csv_gddr7_large, csv_ddr5_large
    )

    if not hbm3_small and not hbm3_large:
        typer.secho(
            "No Insert/Query throughput data found in HBM3 CSV files",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    all_groups = set(hbm3_small.keys()) | set(hbm3_large.keys())
    all_groups |= set(gddr7_small.keys()) | set(gddr7_large.keys())
    groups = sort_groups(
        {g: {**hbm3_small.get(g, {}), **hbm3_large.get(g, {})} for g in all_groups},
        FILTER_GROUP_ORDER,
    )
    filter_order = [g for g in FILTER_GROUP_ORDER if g in groups] + [
        g for g in groups if g not in FILTER_GROUP_ORDER
    ]

    small_chart = relabel_chart_data(hbm3_small, filter_order)
    large_chart = relabel_chart_data(hbm3_large, filter_order)
    gddr7_small_chart = relabel_chart_data(gddr7_small, filter_order)
    gddr7_large_chart = relabel_chart_data(gddr7_large, filter_order)

    fig_width = max(8.0, 2.1 * len(filter_order) + 1.5)
    has_bg = bool(gddr7_small_chart) or bool(gddr7_large_chart)
    has_ddr5 = bool(ddr5_left) or bool(ddr5_right)

    preview_fig, (preview_left, preview_right) = plt.subplots(
        1,
        2,
        figsize=(fig_width, _SUBPLOT_FIGSIZE[1]),
        sharey=True,
        gridspec_kw={"wspace": 0.04},
    )
    legend_left = plot_bar_on_axis(
        preview_left,
        small_chart,
        filter_order,
        show_ylabel=True,
        bg_data=gddr7_small_chart if gddr7_small_chart else None,
        single_source_filters=single_left,
        ddr5_filters=ddr5_left,
    )
    legend_right = plot_bar_on_axis(
        preview_right,
        large_chart,
        filter_order,
        show_ylabel=False,
        bg_data=gddr7_large_chart if gddr7_large_chart else None,
        single_source_filters=single_right,
        ddr5_filters=ddr5_right,
    )

    y_min_left, y_max_left = preview_left.get_ylim()
    y_min_right, y_max_right = preview_right.get_ylim()
    common_ylim = (
        min(y_min_left, y_min_right),
        max(y_max_left, y_max_right),
    )
    plt.close(preview_fig)

    legend_elements = legend_left if len(legend_left) >= len(legend_right) else legend_right
    filter_handles, op_handles, _ = split_bar_legend_elements(
        legend_elements, filter_order
    )
    mem_handles = build_memory_legend_handles(has_bg, has_ddr5)

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))
    left_output = output_dir / f"{_OUTPUT_BASENAME}_left.pdf"
    right_output = output_dir / f"{_OUTPUT_BASENAME}_right.pdf"
    legend_output = output_dir / f"{_OUTPUT_BASENAME}_legend.pdf"

    save_bar_subplot(
        left_output,
        small_chart,
        filter_order,
        show_ylabel=True,
        bg_data=gddr7_small_chart if gddr7_small_chart else None,
        single_source_filters=single_left,
        ddr5_filters=ddr5_left,
        ylim=common_ylim,
        message=f"Left throughput subplot (small / C. elegans) saved to {left_output}",
    )
    save_bar_subplot(
        right_output,
        large_chart,
        filter_order,
        show_ylabel=False,
        bg_data=gddr7_large_chart if gddr7_large_chart else None,
        single_source_filters=single_right,
        ddr5_filters=ddr5_right,
        ylim=common_ylim,
        message=f"Right throughput subplot (large / T2T-CHM13) saved to {right_output}",
    )
    save_bar_legend_figure(
        legend_output,
        filter_handles,
        op_handles,
        mem_handles,
        fig_width=fig_width,
        message=f"Throughput legend saved to {legend_output}",
    )


if __name__ == "__main__":
    app()
