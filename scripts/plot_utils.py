"""Shared plotting utilities for benchmark visualization scripts.

This module provides common constants, styles, and helper functions used across
multiple plotting scripts to reduce code duplication and ensure visual consistency.
"""

import io
import math
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "pdf.use14corefonts": True,
        "ps.useafm": True,
    }
)


def paper_text(text: str, *, bold: bool = False) -> str:
    """Format plain text labels for LaTeX-backed figure output."""
    escaped = (
        text.replace("\\", r"\textbackslash{}").replace("%", r"\%").replace("_", r"\_")
    )
    if bold:
        return rf"\textbf{{{escaped}}}"
    return escaped


FILTER_STYLES = {
    "cusbf": {"color": "#2E86AB", "marker": "o"},
    "superbloom_cpu": {"color": "#7B4397", "marker": "D"},
    "cucobloom": {"color": "#A23B72", "marker": "s"},
    "cuckoogpu": {"color": "#F18F01", "marker": "^"},
    "cuckoogpu_dedup": {"color": "#FFB347", "marker": "^"},
    "gqf": {"color": "#6A994E", "marker": "v"},
    "tcf": {"color": "#BC4749", "marker": "p"},
    "proteincusbf": {"color": "#4FA3C7", "marker": "o"},
    "proteincucobloom": {"color": "#C15A8E", "marker": "s"},
    "proteincuckoogpu": {"color": "#F5A623", "marker": "^"},
    "proteingqf": {"color": "#8CB369", "marker": "v"},
    "proteintcf": {"color": "#D66868", "marker": "p"},
}

FILTER_COLORS = {
    "cusbf": FILTER_STYLES["cusbf"]["color"],
    "superbloom_cpu": FILTER_STYLES["superbloom_cpu"]["color"],
    "cucobloom": FILTER_STYLES["cucobloom"]["color"],
    "cuckoogpu": FILTER_STYLES["cuckoogpu"]["color"],
    "cuckoogpu_dedup": FILTER_STYLES["cuckoogpu_dedup"]["color"],
    "gqf": FILTER_STYLES["gqf"]["color"],
    "tcf": FILTER_STYLES["tcf"]["color"],
    "proteincusbf": FILTER_STYLES["proteincusbf"]["color"],
    "proteincucobloom": FILTER_STYLES["proteincucobloom"]["color"],
    "proteincuckoogpu": FILTER_STYLES["proteincuckoogpu"]["color"],
    "proteingqf": FILTER_STYLES["proteingqf"]["color"],
    "proteintcf": FILTER_STYLES["proteintcf"]["color"],
}

FILTER_DISPLAY_NAMES = {
    "cusbf": "cuSBF",
    "superbloom_cpu": "Super Bloom",
    "cucobloom": "GBBF",
    "cuckoogpu": "Cuckoo-GPU",
    "cuckoogpu_dedup": "Cuckoo-GPU (dedup.)",
    "gqf": "GQF",
    "tcf": "TCF",
    "proteincusbf": "cuSBF (Protein)",
    "proteincucobloom": "GBBF (Protein)",
    "proteincuckoogpu": "Cuckoo-GPU (Protein)",
    "proteingqf": "GQF (Protein)",
    "proteintcf": "TCF (Protein)",
}

OPERATION_COLORS = {
    "Insert": FILTER_COLORS["cusbf"],
    "Query": FILTER_COLORS["cucobloom"],
}


DEFAULT_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 18
LEGEND_FONT_SIZE = 16
BAR_FONT_SIZE = 11
LINE_WIDTH = 2
MARKER_SIZE = 8
GRID_ALPHA = 0.3
BAR_EDGE_WIDTH = 0.5
REFERENCE_LINE_WIDTH = 1.5
LEGEND_FRAME_ALPHA = 0
LEGEND_FRAME_ALPHA_SOLID = 0.9
HATCHED_BAR_ALPHA = 0.7
SCALING_BAR_ALPHA = 0.8
# Benchmark items are k-mers; items_per_second / 1e9 gives GKmer/s.
THROUGHPUT_SCALE = 1_000_000_000
THROUGHPUT_LABEL = "Throughput [GKmer/s]"


def get_filter_display_name(filter_type: str) -> str:
    """Get human-readable display name for a filter type.

    Args:
        filter_type: Internal filter identifier (e.g., 'gcf', 'bbf')

    Returns:
        Display name (e.g., 'Cuckoo-GPU', 'GBBF')
    """
    normalized = filter_type.lower()
    return FILTER_DISPLAY_NAMES.get(normalized, filter_type.capitalize())


def format_power_of_two(n: int) -> str:
    """Format a number as a LaTeX power of 2 for use in plot titles.

    Args:
        n: Number to format (should be a power of 2)

    Returns:
        LaTeX string like '$\\left(n=2^{20}\\right)$'
    """
    if n <= 0:
        return ""
    power = int(math.log2(n))
    return rf"$\left(n=2^{{{power}}}\right)$"


def format_capacity_title(base_title: str, capacity: int | None) -> str:
    """Format a title with capacity as power of 2.

    Args:
        base_title: Base title string
        capacity: Capacity value (power of 2)

    Returns:
        Title with capacity appended, e.g., 'Insert Throughput (n=2^20)'
    """
    if capacity is not None and capacity > 0:
        return f"{base_title} {format_power_of_two(capacity)}"
    return base_title


def to_gkmers_per_sec(items_per_second: float) -> float:
    """Convert benchmark items_per_second (k-mers/s) to GKmer/s."""
    return float(items_per_second) / THROUGHPUT_SCALE


def to_billion_elems_per_sec(items_per_second: float) -> float:
    """Deprecated alias for :func:`to_gkmers_per_sec`."""
    return to_gkmers_per_sec(items_per_second)


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load a CSV file with consistent error handling.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Loaded DataFrame

    Raises:
        typer.Exit: If CSV cannot be read
    """

    try:
        # Benchmark CSVs may include extra user counters (e.g. SuperBloom "s")
        # beyond the header columns; read only declared columns.
        if csv_path == "-":
            data = sys.stdin.read()
            buffer = io.StringIO(data)
            header_df = pd.read_csv(buffer, nrows=0)
            buffer.seek(0)
            return pd.read_csv(buffer, usecols=list(header_df.columns))
        header_df = pd.read_csv(csv_path, nrows=0)
        return pd.read_csv(csv_path, usecols=list(header_df.columns))
    except Exception as e:
        typer.secho(f"Error reading CSV {csv_path}: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


def resolve_output_dir(output_dir: Path | None, script_path: Path) -> Path:
    """Resolve and create the output directory.

    Args:
        output_dir: User-specified output directory, or None for default
        script_path: Path to the calling script (typically __file__)

    Returns:
        Resolved output directory path (created if it doesn't exist)
    """
    if output_dir is None:
        script_dir = Path(script_path).parent
        output_dir = script_dir.parent / "build"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def format_axis(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    xscale: str | None = "log",
    yscale: str | None = None,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    grid: bool = True,
) -> None:
    """Apply consistent formatting to a matplotlib axis.

    Args:
        ax: Matplotlib axis to format
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Optional title for the axis
        xscale: X-axis scale ('log', 'linear', or None to skip)
        yscale: Y-axis scale ('log', 'linear', or None to skip)
        xlim: Optional (min, max) tuple for x-axis limits
        ylim: Optional (min, max) tuple for y-axis limits
        grid: Whether to show grid lines
    """
    ax.set_xlabel(paper_text(xlabel, bold=True), fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(paper_text(ylabel, bold=True), fontsize=AXIS_LABEL_FONT_SIZE)

    if title:
        ax.set_title(paper_text(title, bold=True), fontsize=TITLE_FONT_SIZE)

    if xscale == "log":
        ax.set_xscale("log", base=2)
    elif xscale:
        ax.set_xscale(xscale)

    if yscale == "log":
        ax.set_yscale("log")
    elif yscale:
        ax.set_yscale(yscale)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if grid:
        ax.grid(True, which="both", ls="--", alpha=GRID_ALPHA)


def save_figure(
    fig_or_path,
    output_path: Path,
    message: str | None = None,
    close: bool = True,
) -> None:
    """Save a figure with consistent options and print success message.

    Args:
        fig_or_path: Figure object or None to use plt.savefig
        output_path: Path to save the figure
        message: Optional custom success message (default: 'Saved {path}')
        close: Whether to close the figure after saving
    """
    save_kwargs = {
        "bbox_inches": "tight",
        "pad_inches": 0.03,
        "transparent": True,
        "format": "pdf",
        "dpi": 600,
    }

    figure = plt.gcf() if fig_or_path is None else fig_or_path
    figure.canvas.draw()

    if fig_or_path is None:
        plt.savefig(output_path, **save_kwargs)
    else:
        fig_or_path.savefig(output_path, **save_kwargs)

    if message is None:
        message = f"Saved {output_path}"

    typer.secho(message, fg=typer.colors.GREEN)

    if close:
        if fig_or_path is None:
            plt.close()
        else:
            plt.close(fig_or_path)


def get_filter_style(filter_type: str, positive_negative: str | None = None) -> dict:
    """Get the style dictionary for a filter type.

    Args:
        filter_type: Filter identifier (e.g., 'cuckoo', 'bloom')
        positive_negative: Optional 'Positive' or 'Negative' for query styling

    Returns:
        Dictionary with color, marker, and optionally linestyle
    """
    base_style = FILTER_STYLES.get(
        filter_type.lower(), {"color": "#333333", "marker": "o"}
    )

    style = dict(base_style)

    if positive_negative == "Positive":
        style["linestyle"] = "-"
    elif positive_negative == "Negative":
        style["linestyle"] = "--"

    return style


def setup_figure(
    figsize: tuple[int, int] = (12, 8),
    title: str | None = None,
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """Create a figure with consistent styling.

    Args:
        figsize: Figure size tuple
        title: Optional figure super title
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        sharex: Share X axis
        sharey: Share Y axis

    Returns:
        Tuple of (Figure, Axes/Array of Axes)
    """
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey
    )
    if title:
        fig.suptitle(paper_text(title, bold=True), fontsize=TITLE_FONT_SIZE)
    return fig, axes


def create_legend(ax: plt.Axes, **kwargs):
    """Create a legend with consistent default styling.

    Args:
        ax: Matplotlib axis to add legend to
        **kwargs: Override default legend parameters

    Returns:
        The created Legend object

    Example:
        create_legend(ax, loc="upper right", ncol=2)
    """
    defaults = {
        "fontsize": LEGEND_FONT_SIZE,
        "loc": "best",
        "framealpha": 0,
    }
    defaults.update(kwargs)
    return ax.legend(**defaults)


def normalize_benchmark_name(name: str) -> str:
    """Convert benchmark name to standardized format.

    Handles both new format (GPUCuckoo_5/Insert/...) and old format (CF_5/Insert/...).

    Args:
        name: Benchmark name like "GPUCuckoo_5/Insert/268435456/min_time:0.500/..."

    Returns:
        Standardized name like "gpucuckoo" or "blockedbloom"

    Examples:
        GPUCuckoo_5/Insert/... → "gpucuckoo"
        BlockedBloom_10/Query/... → "blockedbloom"
        CF_5/Insert/... → "cf" (legacy)
    """
    # Extract the filter prefix before the first underscore
    parts = name.split("/")
    if parts:
        # Split on underscore to get filter name: "GPUCuckoo_5" → "GPUCuckoo"
        base = parts[0].split("_")[0]
        normalized = base.lower()
        if normalized.endswith("fixture"):
            normalized = normalized.removesuffix("fixture")
        return normalized

    return name.lower()


def parse_fixture_benchmark_name(name: str) -> tuple[str, str, int] | None:
    """Parse benchmark names in fixture format.

    Supported formats:

    - Memory sweep: ``<FixtureName>/<Operation>/<Size>/...`` (``Size`` is digits)
    - FASTX throughput/FPR: ``<FixtureName>/<Operation>/iterations:.../...``
      (no numeric size segment; size is returned as ``0``)

    Examples:
        ``GCFFixture/Insert/65536/...``
        ``CucoBloomFixture/Insert/iterations:10/repeats:5/manual_time_median``

    Args:
        name: Raw benchmark name from Google Benchmark CSV.

    Returns:
        Tuple ``(filter_key, operation, size)`` on success, else ``None``.
        ``filter_key`` is normalized to lowercase and stripped of a trailing
        ``Fixture`` suffix and optional numeric postfix.
    """
    parts = name.split("/")
    if len(parts) < 2:
        return None

    first_part = parts[0].strip('"')
    operation = parts[1]

    fixture_match = re.match(r"^(?P<base>.+?)Fixture\d*$", first_part)
    if fixture_match is None:
        return None

    size = 0
    if len(parts) >= 3 and parts[2].isdigit():
        size = int(parts[2])

    return fixture_match.group("base").lower(), operation, size


def clustered_bar_chart(
    ax: plt.Axes,
    categories: list[str],
    groups: list[str],
    data: dict[str, dict[str, float]],
    colors: dict[str, str],
    bar_width: float = 0.25,
    group_stride: float | None = None,
    category_stride: float = 1.0,
    show_values: bool = True,
    value_decimals: int = 0,
    value_fontsize: float | None = None,
    hatches: dict[str, str] | None = None,
    alphas: dict[str, float] | None = None,
    labels: dict[str, str] | None = None,
    series: list[str] | None = None,
    series_data: dict[str, dict[str, dict[str, float]]] | None = None,
    series_styles: dict[str, dict[str, object]] | None = None,
) -> None:
    """Create a clustered bar chart.

    Args:
        ax: Matplotlib axis to plot on
        categories: List of category labels (x-axis, e.g., operations)
        groups: List of group labels (bar clusters, e.g., policies)
        data: Nested dict {group: {category: value}}
        colors: Dict mapping group names to colors
        bar_width: Width of each bar
        group_stride: Distance between adjacent groups inside each category
        category_stride: Distance between category clusters on the x-axis (default 1.0)
        show_values: Whether to show values on top of bars
        value_decimals: Number of decimal places for bar value labels
        value_fontsize: Font size for bar value labels (default: BAR_FONT_SIZE)
        hatches: Optional dict mapping group names to hatch patterns
        alphas: Optional dict mapping group names to alpha values
        labels: Optional dict mapping group names to display labels for legend
        series: Optional ordered list of series keys when plotting multiple
            bars per group/category (e.g. ["large", "small"])
        series_data: Optional nested dict
            {series: {group: {category: value}}}
        series_styles: Optional dict mapping series keys to style overrides.
            Supported keys: offset, alpha, hatch, edgecolor, linewidth,
            linestyle, zorder.
    """
    n_groups = len(groups)
    x_positions = [i * category_stride for i in range(len(categories))]
    cluster_stride = bar_width if group_stride is None else group_stride
    active_series = (
        series
        if series is not None
        else list(series_data.keys())
        if series_data is not None
        else []
    )

    bar_label_fontsize = BAR_FONT_SIZE if value_fontsize is None else value_fontsize

    log_y = ax.get_yscale() == "log"

    def annotate_bars(bars, values):
        if not show_values:
            return
        for bar, val in zip(bars, values):
            if val > 0:
                y = bar.get_height()
                if log_y:
                    y *= 1.04
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    f"{val:.{value_decimals}f}",
                    ha="center",
                    va="bottom",
                    fontsize=bar_label_fontsize,
                    fontweight="bold",
                    clip_on=False,
                )

    for i, group in enumerate(groups):
        offset = (i - (n_groups - 1) / 2) * cluster_stride
        group_hatch = hatches.get(group) if hatches else None
        group_alpha = alphas.get(group, 1.0) if alphas else 1.0
        group_label = labels.get(group, group) if labels else group
        group_color = colors.get(group, "#333333")

        if series_data is None:
            values = [data.get(group, {}).get(cat, 0) for cat in categories]
            bars = ax.bar(
                [x + offset for x in x_positions],
                values,
                bar_width,
                label=group_label,
                color=group_color,
                edgecolor="black",
                linewidth=0.5,
                hatch=group_hatch,
                alpha=group_alpha,
            )
            annotate_bars(bars, values)
            continue

        for series_idx, series_name in enumerate(active_series):
            values = [
                series_data.get(series_name, {}).get(group, {}).get(cat, 0)
                for cat in categories
            ]
            style = series_styles.get(series_name, {}) if series_styles else {}
            series_offset = float(style.get("offset", 0.0))  # type: ignore
            series_alpha = style.get("alpha")
            series_hatch = style.get("hatch")
            edgecolor = str(style.get("edgecolor", "black"))
            linewidth = float(style.get("linewidth", 0.5))  # type: ignore
            linestyle = style.get("linestyle")
            zorder = style.get("zorder")
            series_label = group_label if series_idx == 0 else "_nolegend_"

            bar_kwargs: dict[str, object] = {
                "label": series_label,
                "color": group_color,
                "edgecolor": edgecolor,
                "linewidth": linewidth,
                "hatch": group_hatch if series_hatch is None else series_hatch,
                "alpha": group_alpha if series_alpha is None else float(series_alpha),  # type: ignore
            }
            if linestyle is not None:
                bar_kwargs["linestyle"] = linestyle
            if zorder is not None:
                bar_kwargs["zorder"] = zorder

            bars = ax.bar(
                [x + offset + series_offset for x in x_positions],
                values,
                bar_width,
                **bar_kwargs,  # type: ignore
            )
            annotate_bars(bars, values)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, fontsize=DEFAULT_FONT_SIZE)
