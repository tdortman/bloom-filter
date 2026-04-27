#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "seaborn",
#   "typer",
# ]
# ///

import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_utils as pu
import seaborn as sns
import typer

app = typer.Typer(help="Plot S/H parameter sweep benchmark results")

FIXTURE_PATTERN = re.compile(
    r"^SuperBloom_K(?P<k>\d+)_S(?P<s>\d+)_M(?P<m>\d+)_H(?P<h>\d+)_Fixture",
    re.IGNORECASE,
)


@app.command()
def main(
    csv_file: Path = typer.Argument(
        ...,
        help="Path to the CSV file containing benchmark results",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save output plots (default: build/)",
    ),
):
    """
    Parse SH sweep benchmark CSV results and generate heatmaps and Pareto plots.

    Examples:
        plot_sh_sweep.py results.csv
        plot_sh_sweep.py results.csv -o custom/dir
    """
    if not csv_file.exists():
        typer.secho(f"CSV file not found: {csv_file}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.secho(f"Reading CSV from: {csv_file}", fg=typer.colors.CYAN)
    df = pu.load_csv(csv_file)

    # Filter for median aggregate rows
    df = df[df["name"].str.endswith("_median")].copy()

    # Parse fixture names to extract S, H, and operation.
    # Note: the benchmark CSV already contains an ``s`` counter column, so we
    # overwrite it with the parsed integer value to avoid duplicate columns.
    extracted = df["name"].str.extract(FIXTURE_PATTERN)
    df["s"] = extracted["s"].astype(float)
    df["h"] = extracted["h"].astype(float)
    df["operation"] = df["name"].str.split("/").str[1]
    df = df.dropna(subset=["s", "h", "operation"])

    if df.empty:
        typer.secho(
            "No valid benchmark data found in CSV", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(1)

    # Ensure numeric types
    df["s"] = df["s"].astype(int)
    df["h"] = df["h"].astype(int)
    df["real_time"] = pd.to_numeric(df["real_time"], errors="coerce")
    df["fpr_percentage"] = pd.to_numeric(
        df.get("fpr_percentage", np.nan), errors="coerce"
    )

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    # 1. Heatmap grid: Insert time, Query time, FPR
    typer.secho("Generating heatmap grid...", fg=typer.colors.CYAN)

    # Dense grids: disable annotations and increase figure size so cells
    # are large enough to be readable from the color alone.
    n_s = df["s"].nunique()
    n_h = df["h"].nunique()
    is_dense = (n_s * n_h) > 25
    heatmap_figsize = (22, 10) if is_dense else (18, 6)
    annot_fontsize = max(6, min(10, int(120 / max(n_s, n_h))))

    fig, axes = plt.subplots(1, 3, figsize=heatmap_figsize, sharey=True)

    operations = ["Insert", "Query", "FPR"]
    titles = ["Insert Time", "Query Time", "FPR (%)"]
    cbar_labels = ["Time [ms]", "Time [ms]", "FPR (%)"]

    for ax, op, title, cbar_label in zip(axes, operations, titles, cbar_labels):
        if op == "FPR":
            subset = df[df["operation"] == "FPR"].copy()
            pivot = subset.pivot(index="s", columns="h", values="fpr_percentage")
            fmt = ".4f"
            cmap = "rocket_r"
        else:
            subset = df[df["operation"] == op].copy()
            pivot = subset.pivot(index="s", columns="h", values="real_time")
            fmt = ".3f"
            cmap = "rocket_r"

        if pivot.empty:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(title, fontsize=pu.TITLE_FONT_SIZE, fontweight="bold")
            continue

        # Sort axes for consistent display
        pivot = pivot.sort_index(ascending=False).sort_index(axis=1)

        heatmap_kwargs = {
            "data": pivot,
            "ax": ax,
            "fmt": fmt,
            "cmap": cmap,
            "cbar_kws": {"label": cbar_label},
            "linewidths": 0.5,
            "linecolor": "white",
        }
        if is_dense:
            heatmap_kwargs["annot_kws"] = {"size": annot_fontsize}
        heatmap_kwargs["annot"] = not is_dense

        sns.heatmap(**heatmap_kwargs)
        ax.set_title(title, fontsize=pu.TITLE_FONT_SIZE, fontweight="bold")
        ax.set_xlabel(
            "H (hash count)", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
        )
        ax.set_ylabel(
            "S (s-mer width)", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
        )

    plt.tight_layout()
    heatmap_path = output_dir / "sh_sweep_heatmaps.pdf"
    pu.save_figure(fig, heatmap_path, f"Heatmap grid saved to {heatmap_path}")

    # 2. Pareto frontier plots: Insert, Query, Total vs FPR
    typer.secho("Generating Pareto frontier plots...", fg=typer.colors.CYAN)

    # Build a wide table: one row per (s, h) with Insert/Query/FPR times
    insert_df = df[df["operation"] == "Insert"][["s", "h", "real_time"]].rename(
        columns={"real_time": "insert_time"}
    )
    query_df = df[df["operation"] == "Query"][["s", "h", "real_time"]].rename(
        columns={"real_time": "query_time"}
    )
    fpr_df = df[df["operation"] == "FPR"][
        ["s", "h", "real_time", "fpr_percentage"]
    ].rename(columns={"real_time": "fpr_time", "fpr_percentage": "fpr"})

    merged = insert_df.merge(query_df, on=["s", "h"], how="outer")
    merged = merged.merge(fpr_df, on=["s", "h"], how="outer")
    merged = merged.dropna(subset=["fpr"])
    merged["total_time"] = merged["insert_time"] + merged["query_time"]

    if merged.empty:
        typer.secho(
            "Not enough data for Pareto plots (need Insert, Query, and FPR)",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(0)

    def compute_pareto_mask(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
        """Return boolean mask for Pareto-optimal points (minimize both x and y)."""
        sorted_idx = df.sort_values([x_col, y_col]).index

        mask = pd.Series(False, index=df.index)
        best_y = float("inf")

        for idx in sorted_idx:
            y = df.at[idx, y_col]
            if y < best_y:
                mask.at[idx] = True
                best_y = y

        return mask

    pareto_configs = [
        ("insert_time", "Insert Time [ms]"),
        ("query_time", "Query Time [ms]"),
        ("total_time", "Insert + Query Time [ms]"),
    ]

    for time_col, y_label in pareto_configs:
        subset = merged.dropna(subset=[time_col, "fpr"]).copy()
        if subset.empty:
            continue

        subset["pareto"] = compute_pareto_mask(subset, "fpr", time_col)

        # Scale figure for dense grids so points are not crammed.
        pareto_figsize = (18, 12) if len(subset) > 40 else (12, 8)
        fig, ax = plt.subplots(figsize=pareto_figsize)

        # Non-Pareto points
        non_pareto = subset[~subset["pareto"]]
        ax.scatter(
            non_pareto["fpr"],
            non_pareto[time_col],
            c="#AAAAAA",
            s=40,
            alpha=0.4,
            label="Dominated",
            zorder=2,
        )

        # Pareto points
        pareto = subset[subset["pareto"]]
        ax.scatter(
            pareto["fpr"],
            pareto[time_col],
            c="#2E86AB",
            s=100,
            alpha=0.9,
            label="Pareto-optimal",
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

        # Label Pareto-optimal points in bold, and dominated points in
        # smaller gray text so the full trade-off landscape is readable.
        n_pareto = len(pareto)
        n_total = len(subset)
        offsets = [(8, 8), (-8, -8), (8, -8), (-8, 8)]

        # Dominated points — always label, but shrink font for very dense grids
        dominated_label_size = 9 if n_total <= 25 else (8 if n_total <= 60 else 7)
        for idx, (_, row) in enumerate(non_pareto.iterrows()):
            ox, oy = offsets[idx % len(offsets)]
            ax.annotate(
                f"S{int(row['s'])},H{int(row['h'])}",
                (row["fpr"], row[time_col]),
                textcoords="offset points",
                xytext=(ox, oy),
                fontsize=dominated_label_size,
                alpha=0.55,
                color="#555555",
            )

        # Pareto points — larger, bold, on top
        pareto_label_size = 11 if n_pareto <= 20 else 10
        for idx, (_, row) in enumerate(pareto.iterrows()):
            ox, oy = offsets[idx % len(offsets)]
            ax.annotate(
                f"S{int(row['s'])},H{int(row['h'])}",
                (row["fpr"], row[time_col]),
                textcoords="offset points",
                xytext=(ox, oy),
                fontsize=pareto_label_size,
                alpha=0.95,
                fontweight="bold",
                color="#1a5276",
                zorder=5,
            )

        # Draw Pareto frontier line
        pareto_sorted = pareto.sort_values("fpr")
        if len(pareto_sorted) > 1:
            ax.plot(
                pareto_sorted["fpr"],
                pareto_sorted[time_col],
                "--",
                color="#2E86AB",
                alpha=0.5,
                linewidth=1.5,
                zorder=1,
            )

        ax.set_xlabel("FPR [%]", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)
        pu.create_legend(ax, loc="upper left")

        plt.tight_layout()
        suffix = time_col.replace("_time", "")
        pareto_path = output_dir / f"sh_sweep_pareto_{suffix}.pdf"
        pu.save_figure(
            fig,
            pareto_path,
            f"Pareto plot ({suffix}) saved to {pareto_path.absolute()}",
        )

    typer.secho("\nAll plots generated successfully!", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
