#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "typer",
# ]
# ///

import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot S/M/H parameter sweep benchmark results")

FIXTURE_PATTERN = re.compile(
    r"^SuperBloom_K(?P<k>\d+)_S(?P<s>\d+)_M(?P<m>\d+)_H(?P<h>\d+)_Fixture",
    re.IGNORECASE,
)

M_CMAP = plt.colormaps["viridis"]


def parse_csv(csv_file: Path) -> pd.DataFrame:
    """Load and parse benchmark CSV, extracting S, M, H, and operation."""
    df = pu.load_csv(csv_file)
    df = df[df["name"].str.endswith("_median")].copy()

    extracted = df["name"].str.extract(FIXTURE_PATTERN)
    df["s"] = extracted["s"].astype(float)
    df["m"] = extracted["m"].astype(float)
    df["h"] = extracted["h"].astype(float)
    df["operation"] = df["name"].str.split("/").str[1]
    df = df.dropna(subset=["s", "m", "h", "operation"])

    df["s"] = df["s"].astype(int)
    df["m"] = df["m"].astype(int)
    df["h"] = df["h"].astype(int)
    df["real_time"] = pd.to_numeric(df["real_time"], errors="coerce")
    df["fpr_percentage"] = pd.to_numeric(
        df.get("fpr_percentage", np.nan), errors="coerce"
    )
    return df


def build_merged_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per (s, m, h) with Insert/Query/FPR metrics."""
    insert_df = df[df["operation"] == "Insert"][["s", "m", "h", "real_time"]].rename(
        columns={"real_time": "insert_time"}
    )
    query_df = df[df["operation"] == "Query"][["s", "m", "h", "real_time"]].rename(
        columns={"real_time": "query_time"}
    )
    fpr_df = df[df["operation"] == "FPR"][
        ["s", "m", "h", "real_time", "fpr_percentage"]
    ].rename(columns={"real_time": "fpr_time", "fpr_percentage": "fpr"})

    merged = insert_df.merge(query_df, on=["s", "m", "h"], how="outer")
    merged = merged.merge(fpr_df, on=["s", "m", "h"], how="outer")
    merged = merged.dropna(subset=["fpr"])
    merged["total_time"] = merged["insert_time"] + merged["query_time"]
    return merged


def compute_pareto_mask(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    """Boolean mask for Pareto-optimal points (minimize both x and y)."""
    sorted_idx = df.sort_values([x_col, y_col]).index
    mask = pd.Series(False, index=df.index)
    best_y = float("inf")
    for idx in sorted_idx:
        y = df.at[idx, y_col]
        if y < best_y:
            mask.at[idx] = True
            best_y = y
    return mask


@app.command()
def main(
    csv_file: Path = typer.Argument(..., help="Path to benchmark CSV results"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Output directory (default: build/)"
    ),
):
    if not csv_file.exists():
        typer.secho(f"CSV not found: {csv_file}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    df = parse_csv(csv_file)
    if df.empty:
        typer.secho("No data", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    merged = build_merged_table(df)
    if merged.empty:
        typer.secho("Not enough data for plots", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(0)

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    m_values_sorted = sorted(merged["m"].unique())
    m_colors = {
        m: M_CMAP(i / max(len(m_values_sorted) - 1, 1))
        for i, m in enumerate(m_values_sorted)
    }

    # 1. 2-D Pareto plots: FPR vs Insert / Query / Total time
    typer.secho("Generating 2-D Pareto plots...", fg=typer.colors.CYAN)

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
        n_total = len(subset)

        figsize = (18, 12) if n_total > 40 else (12, 8)
        fig, ax = plt.subplots(figsize=figsize)

        # Plot all points coloured by M
        for is_pareto in [False, True]:
            group = subset[subset["pareto"] == is_pareto]
            if group.empty:
                continue
            for m_val in m_values_sorted:
                m_group = group[group["m"] == m_val]
                if m_group.empty:
                    continue
                color = m_colors[m_val]
                if is_pareto:
                    ax.scatter(
                        m_group["fpr"],
                        m_group[time_col],
                        c=[color],
                        s=120,
                        alpha=0.95,
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=4,
                    )
                else:
                    ax.scatter(
                        m_group["fpr"],
                        m_group[time_col],
                        c=[color],
                        s=50,
                        alpha=0.4,
                        zorder=2,
                    )

        # Labels for all points
        offsets = [(10, 10), (-10, -10), (10, -10), (-10, 10)]
        dominated_label_size = 9 if n_total <= 40 else 8
        pareto_label_size = 11 if n_total <= 40 else 9

        for _, row in subset[~subset["pareto"]].iterrows():
            ox, oy = offsets[int(row["s"]) % 4]
            ax.annotate(
                f"S{int(row['s'])},H{int(row['h'])}",
                (row["fpr"], row[time_col]),
                textcoords="offset points",
                xytext=(ox, oy),
                fontsize=dominated_label_size,
                alpha=0.5,
                color="#555555",
            )

        for _, row in subset[subset["pareto"]].iterrows():
            ox, oy = offsets[int(row["s"]) % 4]
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

        # Pareto frontier line
        pareto = subset[subset["pareto"]].sort_values("fpr")
        if len(pareto) > 1:
            ax.plot(
                pareto["fpr"],
                pareto[time_col],
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

        # Legend: M values as coloured patches

        legend_elements = [
            Patch(facecolor=m_colors[m], edgecolor="black", label=f"M={m}")
            for m in m_values_sorted
        ]
        ax.legend(
            handles=legend_elements,
            fontsize=pu.LEGEND_FONT_SIZE,
            loc="upper left",
            framealpha=0,
        )

        plt.tight_layout()
        suffix = time_col.replace("_time", "")
        pareto_path = output_dir / f"param_sweep_pareto_{suffix}.pdf"
        pu.save_figure(
            fig, pareto_path, f"Pareto plot ({suffix}) saved to {pareto_path}"
        )

    typer.secho("\nAll plots generated successfully!", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
