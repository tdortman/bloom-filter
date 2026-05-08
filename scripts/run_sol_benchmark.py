#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "typer",
# ]
# ///

import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from ncu_profiler import FILTERS, OPERATIONS, run_ncu_profile, to_superscript

app = typer.Typer(
    help="Run Speed of Light throughput benchmarks with NVIDIA Nsight Compute"
)

DEFAULT_MIN_CAPACITY_LOG2 = 16
DEFAULT_MAX_CAPACITY_LOG2 = 28

SOL_METRICS_MAP = {
    "sm_throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "memory_throughput": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_throughput": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "l2_throughput": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
}

SOL_METRICS_LIST = list(SOL_METRICS_MAP.values())


@app.command()
def main(
    executable: Path = typer.Argument(
        ...,
        help="Path to the gpu-filter-profiler executable",
        exists=True,
    ),
    output: Path = typer.Option(
        "benchmark_sol.csv",
        "--output",
        "-o",
        help="Output CSV file for results",
    ),
    min_capacity_log2: int = typer.Option(
        DEFAULT_MIN_CAPACITY_LOG2,
        "--min-log2",
        help="Minimum capacity as log2(capacity)",
    ),
    max_capacity_log2: int = typer.Option(
        DEFAULT_MAX_CAPACITY_LOG2,
        "--max-log2",
        help="Maximum capacity as log2(capacity)",
    ),
    load_factor: float = typer.Option(
        0.95,
        "--load-factor",
        "-l",
        help="Load factor for all runs",
    ),
    filters: Optional[list[str]] = typer.Option(
        None,
        "--filter",
        "-f",
        help="Specific filters to test (can specify multiple)",
    ),
):
    """Profile Speed of Light throughputs across varying input sizes."""
    try:
        subprocess.run(["ncu", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        typer.secho(
            "NVIDIA Nsight Compute (ncu) not found. Please install it.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    test_filters = filters if filters else FILTERS
    unknown_filters = sorted(set(test_filters) - set(FILTERS))
    if unknown_filters:
        typer.secho(
            f"Unknown filter(s): {', '.join(unknown_filters)}. Valid filters: {', '.join(FILTERS)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    if min_capacity_log2 > max_capacity_log2:
        typer.secho("--min-log2 must be <= --max-log2", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    exponents = list(range(min_capacity_log2, max_capacity_log2 + 1))
    capacities = [2**e for e in exponents]

    typer.secho(
        f"Testing {len(test_filters)} filter(s) across {len(exponents)} capacity values with load factor {load_factor}",
        fg=typer.colors.GREEN,
        err=True,
    )
    typer.secho(
        f"Capacity range: 2{to_superscript(exponents[0])} ({capacities[0]:,}) to 2{to_superscript(exponents[-1])} ({capacities[-1]:,})",
        fg=typer.colors.GREEN,
        err=True,
    )

    results = []
    total_runs = sum(len(OPERATIONS[f]) for f in test_filters) * len(exponents)
    current_run = 0

    for filter_type in test_filters:
        for operation in OPERATIONS[filter_type]:
            for exponent in exponents:
                current_run += 1
                typer.secho(
                    f"\n[{current_run}/{total_runs}] ",
                    fg=typer.colors.BRIGHT_BLUE,
                    err=True,
                    nl=False,
                )

                metrics = run_ncu_profile(
                    executable,
                    filter_type,
                    operation,
                    exponent,
                    load_factor,
                    SOL_METRICS_LIST,
                )

                if metrics:
                    result_entry = {
                        "filter": filter_type,
                        "operation": operation,
                        "capacity": 2**exponent,
                        "load_factor": load_factor,
                    }

                    log_parts = []
                    for friendly_name, ncu_name in SOL_METRICS_MAP.items():
                        value = metrics.get(ncu_name, 0.0)
                        result_entry[friendly_name] = value
                        log_parts.append(f"{friendly_name.split('_')[0].upper()}: {value:.1f}%")

                    results.append(result_entry)
                    typer.secho("  " + ", ".join(log_parts), fg=typer.colors.GREEN, err=True)

    if not results:
        typer.secho("\nNo results collected", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output, index=False)

    typer.secho(
        f"\nResults saved to {output} ({len(results)} data points)",
        fg=typer.colors.GREEN,
        err=True,
    )

    typer.secho("\nSummary Statistics (Mean %):", fg=typer.colors.BRIGHT_CYAN, err=True)
    summary = df.groupby(["filter", "operation"])[list(SOL_METRICS_MAP.keys())].mean().round(2)
    print(summary)


if __name__ == "__main__":
    app()
