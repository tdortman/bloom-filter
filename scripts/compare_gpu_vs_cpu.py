#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "typer",
#   "matplotlib",
#   "pandas",
# ]
# ///
"""Compare GPU SuperBloom (CUDA) vs CPU SuperBloom (Rust) performance.

Runs both benchmark binaries with matched parameters, collects timing data
over multiple repeats, and reports median values. Optionally generates a
comparison bar chart.

Usage:
    python scripts/compare_gpu_vs_cpu.py \
        --index-fasta data/genomes/chr14.fna \
        --query-fasta data/genomes/chr14.fna

    python scripts/compare_gpu_vs_cpu.py \
        --index-fasta data/genomes/chr14.fna \
        --query-fasta data/genomes/chr14.fna \
        --repeats 7 --plot
"""

import csv
import re
import subprocess
from pathlib import Path
from statistics import median
from typing import Optional

import typer
from benchmark_utils import validate_executable

app = typer.Typer(add_completion=False)

# Shared fixed config: K=31, S=27, M=21, H=3
GPU_CONFIG = {"k": 31, "s": 27, "m": 21, "n_hashes": 3}

METRIC_KEYS = [
    "build_s",
    "index_s",
    "query_s",
    "total_s",
    "index_kmers_per_s",
    "query_kmers_per_s",
    "kmers_inserted",
    "queried_kmers",
    "positive_kmers",
    "load_factor",
]


def _parse_key_value_output(stdout: str) -> dict[str, float]:
    """Parse 'key: value' lines from GPU benchmark output into a dict."""
    result: dict[str, float] = {}
    for line in stdout.splitlines():
        match = re.match(r"^\s*(\w+):\s+([\d.eE+\-]+)\s*$", line)
        if match:
            key, value = match.group(1), match.group(2)
            try:
                result[key] = float(value)
            except ValueError:
                pass
    return result


def _parse_rust_output(stdout: str) -> dict[str, float]:
    """Parse the Rust SuperBloom benchmark output format.

    Expected lines like:
        1) build index object: 0.123s
        2) index fasta: 1.234s
           indexing throughput: 12345 k-mers/s
        3) query fasta: 0.567s
           queried k-mers:    54321
           positive k-mers:   100
           query throughput:   98765 k-mers/s
        Total: 1.924s
    """
    result: dict[str, float] = {}
    patterns = [
        (r"build index object:\s+([\d.]+)s", "build_s"),
        (r"index fasta:\s+([\d.]+)s", "index_s"),
        (r"query fasta:\s+([\d.]+)s", "query_s"),
        (r"indexing throughput:\s+([\d.]+)\s+k-mers/s", "index_kmers_per_s"),
        (r"query throughput:\s+([\d.]+)\s+k-mers/s", "query_kmers_per_s"),
        (r"k-mers added:\s+(\d+)", "kmers_inserted"),
        (r"queried k-mers:\s+(\d+)", "queried_kmers"),
        (r"positive k-mers:\s+(\d+)", "positive_kmers"),
        (r"Total:\s+([\d.]+)s", "total_s"),
    ]
    for line in stdout.splitlines():
        for pattern, key in patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    result[key] = float(match.group(1))
                except ValueError:
                    pass

    return result


def _run_gpu_benchmark(
    executable: Path,
    index_fasta: str,
    query_fasta: str,
    filter_bits: int,
) -> dict[str, float]:
    """Run the GPU benchmark binary once and parse its output."""
    cmd = [
        str(executable),
        "--index-fasta",
        index_fasta,
        "--query-fasta",
        query_fasta,
        "--filter-bits",
        str(filter_bits),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        typer.secho(
            f"GPU benchmark failed:\n{result.stderr}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(1)
    return _parse_key_value_output(result.stdout)


def _run_cpu_benchmark(
    executable: Path,
    index_fasta: str,
    query_fasta: str,
    bit_vector_size_exponent: int,
    threads: int,
) -> dict[str, float]:
    """Run the Rust CPU benchmark binary once and parse its output."""
    cmd = [
        str(executable),
        "--index-fasta",
        index_fasta,
        "--query-fasta",
        query_fasta,
        "--k",
        str(GPU_CONFIG["k"]),
        "--m",
        str(GPU_CONFIG["m"]),
        "--s",
        str(GPU_CONFIG["s"]),
        "--n-hashes",
        str(GPU_CONFIG["n_hashes"]),
        "--bit-vector-size-exponent",
        str(bit_vector_size_exponent),
        "--threads",
        str(threads),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        typer.secho(
            f"CPU benchmark failed:\n{result.stderr}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(1)
    return _parse_rust_output(result.stdout)


def _collect_medians(runs: list[dict[str, float]]) -> dict[str, float]:
    """Compute median for each metric across runs."""
    if not runs:
        return {}
    keys = runs[0].keys()
    return {k: median([r[k] for r in runs if k in r]) for k in keys}


def _filter_bits_from_exponent(exponent: int) -> int:
    """Convert Rust-style bit_vector_size_exponent to absolute filter bits."""
    return 1 << exponent


@app.command()
def main(
    index_fasta: str = typer.Option(
        ..., "--index-fasta", "-i", help="FASTA/FASTQ file to index"
    ),
    query_fasta: str = typer.Option(
        ..., "--query-fasta", "-q", help="FASTA/FASTQ file to query"
    ),
    filter_bits_exponent: int = typer.Option(
        35,
        "--filter-bits-exponent",
        help="Filter size as power of 2 (bits = 2^exponent). Used by both GPU and CPU.",
    ),
    repeats: int = typer.Option(
        5, "--repeats", "-r", help="Number of benchmark runs per implementation"
    ),
    threads: int = typer.Option(
        8, "--threads", "-t", help="Thread count for CPU (Rust) benchmark"
    ),
    output_csv: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output CSV path"
    ),
    plot: bool = typer.Option(
        False, "--plot", "-p", help="Generate comparison bar chart"
    ),
    plot_output: Optional[Path] = typer.Option(
        None, "--plot-output", help="Plot output path (PDF)"
    ),
    gpu_executable: Optional[Path] = typer.Option(
        None, "--gpu-exe", help="Path to GPU benchmark binary"
    ),
    cpu_executable: Optional[Path] = typer.Option(
        None, "--cpu-exe", help="Path to CPU benchmark binary"
    ),
) -> None:
    """Run GPU vs CPU SuperBloom comparison benchmark."""
    script_dir = Path(__file__).parent
    build_dir = script_dir.parent / "build"
    bench_dir = build_dir / "benchmarks"

    if gpu_executable is None:
        gpu_executable = bench_dir / "superbloom-gpu-benchmark"
    if cpu_executable is None:
        # Built by the superbloom-cpu subproject via Cargo custom_target
        cpu_executable = (
            build_dir / "subprojects" / "superbloom-cpu" / "release" / "benchmark"
        )

    validate_executable(gpu_executable)
    validate_executable(cpu_executable)

    filter_bits = _filter_bits_from_exponent(filter_bits_exponent)

    typer.secho(
        f"Config: K={GPU_CONFIG['k']} S={GPU_CONFIG['s']} M={GPU_CONFIG['m']} H={GPU_CONFIG['n_hashes']}",
        fg=typer.colors.CYAN,
    )
    typer.secho(
        f"Filter bits: 2^{filter_bits_exponent} = {filter_bits:,}", fg=typer.colors.CYAN
    )
    typer.secho(f"Repeats: {repeats}  |  CPU threads: {threads}", fg=typer.colors.CYAN)
    typer.echo()

    # --- GPU runs ---
    typer.secho("Running GPU benchmark...", fg=typer.colors.BLUE, bold=True)
    gpu_runs: list[dict[str, float]] = []
    for i in range(repeats):
        typer.echo(f"  GPU run {i + 1}/{repeats}")
        metrics = _run_gpu_benchmark(
            gpu_executable, index_fasta, query_fasta, filter_bits
        )
        gpu_runs.append(metrics)

    # --- CPU runs ---
    typer.secho("Running CPU benchmark...", fg=typer.colors.BLUE, bold=True)
    cpu_runs: list[dict[str, float]] = []
    for i in range(repeats):
        typer.echo(f"  CPU run {i + 1}/{repeats}")
        metrics = _run_cpu_benchmark(
            cpu_executable, index_fasta, query_fasta, filter_bits_exponent, threads
        )
        cpu_runs.append(metrics)

    gpu_medians = _collect_medians(gpu_runs)
    cpu_medians = _collect_medians(cpu_runs)

    # --- Print results ---
    typer.echo()
    typer.secho("=== Median Results ===", fg=typer.colors.GREEN, bold=True)
    header = f"{'Metric':<25s} {'GPU':>15s} {'CPU (Rust)':>15s} {'Speedup':>10s}"
    typer.echo(header)
    typer.echo("-" * len(header))

    for key in METRIC_KEYS:
        gpu_val = gpu_medians.get(key)
        cpu_val = cpu_medians.get(key)
        if gpu_val is None and cpu_val is None:
            continue

        gpu_str = f"{gpu_val:.4g}" if gpu_val is not None else "n/a"
        cpu_str = f"{cpu_val:.4g}" if cpu_val is not None else "n/a"

        speedup_str = ""
        if gpu_val is not None and cpu_val is not None and cpu_val > 0:
            if key.endswith("_per_s"):
                speedup = gpu_val / cpu_val
                speedup_str = f"{speedup:.2f}x"
            elif key.endswith("_s"):
                speedup = cpu_val / gpu_val if gpu_val > 0 else float("inf")
                speedup_str = f"{speedup:.2f}x"

        typer.echo(f"{key:<25s} {gpu_str:>15s} {cpu_str:>15s} {speedup_str:>10s}")

    # --- Write CSV ---
    if output_csv is None:
        output_csv = script_dir.parent / "build" / "gpu_vs_cpu_comparison.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["implementation"] + METRIC_KEYS
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({"implementation": "gpu", **gpu_medians})
        writer.writerow({"implementation": "cpu_rust", **cpu_medians})

    typer.secho(f"\nWrote {output_csv}", fg=typer.colors.GREEN)

    # --- Plot ---
    if plot:
        from plot_gpu_vs_cpu import generate_plot

        if plot_output is None:
            plot_output = output_csv.with_suffix(".pdf")
        generate_plot(gpu_medians, cpu_medians, plot_output, config=GPU_CONFIG)


if __name__ == "__main__":
    app()
