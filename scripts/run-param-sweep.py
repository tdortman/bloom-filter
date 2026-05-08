#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "typer",
# ]
# ///
"""Run the parameter-sweep group binaries and merge their CSV results.

Usage:
  python scripts/run-param-sweep.py --build-dir build --insert-fastx reads.fasta -o results.csv
  python scripts/run-param-sweep.py -b build --insert-fastx reads.fasta --query-fastx queries.fasta

The script discovers all param-sweep-groupN executables, runs each one
with --benchmark_out=tmp.csv, then merges the CSV files into a single output.
"""

from pathlib import Path
from typing import Optional

import typer

from benchmark_utils import run_benchmarks_and_merge

app = typer.Typer()


def _find_group_binaries(benchmark_dir: Path) -> list[tuple[int, Path]]:
    """Find param-sweep-groupN executables, sorted by group number."""
    result = []
    for p in sorted(benchmark_dir.glob("param-sweep-group*")):
        name = p.name
        if name.startswith("param-sweep-group"):
            try:
                group_num = int(name[len("param-sweep-group") :])
            except ValueError:
                continue
            result.append((group_num, p))
    result.sort(key=lambda x: x[0])
    return result


@app.command()
def main(
    build_dir: Path = typer.Option(
        ...,
        "--build-dir",
        "-b",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Build directory (e.g. 'build')",
    ),
    insert_fastx: Path = typer.Option(
        ...,
        "--insert-fastx",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="FASTA/FASTQ file for insert (k-mer set)",
    ),
    query_fastx: Optional[Path] = typer.Option(
        None,
        "--query-fastx",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="FASTA/FASTQ file for query throughput (default: GPU-generated)",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output CSV path",
    ),
    benchmark_filter: Optional[str] = typer.Option(
        None,
        "--benchmark-filter",
        help="Google Benchmark filter (passed through to each binary)",
    ),
):
    """Run param-sweep group binaries and merge CSV results."""
    benchmark_dir = build_dir / "benchmarks"
    groups = _find_group_binaries(benchmark_dir)

    if not groups:
        typer.secho(
            "No param-sweep-group* binaries found. Build with: "
            "meson setup build -Dparam_sweep=enabled && ninja -C build",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Found {len(groups)} group binaries")

    extra_args = [f"--insert-fastx={insert_fastx}"]
    if query_fastx:
        extra_args.append(f"--query-fastx={query_fastx}")
    if benchmark_filter:
        extra_args.append(f"--benchmark_filter={benchmark_filter}")

    entries = [(exe, None, None, extra_args) for _, exe in groups]

    run_benchmarks_and_merge(entries, output)  # ty:ignore[invalid-argument-type]
    typer.echo(f"\nMerged {len(groups)} CSVs -> {output}")


if __name__ == "__main__":
    app()
