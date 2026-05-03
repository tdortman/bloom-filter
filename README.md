# GPU Super Bloom Filter

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://tdortman.github.io/bloom-filter/)

## Overview

This library provides a high-performance GPU adaption of the [Super Bloom filter](https://www.biorxiv.org/content/10.64898/2026.03.17.712354v1.article-info) optimised for high-throughput batch k-mer insertion and query on nucleotide (DNA) and protein sequences (or any other sequence type as long as a valid alphabet is provided).

It exploits the streaming nature of sequence-derived k-mers by using **minimisers** to group consecutive k-mers sharing the same minimiser into super-k-mers, assigning all k-mers of a super-k-mer to the same 256-bit memory shard. This amortizes random memory accesses across consecutive k-mer queries, reducing memory-bandwidth pressure. The **findere** scheme further reduces false positives dramatically by inserting overlapping s-mers and requiring a full run of consecutive s-mer matches.

## Features

- CUDA-accelerated batch k-mer insert and query from sequences
- Configurable k-mer length, minimiser width, s-mer width, and hash function count
- Minimizer-based shard selection for cache-efficient streaming queries
- Findere false-positive reduction via overlapping s-mer membership
- Header-only library design
- FASTA/FASTQ stream and file support

## Performance

Benchmarks (`Config<31, 28, 16, 4>`) comparing Super Bloom against [NVIDIA's](https://github.com/NVIDIA/cuCollections) `bloom_filter` on an NVIDIA RTX PRO 6000 Blackwell GPU with random DNA sequence inputs. Throughput is reported in billions of k-mers per second (Gk-mers/s). Timings include GPU-side sequence encoding for cuco, so both methods consume the same input sequence.

| Dataset Size | Operation | Super Bloom    | cuco Bloom     | Speed-up |
| ------------ | --------- | -------------- | -------------- | -------- |
| ~4M k-mers   | Insert    | 62.9 Gk-mers/s | 31.5 Gk-mers/s | 2.0×     |
| ~4M k-mers   | Query     | 109 Gk-mers/s  | 44.2 Gk-mers/s | 2.5×     |
| ~268M k-mers | Insert    | 46.6 Gk-mers/s | 5.9 Gk-mers/s  | 7.9×     |
| ~268M k-mers | Query     | 101 Gk-mers/s  | 13.2 Gk-mers/s | 7.7×     |

The findere scheme (s-mer width) provides strong false-positive reduction at equivalent memory. For `Config<31, 28, 16, 4>` on ~4.6M inserted k-mers queried against 10⁹ random k-mers:

| Bits/k-mer | Super Bloom FPR | cuco Bloom FPR |
| ---------- | --------------- | -------------- |
| 58         | 0.0035%         | 0.064%         |
| 116        | 0.00036%        | 0.017%         |
| 231        | 0.000092%       | 0.0054%        |

Benchmarks can be reproduced with:

```bash
./build/benchmarks/gpu-filter-comparison --benchmark_filter="SuperBloom"
./build/benchmarks/fpr-fastx-sweep
```

## Requirements

- CUDA Toolkit (>= 12.9, tested with 13.2)
- C++20 compatible host compiler
- Meson build system
- NVIDIA GPU with compute capability 8.0+ (Ampere, Lovelace, Hopper, Blackwell)

## Building

```bash
meson setup build
ninja -C build
```

Benchmarks and tests are built automatically when this is a standalone project, but skipped when used as a subproject. Control with Meson feature options:

| Option                        | Behaviour                                                 |
| ----------------------------- | --------------------------------------------------------- |
| `-Dbenchmarks=auto` (default) | Build benchmarks when standalone, skip when subproject    |
| `-Dbenchmarks=enabled`        | Always build benchmarks                                   |
| `-Dbenchmarks=disabled`       | Never build benchmarks                                    |
| `-Dtests=auto` (default)      | Build tests when standalone, skip when subproject         |
| `-Dtests=enabled`             | Always build tests                                        |
| `-Dtests=disabled`            | Never build tests                                         |
| `-Dparam_sweep=enabled`       | Build parameter-sweep benchmark (s, m) for DNA or protein |
| `-Dparam_sweep_alphabet=dna`  | Alphabet for param_sweep: `dna` or `protein`              |

> [!IMPORTANT]
> The parameter sweep is disabled by default for a reason, there are 208 binaries for the entire sweep when using the DNA alphabet.

Examples:

```bash
# Standalone: build everything except parameter sweep (default)
meson setup build

# Standalone: skip benchmarks and tests
meson setup build -Dbenchmarks=disabled -Dtests=disabled

# Subproject: force benchmarks and tests on
meson setup build -Dbenchmarks=enabled -Dtests=enabled

# Parameter-sweep benchmark (DNA, default)
meson setup build -Dparam_sweep=enabled

# Parameter-sweep benchmark (protein)
meson setup build -Dparam_sweep=enabled -Dparam_sweep_alphabet=protein
```

## Usage

```cpp
#include <bloom/BloomFilter.cuh>

// Configure the filter: k-mer length, s-mer width, minimizer width, hash count
using Config = bloom::Config<31, 28, 16, 4>;

// Create a filter with the desired capacity (in bits)
bloom::Filter<Config> filter(1 << 24);  // ~16M bits

// Insert k-mers from a DNA sequence (synchronous)
filter.insertSequence("ACGTACGTACGTACGTACGTACGTACGTACGT");

// Query k-mers (returns vector<uint8_t>, 1 = present, 0 = absent)
auto hits = filter.containsSequence("ACGTACGTACGTACGTACGTACGTACGTACGT");

// Device-resident API (async, no synchronization)
thrust::device_vector<char> d_seq = ...;
thrust::device_vector<uint8_t> d_results(numKmers);
filter.insertSequenceDevice(bloom::device_span<const char>(d_seq));
filter.containsSequenceDevice(
    bloom::device_span<const char>(d_seq),
    bloom::device_span<uint8_t>(d_results)
);

// FASTA/FASTQ file insertion and query
filter.insertFastxFile("reference.fasta");
auto reports = filter.queryFastxFile("queries.fastq");

// Inspect filter state
double load = filter.loadFactor();  // fraction of set bits
uint64_t bits = filter.filterBits();
```

### Configuration Options

The `Config` template accepts the following parameters:

| Parameter       | Description                                       | Default       |
| --------------- | ------------------------------------------------- | ------------- |
| `K`             | k-mer length (max depends on alphabet)            | -             |
| `S`             | s-mer width for findere Bloom hash seed (1-K)     | -             |
| `M`             | Minimiser width for shard selection (1-K)         | -             |
| `HashCount`     | Number of independent Bloom hash functions (1-16) | -             |
| `CudaBlockSize` | CUDA threads per block                            | 256           |
| `Alphabet`      | Symbol encoding (DNA or protein)                  | `DnaAlphabet` |

### Protein Alphabet Support

```cpp
#include <bloom/BloomFilter.cuh>

using ProteinConfig = bloom::Config<12, 10, 6, 4, 256, bloom::ProteinAlphabet>;
bloom::Filter<ProteinConfig> filter(1 << 24);

filter.insertSequence("ACDEFGHIKLMNPQRSTVWY");
auto hits = filter.containsSequence("ACDEFGHIKLMNPQRSTVWY");
```

## Related Publications

- E. Conchon-Kerjan, T. Rouzé, L. Robidou, F. Ingels, and A. Limasset, “Super Bloom: Fast and precise filter for streaming k-mer queries,” bioRxiv, 2026, doi: 10.64898/2026.03.17.712354.
- D. Jünger, K. Kristensen, Y. Wang, X. Yu, and B. Schmidt, “Optimizing Bloom Filters for Modern GPU Architectures.” 2025. [Online]. Available: https://arxiv.org/abs/2512.15595
