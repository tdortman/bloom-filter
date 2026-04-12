#include <CLI/CLI.hpp>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

#include <bloom/BloomFilter.cuh>
#include <bloom/helpers.cuh>

#include "benchmark_common.cuh"

// Fixed compile-time configuration: K=31, S=27, M=21, H=3
using BenchConfig = bloom::Config<31, 27, 21, 3>;

int main(int argc, char** argv) {
    CLI::App app{"GPU SuperBloom benchmark"};

    std::string indexFasta;
    std::string queryFasta;
    uint64_t filterBits = 1ULL << 35;

    app.add_option("--index-fasta,-i", indexFasta, "FASTA/FASTQ file to index")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--query-fasta,-q", queryFasta, "FASTA/FASTQ file to query")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--filter-bits", filterBits, "Total bloom filter bits")
        ->default_val(filterBits);

    CLI11_PARSE(app, argc, argv);

    try {
        std::cout << "SuperBloom GPU Benchmark\n";
        std::cout << "========================\n";
        std::cout << "index fasta: " << indexFasta << "\n";
        std::cout << "query fasta: " << queryFasta << "\n";
        std::cout << "config: K=" << BenchConfig::k << " S=" << BenchConfig::s
                  << " M=" << BenchConfig::m << " H=" << BenchConfig::hashCount << "\n";
        std::cout << "filter bits: " << filterBits << "\n";

        benchmark_common::GPUTimer timer;

        timer.start();
        bloom::Filter<BenchConfig> filter(filterBits);
        double buildSecs = timer.elapsed();
        std::printf("build_s: %.6f\n", buildSecs);

        timer.start();
        auto insertReport = filter.insertFastxFile(indexFasta);
        double indexSecs = timer.elapsed();
        std::printf("index_s: %.6f\n", indexSecs);
        std::printf("records_indexed: %lu\n", insertReport.recordsIndexed);
        std::printf("indexed_bases: %lu\n", insertReport.indexedBases);
        std::printf("kmers_inserted: %lu\n", insertReport.insertedKmers);
        if (indexSecs > 0.0) {
            std::printf(
                "index_kmers_per_s: %.0f\n",
                static_cast<double>(insertReport.insertedKmers) / indexSecs
            );
        }

        timer.start();
        auto queryReport = filter.queryFastxFile(queryFasta);
        double querySecs = timer.elapsed();
        std::printf("query_s: %.6f\n", querySecs);
        std::printf("records_queried: %lu\n", queryReport.recordsQueried);
        std::printf("queried_bases: %lu\n", queryReport.queriedBases);
        std::printf("queried_kmers: %lu\n", queryReport.queriedKmers);
        std::printf("positive_kmers: %lu\n", queryReport.positiveKmers);
        if (querySecs > 0.0) {
            std::printf(
                "query_kmers_per_s: %.0f\n",
                static_cast<double>(queryReport.queriedKmers) / querySecs
            );
        }

        double fpr = 0.0;
        if (queryReport.queriedKmers > 0) {
            fpr = static_cast<double>(queryReport.positiveKmers) /
                  static_cast<double>(queryReport.queriedKmers) * 100.0;
        }
        std::printf("fpr_percentage: %.6f\n", fpr);
        std::printf("load_factor: %.6f\n", filter.loadFactor());
        std::printf("actual_filter_bits: %lu\n", filter.filterBits());

        double totalSecs = buildSecs + indexSecs + querySecs;
        std::printf("total_s: %.6f\n", totalSecs);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
