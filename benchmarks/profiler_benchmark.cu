#include <cuda/__cmath/ceil_div.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <CLI/CLI.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include <bloom/BloomFilter.cuh>
#include <bloom/device_span.cuh>
#include <bloom/helpers.cuh>
#include <cuckoogpu/CuckooFilter.cuh>
#include <cuco/bloom_filter.cuh>

#include "benchmark_common.cuh"

using SuperBloomConfig = bloom::Config<31, 28, 16, 4>;
using SuperBloomFilter = bloom::Filter<SuperBloomConfig>;
using CucoBloom = cuco::bloom_filter<uint64_t>;
using CuckooGpuConfig = cuckoogpu::Config<uint64_t, 16, 500, 256, 16>;
using CuckooGpuFilter = cuckoogpu::Filter<CuckooGpuConfig>;

constexpr uint64_t kBitsPerItem = 16;

uint64_t cucoNumBlocks(uint64_t numItems) {
    constexpr auto bitsPerWord = sizeof(typename CucoBloom::word_type) * 8;
    return cuda::ceil_div(numItems * kBitsPerItem, CucoBloom::words_per_block * bitsPerWord);
}

struct BenchmarkInput {
    explicit BenchmarkInput(uint64_t numKmers) : sequenceLength(numKmers + SuperBloomConfig::k - 1) {
        benchmark_common::gpuGenerateDna(d_sequence, sequenceLength, 42);
        d_packedKmers.resize(numKmers);
        benchmark_common::gpuEncodePackedKmers<SuperBloomConfig::k, bloom::DnaAlphabet>(
            thrust::raw_pointer_cast(d_sequence.data()),
            sequenceLength,
            thrust::raw_pointer_cast(d_packedKmers.data())
        );
        BLOOM_CUDA_CALL(cudaDeviceSynchronize());
    }

    uint64_t sequenceLength{};
    thrust::device_vector<char> d_sequence;
    thrust::device_vector<uint64_t> d_packedKmers;
};

void benchmarkSuperBloomInsert(uint64_t capacity, double loadFactor) {
    const auto n = static_cast<uint64_t>(capacity * loadFactor);
    BenchmarkInput input(n);
    SuperBloomFilter filter(cuda::std::bit_ceil(n * kBitsPerItem));

    benchmark::DoNotOptimize(filter.insertSequenceDevice(bloom::device_span<const char>{
        thrust::raw_pointer_cast(input.d_sequence.data()), input.sequenceLength
    }));
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());

    filter.clear();
    benchmark::DoNotOptimize(filter.insertSequenceDevice(bloom::device_span<const char>{
        thrust::raw_pointer_cast(input.d_sequence.data()), input.sequenceLength
    }));
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());
}

void benchmarkSuperBloomQuery(uint64_t capacity, double loadFactor) {
    const auto n = static_cast<uint64_t>(capacity * loadFactor);
    BenchmarkInput input(n);
    thrust::device_vector<uint8_t> d_output(n);
    SuperBloomFilter filter(cuda::std::bit_ceil(n * kBitsPerItem));

    benchmark::DoNotOptimize(filter.insertSequenceDevice(bloom::device_span<const char>{
        thrust::raw_pointer_cast(input.d_sequence.data()), input.sequenceLength
    }));
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());

    filter.containsSequenceDevice(
        bloom::device_span<const char>{
            thrust::raw_pointer_cast(input.d_sequence.data()), input.sequenceLength
        },
        bloom::device_span<uint8_t>{thrust::raw_pointer_cast(d_output.data()), d_output.size()}
    );
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());
    benchmark::DoNotOptimize(thrust::raw_pointer_cast(d_output.data()));
}

void benchmarkCucoBloomInsert(uint64_t capacity, double loadFactor) {
    const auto n = static_cast<uint64_t>(capacity * loadFactor);
    BenchmarkInput input(n);
    CucoBloom filter(cucoNumBlocks(n));

    filter.add(input.d_packedKmers.begin(), input.d_packedKmers.end());
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());

    filter.clear();
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());
    filter.add(input.d_packedKmers.begin(), input.d_packedKmers.end());
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());
}

void benchmarkCucoBloomQuery(uint64_t capacity, double loadFactor) {
    const auto n = static_cast<uint64_t>(capacity * loadFactor);
    BenchmarkInput input(n);
    thrust::device_vector<uint8_t> d_output(n);
    CucoBloom filter(cucoNumBlocks(n));

    filter.add(input.d_packedKmers.begin(), input.d_packedKmers.end());
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());

    filter.contains(
        input.d_packedKmers.begin(),
        input.d_packedKmers.end(),
        reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
    );
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());
    benchmark::DoNotOptimize(thrust::raw_pointer_cast(d_output.data()));
}

void benchmarkCuckooGpuInsert(uint64_t capacity, double loadFactor) {
    const auto n = static_cast<uint64_t>(capacity * loadFactor);
    BenchmarkInput input(n);
    CuckooGpuFilter filter(capacity);

    filter.insertMany(input.d_packedKmers);
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());

    filter.clear();
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());
    filter.insertMany(input.d_packedKmers);
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());
}

void benchmarkCuckooGpuQuery(uint64_t capacity, double loadFactor) {
    const auto n = static_cast<uint64_t>(capacity * loadFactor);
    BenchmarkInput input(n);
    thrust::device_vector<uint8_t> d_output(n);
    CuckooGpuFilter filter(capacity);

    filter.insertMany(input.d_packedKmers);
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());

    filter.containsMany(input.d_packedKmers, d_output);
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());
    benchmark::DoNotOptimize(thrust::raw_pointer_cast(d_output.data()));
}

int main(int argc, char** argv) {
    CLI::App app{"GPU filter hardware profiler benchmark"};

    std::string filter = "superbloom";
    std::string operation = "insert";
    uint64_t exponent = 24;
    double loadFactor = 0.95;

    app.add_option("filter", filter, "Filter type: superbloom, cucobloom, cuckoogpu")
        ->required()
        ->check(CLI::IsMember({"superbloom", "cucobloom", "cuckoogpu"}));
    app.add_option("operation", operation, "Operation: insert, query")
        ->required()
        ->check(CLI::IsMember({"insert", "query"}));
    app.add_option("exponent", exponent, "Exponent for capacity = 2^x")
        ->required()
        ->check(CLI::PositiveNumber);
    app.add_option("-l,--load-factor", loadFactor, "Load factor (0.0-1.0)")
        ->default_val(0.95)
        ->check(CLI::Range(0.0, 1.0));

    CLI11_PARSE(app, argc, argv);

    const uint64_t capacity = uint64_t{1} << exponent;
    const auto n = static_cast<uint64_t>(capacity * loadFactor);

    std::cout << "Filter: " << filter << '\n';
    std::cout << "Operation: " << operation << '\n';
    std::cout << "Capacity: " << capacity << '\n';
    std::cout << "Load Factor: " << loadFactor << '\n';
    std::cout << "Number of keys: " << n << '\n';

    if (filter == "superbloom") {
        if (operation == "insert") {
            benchmarkSuperBloomInsert(capacity, loadFactor);
        } else {
            benchmarkSuperBloomQuery(capacity, loadFactor);
        }
    } else if (filter == "cucobloom") {
        if (operation == "insert") {
            benchmarkCucoBloomInsert(capacity, loadFactor);
        } else {
            benchmarkCucoBloomQuery(capacity, loadFactor);
        }
    } else if (filter == "cuckoogpu") {
        if (operation == "insert") {
            benchmarkCuckooGpuInsert(capacity, loadFactor);
        } else {
            benchmarkCuckooGpuQuery(capacity, loadFactor);
        }
    }

    return 0;
}
