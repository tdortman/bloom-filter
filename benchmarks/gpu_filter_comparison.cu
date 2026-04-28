#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <cuda/std/bit>
#include <cuda/std/span>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <bloom/BloomFilter.cuh>
#include <bloom/device_span.cuh>
#include <bloom/helpers.cuh>
#include <cuckoogpu/CuckooFilter.cuh>
#include <cuco/bloom_filter.cuh>

#include "benchmark_common.cuh"

namespace bm = benchmark;

using CucoBloom = cuco::bloom_filter<uint64_t>;
using CuckooGpuConfig = cuckoogpu::Config<uint64_t, 16, 500, 256, 16>;
using CuckooGpuFilter = cuckoogpu::Filter<CuckooGpuConfig>;

// It's K - S - M - H

#define SUPERBLOOM_FIRST_INSERT_QUERY_FPR_CONFIG(X) X(31, 28, 16, 4)

#define SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR(X) SUPERBLOOM_FIRST_INSERT_QUERY_FPR_CONFIG(X)

#define SUPERBLOOM_CONFIGS_FPR_ONLY(X) \
    X(31, 31, 21, 4)                   \
    X(31, 30, 21, 4)                   \
    X(31, 28, 21, 4)                   \
    X(31, 27, 21, 4)                   \
    X(31, 20, 21, 4)                   \
    X(31, 16, 21, 4)

#define FOR_EACH_SUPERBLOOM_CONFIG(X)      \
    SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR(X) \
    SUPERBLOOM_CONFIGS_FPR_ONLY(X)

constexpr uint64_t kBitsPerItem = 16;

FOR_EACH_SUPERBLOOM_CONFIG(BENCHMARK_DEFINE_SUPERBLOOM_CONFIG_AND_FIXTURE)

#define DEFINE_CUCO_REFERENCE_CONFIG(K, S, M, H) \
    using CucoReferenceConfig = BENCHMARK_SUPERBLOOM_CONFIG_SYMBOL(K, S, M, H);

SUPERBLOOM_FIRST_INSERT_QUERY_FPR_CONFIG(DEFINE_CUCO_REFERENCE_CONFIG)

#undef DEFINE_CUCO_REFERENCE_CONFIG

uint64_t cucoNumBlocks(uint64_t numItems) {
    constexpr auto bitsPerWord = sizeof(typename CucoBloom::word_type) * 8;
    return bloom::detail::divUp(numItems * kBitsPerItem, CucoBloom::words_per_block * bitsPerWord);
}

class CucoBloomFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    static constexpr uint64_t k = CucoReferenceConfig::k;

    void SetUp(const bm::State& state) override {
        sequenceLength = static_cast<uint64_t>(state.range(0));
        benchData = &benchmark_common::getBenchmarkData<CucoReferenceConfig::k>(sequenceLength);
        numKmers = benchData->numKmers;

        d_output.resize(numKmers);

        filter = std::make_unique<CucoBloom>(cucoNumBlocks(numKmers));
        filterMemory = filter->block_extent() * CucoBloom::words_per_block *
                       sizeof(typename CucoBloom::word_type);
    }

    void TearDown(const bm::State&) override {
        filter.reset();
        benchData = nullptr;
        d_output.clear();
        d_output.shrink_to_fit();
    }

    void setCounters(bm::State& state) const {
        benchmark_common::setBenchmarkCounters(state, filterMemory, sequenceLength, numKmers);
    }

    uint64_t sequenceLength{};
    uint64_t numKmers{};
    uint64_t filterMemory{};
    benchmark_common::BenchmarkData<CucoReferenceConfig::k>* benchData{};
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<CucoBloom> filter;
    benchmark_common::GPUTimer timer;
};

class CuckooGpuFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    static constexpr uint64_t k = CucoReferenceConfig::k;

    void SetUp(const bm::State& state) override {
        sequenceLength = static_cast<uint64_t>(state.range(0));
        benchData = &benchmark_common::getBenchmarkData<CucoReferenceConfig::k>(sequenceLength);
        numKmers = benchData->numKmers;

        d_output.resize(numKmers);

        filter = std::make_unique<CuckooGpuFilter>(numKmers / 0.95);
        filterMemory = filter->sizeInBytes();
    }

    void TearDown(const bm::State&) override {
        filter.reset();
        benchData = nullptr;
        d_output.clear();
        d_output.shrink_to_fit();
    }

    void setCounters(bm::State& state) const {
        benchmark_common::setBenchmarkCounters(state, filterMemory, sequenceLength, numKmers);
    }

    uint64_t sequenceLength{};
    uint64_t numKmers{};
    uint64_t filterMemory{};
    benchmark_common::BenchmarkData<CucoReferenceConfig::k>* benchData{};
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<CuckooGpuFilter> filter;
    benchmark_common::GPUTimer timer;
};


void runCucoInsertBenchmark(CucoBloomFixture& fixture, bm::State& state) {
    for (auto _ : state) {
        fixture.filter->clear();
        BLOOM_CUDA_CALL(cudaDeviceSynchronize());

        fixture.timer.start();
        benchmark_common::gpuEncodePackedKmers<CucoBloomFixture::k>(
            thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
            fixture.sequenceLength,
            thrust::raw_pointer_cast(fixture.benchData->d_throughputPackedKmers.data())
        );
        fixture.filter->add(
            fixture.benchData->d_throughputPackedKmers.begin(),
            fixture.benchData->d_throughputPackedKmers.end()
        );
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
    }
    fixture.setCounters(state);
}

void runCucoQueryBenchmark(CucoBloomFixture& fixture, bm::State& state) {
    fixture.filter->clear();
    benchmark_common::gpuEncodePackedKmers<CucoBloomFixture::k>(
        thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
        fixture.sequenceLength,
        thrust::raw_pointer_cast(fixture.benchData->d_throughputPackedKmers.data())
    );
    fixture.filter->add(
        fixture.benchData->d_throughputPackedKmers.begin(),
        fixture.benchData->d_throughputPackedKmers.end()
    );
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fixture.timer.start();
        benchmark_common::gpuEncodePackedKmers<CucoBloomFixture::k>(
            thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
            fixture.sequenceLength,
            thrust::raw_pointer_cast(fixture.benchData->d_throughputPackedKmers.data())
        );
        fixture.filter->contains(
            fixture.benchData->d_throughputPackedKmers.begin(),
            fixture.benchData->d_throughputPackedKmers.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(fixture.d_output.data()))
        );
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }
    fixture.setCounters(state);
}

void runCucoFprBenchmark(CucoBloomFixture& fixture, bm::State& state) {
    fixture.benchData->ensureFprData();

    fixture.filter->clear();
    fixture.filter->add(
        fixture.benchData->d_fprInsertPackedKmers.begin(),
        fixture.benchData->d_fprInsertPackedKmers.end()
    );
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());

    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        fixture.filter->contains(
            fixture.benchData->d_zeroOverlapPackedKmers.begin(),
            fixture.benchData->d_zeroOverlapPackedKmers.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(fixture.d_output.data()))
        );
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }

    falsePositives = static_cast<uint64_t>(
        thrust::count(fixture.d_output.begin(), fixture.d_output.end(), uint8_t{1})
    );
    fixture.setCounters(state);
    benchmark_common::setFprCounters(state, falsePositives, fixture.numKmers);
}


void runCuckooGpuInsertBenchmark(CuckooGpuFixture& fixture, bm::State& state) {
    for (auto _ : state) {
        fixture.filter->clear();
        BLOOM_CUDA_CALL(cudaDeviceSynchronize());

        fixture.timer.start();
        benchmark_common::gpuEncodePackedKmers<CuckooGpuFixture::k>(
            thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
            fixture.sequenceLength,
            thrust::raw_pointer_cast(fixture.benchData->d_throughputPackedKmers.data())
        );
        fixture.filter->insertMany(fixture.benchData->d_throughputPackedKmers);
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
    }
    fixture.setCounters(state);
}

void runCuckooGpuQueryBenchmark(CuckooGpuFixture& fixture, bm::State& state) {
    fixture.filter->clear();
    benchmark_common::gpuEncodePackedKmers<CuckooGpuFixture::k>(
        thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
        fixture.sequenceLength,
        thrust::raw_pointer_cast(fixture.benchData->d_throughputPackedKmers.data())
    );
    fixture.filter->insertMany(fixture.benchData->d_throughputPackedKmers);
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fixture.timer.start();
        benchmark_common::gpuEncodePackedKmers<CuckooGpuFixture::k>(
            thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
            fixture.sequenceLength,
            thrust::raw_pointer_cast(fixture.benchData->d_throughputPackedKmers.data())
        );
        fixture.filter->containsMany(fixture.benchData->d_throughputPackedKmers, fixture.d_output);
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }
    fixture.setCounters(state);
}

void runCuckooGpuFprBenchmark(CuckooGpuFixture& fixture, bm::State& state) {
    fixture.benchData->ensureFprData();

    fixture.filter->clear();
    fixture.filter->insertMany(fixture.benchData->d_fprInsertPackedKmers);

    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        fixture.filter->containsMany(fixture.benchData->d_zeroOverlapPackedKmers, fixture.d_output);
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }

    falsePositives = static_cast<uint64_t>(
        thrust::count(fixture.d_output.begin(), fixture.d_output.end(), uint8_t{1})
    );
    fixture.setCounters(state);
    benchmark_common::setFprCounters(state, falsePositives, fixture.numKmers);
}


BENCHMARK_DEFINE_F(CucoBloomFixture, Insert)(bm::State& state) {
    runCucoInsertBenchmark(*this, state);
}

BENCHMARK_DEFINE_F(CucoBloomFixture, Query)(bm::State& state) {
    runCucoQueryBenchmark(*this, state);
}

BENCHMARK_DEFINE_F(CucoBloomFixture, FPR)(bm::State& state) {
    runCucoFprBenchmark(*this, state);
}

BENCHMARK_DEFINE_F(CuckooGpuFixture, Insert)(bm::State& state) {
    runCuckooGpuInsertBenchmark(*this, state);
}

BENCHMARK_DEFINE_F(CuckooGpuFixture, Query)(bm::State& state) {
    runCuckooGpuQueryBenchmark(*this, state);
}

BENCHMARK_DEFINE_F(CuckooGpuFixture, FPR)(bm::State& state) {
    runCuckooGpuFprBenchmark(*this, state);
}

#define DEFINE_SUPERBLOOM_INSERT_QUERY_FPR_BENCHMARKS(K, S, M, H) \
    BENCHMARK_DEFINE_SUPERBLOOM_ALL(BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

#define DEFINE_SUPERBLOOM_FPR_ONLY_BENCHMARKS(K, S, M, H) \
    BENCHMARK_DEFINE_SUPERBLOOM_FPR_ONLY(BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR(DEFINE_SUPERBLOOM_INSERT_QUERY_FPR_BENCHMARKS)
SUPERBLOOM_CONFIGS_FPR_ONLY(DEFINE_SUPERBLOOM_FPR_ONLY_BENCHMARKS)

#undef DEFINE_SUPERBLOOM_FPR_ONLY_BENCHMARKS
#undef DEFINE_SUPERBLOOM_INSERT_QUERY_FPR_BENCHMARKS


#define REGISTER_SUPERBLOOM_INSERT_QUERY_FPR_BENCHMARKS(K, S, M, H) \
    BENCHMARK_REGISTER_SUPERBLOOM_ALL(BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

#define REGISTER_SUPERBLOOM_FPR_ONLY_BENCHMARKS(K, S, M, H) \
    BENCHMARK_REGISTER_SUPERBLOOM_FPR_ONLY(BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR(REGISTER_SUPERBLOOM_INSERT_QUERY_FPR_BENCHMARKS)
SUPERBLOOM_CONFIGS_FPR_ONLY(REGISTER_SUPERBLOOM_FPR_ONLY_BENCHMARKS)

#undef REGISTER_SUPERBLOOM_FPR_ONLY_BENCHMARKS
#undef REGISTER_SUPERBLOOM_INSERT_QUERY_FPR_BENCHMARKS

REGISTER_BENCHMARK(CucoBloomFixture, Insert);
REGISTER_BENCHMARK(CucoBloomFixture, Query);
REGISTER_BENCHMARK(CucoBloomFixture, FPR);

REGISTER_BENCHMARK(CuckooGpuFixture, Insert);
REGISTER_BENCHMARK(CuckooGpuFixture, Query);
REGISTER_BENCHMARK(CuckooGpuFixture, FPR);

#undef FOR_EACH_SUPERBLOOM_CONFIG
#undef SUPERBLOOM_CONFIGS_FPR_ONLY
#undef SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR
#undef SUPERBLOOM_FIRST_INSERT_QUERY_FPR_CONFIG

STANDARD_BENCHMARK_MAIN()
