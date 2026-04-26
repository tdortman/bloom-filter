#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <cstdint>
#include <cstdio>

#include <bloom/BloomFilter.cuh>
#include <bloom/helpers.cuh>

namespace benchmark_common {

class GPUTimer {
   public:
    GPUTimer() {
        CUDA_CALL(cudaEventCreate(&start_));
        CUDA_CALL(cudaEventCreate(&stop_));
    }

    ~GPUTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    GPUTimer(const GPUTimer&) = delete;
    GPUTimer& operator=(const GPUTimer&) = delete;

    void start(cudaStream_t stream = {}) {
        CUDA_CALL(cudaEventRecord(start_, stream));
    }

    [[nodiscard]] double elapsed(cudaStream_t stream = {}) {
        CUDA_CALL(cudaEventRecord(stop_, stream));
        CUDA_CALL(cudaEventSynchronize(stop_));

        float milliseconds = 0.0f;
        CUDA_CALL(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return static_cast<double>(milliseconds) / 1000.0;
    }

   private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

inline void setCommonCounters(
    benchmark::State& state,
    uint64_t memoryBytes,
    uint64_t itemsProcessed,
    uint64_t sequenceBases
) {
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * itemsProcessed));
    state.counters["sequence_bases"] = benchmark::Counter(static_cast<double>(sequenceBases));
    state.counters["memory_bytes"] = benchmark::Counter(
        static_cast<double>(memoryBytes), benchmark::Counter::kDefaults, benchmark::Counter::kIs1024
    );
    state.counters["bits_per_item"] = benchmark::Counter(
        static_cast<double>(memoryBytes * 8) / static_cast<double>(itemsProcessed),
        benchmark::Counter::kDefaults,
        benchmark::Counter::kIs1024
    );
    state.counters["fpr_percentage"] = 0.0;
    state.counters["false_positives"] = 0.0;
}

inline void setFprCounters(benchmark::State& state, uint64_t falsePositives, uint64_t numKmers) {
    state.counters["false_positives"] = benchmark::Counter(static_cast<double>(falsePositives));
    state.counters["fpr_percentage"] = benchmark::Counter(
        100.0 * static_cast<double>(falsePositives) / static_cast<double>(numKmers)
    );
}

inline void
gpuGenerateDna(thrust::device_vector<char>& d_seq, uint64_t length, uint32_t seed = 42) {
    d_seq.resize(length);
    thrust::transform(
        thrust::counting_iterator<uint64_t>(0),
        thrust::counting_iterator<uint64_t>(length),
        d_seq.begin(),
        [seed] __device__(uint64_t idx) {
            thrust::default_random_engine rng(seed);
            thrust::uniform_int_distribution<uint32_t> dist(0, 3);
            rng.discard(idx);
            static constexpr char bases[] = {'A', 'C', 'G', 'T'};
            return bases[dist(rng)];
        }
    );
}

inline void gpuGeneratePackedKmers(
    thrust::device_vector<uint64_t>& d_kmers,
    uint64_t count,
    uint32_t seed = 42
) {
    d_kmers.resize(count);
    thrust::transform(
        thrust::counting_iterator<uint64_t>(0),
        thrust::counting_iterator<uint64_t>(count),
        d_kmers.begin(),
        [seed] __device__(uint64_t idx) {
            thrust::default_random_engine rng(seed);
            thrust::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
            rng.discard(idx);
            return dist(rng);
        }
    );
}

template <uint64_t K>
__global__ void encodePackedKmersKernel(const char* sequence, uint64_t numKmers, uint64_t* output) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numKmers) {
        return;
    }

    uint64_t packed = 0;
    for (uint64_t i = 0; i < K; ++i) {
        const uint8_t encoded = bloom::detail::encodeBase(static_cast<uint8_t>(sequence[idx + i]));
        packed = (packed << 2) | static_cast<uint64_t>(encoded & 0x3);
    }
    output[idx] = packed;
}

template <uint64_t K>
inline void gpuEncodePackedKmers(
    const char* d_sequence,
    uint64_t sequenceLength,
    uint64_t* d_output,
    cudaStream_t stream = {}
) {
    const uint64_t numKmers = sequenceLength >= K ? sequenceLength - K + 1 : 0;
    if (numKmers == 0) {
        return;
    }
    constexpr uint64_t blockSize = 256;
    const uint64_t gridSize = bloom::detail::divUp(numKmers, blockSize);
    encodePackedKmersKernel<K><<<gridSize, blockSize, 0, stream>>>(d_sequence, numKmers, d_output);
}

}  // namespace benchmark_common

#define BENCHMARK_CONFIG                \
    ->RangeMultiplier(2)                \
        ->Range(1 << 16, 1ULL << 28)    \
        ->Unit(benchmark::kMillisecond) \
        ->UseManualTime()               \
        ->Iterations(10)                \
        ->Repetitions(5)                \
        ->ReportAggregatesOnly(true)

#define BENCHMARK_CONFIG_FPR_FASTX_SWEEP \
    ->RangeMultiplier(2)                 \
        ->Range(1ULL << 22, 1ULL << 32)  \
        ->Unit(benchmark::kMillisecond)  \
        ->UseManualTime()                \
        ->Iterations(1)                  \
        ->Repetitions(1)                 \
        ->ReportAggregatesOnly(true)

#define REGISTER_BENCHMARK(FixtureName, BenchName) \
    BENCHMARK_REGISTER_F(FixtureName, BenchName)   \
    BENCHMARK_CONFIG

#define REGISTER_BENCHMARK_FPR_FASTX_SWEEP(FixtureName, BenchName) \
    BENCHMARK_REGISTER_F(FixtureName, BenchName)                   \
    BENCHMARK_CONFIG_FPR_FASTX_SWEEP

#define STANDARD_BENCHMARK_MAIN()                                   \
    int main(int argc, char** argv) {                               \
        ::benchmark::Initialize(&argc, argv);                       \
        if (::benchmark::ReportUnrecognizedArguments(argc, argv)) { \
            return 1;                                               \
        }                                                           \
        ::benchmark::RunSpecifiedBenchmarks();                      \
        ::benchmark::Shutdown();                                    \
        fflush(stdout);                                             \
        std::_Exit(0);                                              \
    }
