#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <cstdio>

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

}  // namespace benchmark_common

#define BENCHMARK_CONFIG                \
    ->RangeMultiplier(2)                \
        ->Range(1 << 16, 1ULL << 28)    \
        ->Unit(benchmark::kMillisecond) \
        ->UseManualTime()               \
        ->Iterations(10)                \
        ->Repetitions(5)                \
        ->ReportAggregatesOnly(true)

#define REGISTER_BENCHMARK(FixtureName, BenchName) \
    BENCHMARK_REGISTER_F(FixtureName, BenchName)   \
    BENCHMARK_CONFIG

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
