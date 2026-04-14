#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <cuda/std/bit>
#include <cuda/std/span>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <bloom/BloomFilter.cuh>
#include <bloom/device_span.cuh>
#include <bloom/helpers.cuh>
#include <cuco/bloom_filter.cuh>

#include "benchmark_common.cuh"

namespace bm = benchmark;

enum class InputMode {
    Packed,
    Sequence
};

static InputMode g_inputMode = InputMode::Packed;

using CucoBloom = cuco::bloom_filter<uint64_t>;

// It's K - S - M - H

#define SUPERBLOOM_FIRST_INSERT_QUERY_FPR_CONFIG(X) X(31, 24, 21, 4)

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
void gpuEncodePackedKmers(
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

void gpuGenerateDna(thrust::device_vector<char>& d_seq, uint64_t length, uint32_t seed = 42) {
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

void gpuGeneratePackedKmers(
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
struct BenchmarkData {
    uint64_t sequenceLength{};
    uint64_t numKmers{};

    thrust::device_vector<char> d_throughputSequence;
    thrust::device_vector<uint64_t> d_throughputPackedKmers;

    thrust::device_vector<char> d_fprInsertSequence;
    thrust::device_vector<uint64_t> d_fprInsertPackedKmers;
    thrust::device_vector<char> d_zeroOverlapSequence;
    thrust::device_vector<uint64_t> d_zeroOverlapPackedKmers;
    bool fprDataReady = false;

    void generateThroughputData() {
        gpuGenerateDna(d_throughputSequence, sequenceLength, 42);
        numKmers = sequenceLength >= K ? sequenceLength - K + 1 : 0;
        gpuGeneratePackedKmers(d_throughputPackedKmers, numKmers, 42);
    }

    void ensureFprData() const {
        if (fprDataReady) {
            return;
        }
        const_cast<BenchmarkData*>(this)->generateFprData();
    }

   private:
    void generateFprData() {
        gpuGenerateDna(d_fprInsertSequence, sequenceLength, 7);
        const uint64_t fprNumKmers = sequenceLength >= K ? sequenceLength - K + 1 : 0;
        d_fprInsertPackedKmers.resize(fprNumKmers);
        gpuEncodePackedKmers<K>(
            thrust::raw_pointer_cast(d_fprInsertSequence.data()),
            sequenceLength,
            thrust::raw_pointer_cast(d_fprInsertPackedKmers.data())
        );

        gpuGenerateDna(d_zeroOverlapSequence, sequenceLength, 1337);
        d_zeroOverlapPackedKmers.resize(fprNumKmers);
        gpuEncodePackedKmers<K>(
            thrust::raw_pointer_cast(d_zeroOverlapSequence.data()),
            sequenceLength,
            thrust::raw_pointer_cast(d_zeroOverlapPackedKmers.data())
        );

        CUDA_CALL(cudaDeviceSynchronize());
        fprDataReady = true;
    }
};

template <uint64_t K>
BenchmarkData<K>& getBenchmarkData(uint64_t length) {
    static std::unordered_map<uint64_t, BenchmarkData<K>> cache;

    auto it = cache.find(length);
    if (it != cache.end()) {
        return it->second;
    }

    // Evict all old entries to avoid accumulating GPU memory across sizes.
    cache.clear();

    BenchmarkData<K> data;
    data.sequenceLength = length;
    data.generateThroughputData();
    CUDA_CALL(cudaDeviceSynchronize());

    return cache.emplace(length, std::move(data)).first->second;
}

uint64_t cucoNumBlocks(uint64_t numItems) {
    constexpr auto bitsPerWord = sizeof(typename CucoBloom::word_type) * 8;
    return bloom::detail::divUp(numItems * kBitsPerItem, CucoBloom::words_per_block * bitsPerWord);
}

void setBenchmarkCounters(
    bm::State& state,
    uint64_t memoryBytes,
    uint64_t sequenceLength,
    uint64_t numKmers
) {
    benchmark_common::setCommonCounters(state, memoryBytes, numKmers, sequenceLength);
    state.counters["num_kmers"] = bm::Counter(static_cast<double>(numKmers));
    state.counters["packed_input"] = bm::Counter(g_inputMode == InputMode::Packed ? 1.0 : 0.0);
    state.counters["sequence_input"] = bm::Counter(g_inputMode == InputMode::Sequence ? 1.0 : 0.0);
}

template <typename Config>
class SuperBloomFixtureBase : public bm::Fixture {
   public:
    static constexpr uint64_t k = Config::k;

    void setupCommon(const bm::State& state) {
        sequenceLength = static_cast<uint64_t>(state.range(0));
        benchData = &getBenchmarkData<Config::k>(sequenceLength);

        numKmers = benchData->numKmers;
        numSmers = sequenceLength - Config::s + 1;

        const uint64_t requestedFilterBits = cuda::std::bit_ceil(numKmers * kBitsPerItem);
        filter = std::make_unique<bloom::Filter<Config>>(requestedFilterBits);
        filterMemory = filter->filterBits() / 8;
        d_output.resize(numKmers);
    }

    void tearDownCommon() {
        filter.reset();
        benchData = nullptr;
        d_output.clear();
        d_output.shrink_to_fit();
    }

    void setCounters(bm::State& state) const {
        setBenchmarkCounters(state, filterMemory, sequenceLength, numKmers);
        state.counters["s"] = bm::Counter(static_cast<double>(Config::s));
        state.counters["hashes"] = bm::Counter(static_cast<double>(Config::hashCount));
    }

    uint64_t sequenceLength{};
    uint64_t numKmers{};
    uint64_t numSmers{};
    uint64_t filterMemory{};
    BenchmarkData<Config::k>* benchData{};
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<bloom::Filter<Config>> filter;
    benchmark_common::GPUTimer timer;
};

template <typename Config>
class SuperBloomConfigFixture : public SuperBloomFixtureBase<Config> {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        this->setupCommon(state);
    }

    void TearDown(const bm::State&) override {
        this->tearDownCommon();
    }
};

#define SUPERBLOOM_CONFIG_SYMBOL(K, S, M, H) SuperBloom_K##K##_S##S##_M##M##_H##H##_Config
#define SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H) SuperBloom_K##K##_S##S##_M##M##_H##H##_Fixture

#define DEFINE_SUPERBLOOM_CONFIG_AND_FIXTURE(K, S, M, H)                         \
    using SUPERBLOOM_CONFIG_SYMBOL(K, S, M, H) = bloom::Config<K, S, M, H, 256>; \
    using SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H) =                                \
        SuperBloomConfigFixture<SUPERBLOOM_CONFIG_SYMBOL(K, S, M, H)>;

FOR_EACH_SUPERBLOOM_CONFIG(DEFINE_SUPERBLOOM_CONFIG_AND_FIXTURE)

#undef DEFINE_SUPERBLOOM_CONFIG_AND_FIXTURE

#define DEFINE_CUCO_REFERENCE_CONFIG(K, S, M, H) \
    using CucoReferenceConfig = SUPERBLOOM_CONFIG_SYMBOL(K, S, M, H);

SUPERBLOOM_FIRST_INSERT_QUERY_FPR_CONFIG(DEFINE_CUCO_REFERENCE_CONFIG)

#undef DEFINE_CUCO_REFERENCE_CONFIG

class CucoBloomFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    static constexpr uint64_t k = CucoReferenceConfig::k;

    void SetUp(const bm::State& state) override {
        sequenceLength = static_cast<uint64_t>(state.range(0));
        benchData = &getBenchmarkData<CucoReferenceConfig::k>(sequenceLength);
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
        setBenchmarkCounters(state, filterMemory, sequenceLength, numKmers);
    }

    uint64_t sequenceLength{};
    uint64_t numKmers{};
    uint64_t filterMemory{};
    BenchmarkData<CucoReferenceConfig::k>* benchData{};
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<CucoBloom> filter;
    benchmark_common::GPUTimer timer;
};

void setFprCounters(bm::State& state, uint64_t falsePositives, uint64_t numKmers) {
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["fpr_percentage"] =
        bm::Counter(100.0 * static_cast<double>(falsePositives) / static_cast<double>(numKmers));
}

template <typename Fixture>
void runSuperBloomInsertBenchmark(Fixture& fixture, bm::State& state) {
    for (auto _ : state) {
        fixture.filter->clear();
        CUDA_CALL(cudaDeviceSynchronize());

        fixture.timer.start();
        if (g_inputMode == InputMode::Packed) {
            benchmark::DoNotOptimize(fixture.filter->insertPackedKmersDevice(
                bloom::device_span<const uint64_t>{
                    thrust::raw_pointer_cast(fixture.benchData->d_throughputPackedKmers.data()),
                    fixture.numKmers
                }
            ));
        } else {
            benchmark::DoNotOptimize(fixture.filter->insertSequenceDevice(
                bloom::device_span<const char>{
                    thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
                    fixture.sequenceLength
                }
            ));
        }
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
    }
    fixture.setCounters(state);
}

template <typename Fixture>
void runSuperBloomQueryBenchmark(Fixture& fixture, bm::State& state) {
    fixture.filter->clear();
    if (g_inputMode == InputMode::Packed) {
        benchmark::DoNotOptimize(fixture.filter->insertPackedKmersDevice(
            bloom::device_span<const uint64_t>{
                thrust::raw_pointer_cast(fixture.benchData->d_throughputPackedKmers.data()),
                fixture.numKmers
            }
        ));
    } else {
        benchmark::DoNotOptimize(fixture.filter->insertSequenceDevice(
            bloom::device_span<const char>{
                thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
                fixture.sequenceLength
            }
        ));
    }
    CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fixture.timer.start();
        if (g_inputMode == InputMode::Packed) {
            fixture.filter->containsPackedKmersDevice(
                bloom::device_span<const uint64_t>{
                    thrust::raw_pointer_cast(fixture.benchData->d_throughputPackedKmers.data()),
                    fixture.numKmers
                },
                bloom::device_span<uint8_t>{
                    thrust::raw_pointer_cast(fixture.d_output.data()), fixture.d_output.size()
                }
            );
        } else {
            fixture.filter->containsSequenceDevice(
                bloom::device_span<const char>{
                    thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
                    fixture.sequenceLength
                },
                bloom::device_span<uint8_t>{
                    thrust::raw_pointer_cast(fixture.d_output.data()), fixture.d_output.size()
                }
            );
        }
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }
    fixture.setCounters(state);
}

template <typename Fixture>
void runSuperBloomFprBenchmark(Fixture& fixture, bm::State& state) {
    fixture.benchData->ensureFprData();

    fixture.filter->clear();
    if (g_inputMode == InputMode::Packed) {
        benchmark::DoNotOptimize(fixture.filter->insertPackedKmersDevice(
            bloom::device_span<const uint64_t>{
                thrust::raw_pointer_cast(fixture.benchData->d_fprInsertPackedKmers.data()),
                fixture.numKmers
            }
        ));
    } else {
        benchmark::DoNotOptimize(fixture.filter->insertSequenceDevice(
            bloom::device_span<const char>{
                thrust::raw_pointer_cast(fixture.benchData->d_fprInsertSequence.data()),
                fixture.sequenceLength
            }
        ));
    }
    CUDA_CALL(cudaDeviceSynchronize());

    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        if (g_inputMode == InputMode::Packed) {
            fixture.filter->containsPackedKmersDevice(
                bloom::device_span<const uint64_t>{
                    thrust::raw_pointer_cast(fixture.benchData->d_zeroOverlapPackedKmers.data()),
                    fixture.numKmers
                },
                bloom::device_span<uint8_t>{
                    thrust::raw_pointer_cast(fixture.d_output.data()), fixture.d_output.size()
                }
            );
        } else {
            fixture.filter->containsSequenceDevice(
                bloom::device_span<const char>{
                    thrust::raw_pointer_cast(fixture.benchData->d_zeroOverlapSequence.data()),
                    fixture.sequenceLength
                },
                bloom::device_span<uint8_t>{
                    thrust::raw_pointer_cast(fixture.d_output.data()), fixture.d_output.size()
                }
            );
        }
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }

    falsePositives = static_cast<uint64_t>(
        thrust::count(fixture.d_output.begin(), fixture.d_output.end(), uint8_t{1})
    );
    fixture.setCounters(state);
    setFprCounters(state, falsePositives, fixture.numKmers);
}

void runCucoInsertBenchmark(CucoBloomFixture& fixture, bm::State& state) {
    for (auto _ : state) {
        fixture.filter->clear();
        CUDA_CALL(cudaDeviceSynchronize());

        fixture.timer.start();
        if (g_inputMode == InputMode::Packed) {
            fixture.filter->add(
                fixture.benchData->d_throughputPackedKmers.begin(),
                fixture.benchData->d_throughputPackedKmers.end()
            );
        } else {
            gpuEncodePackedKmers<CucoBloomFixture::k>(
                thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
                fixture.sequenceLength,
                thrust::raw_pointer_cast(fixture.benchData->d_throughputPackedKmers.data())
            );
            fixture.filter->add(
                fixture.benchData->d_throughputPackedKmers.begin(),
                fixture.benchData->d_throughputPackedKmers.end()
            );
        }
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
    }
    fixture.setCounters(state);
}

void runCucoQueryBenchmark(CucoBloomFixture& fixture, bm::State& state) {
    fixture.filter->clear();
    if (g_inputMode == InputMode::Packed) {
        fixture.filter->add(
            fixture.benchData->d_throughputPackedKmers.begin(),
            fixture.benchData->d_throughputPackedKmers.end()
        );
    } else {
        gpuEncodePackedKmers<CucoBloomFixture::k>(
            thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
            fixture.sequenceLength,
            thrust::raw_pointer_cast(fixture.benchData->d_throughputPackedKmers.data())
        );
        fixture.filter->add(
            fixture.benchData->d_throughputPackedKmers.begin(),
            fixture.benchData->d_throughputPackedKmers.end()
        );
    }
    CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fixture.timer.start();
        if (g_inputMode == InputMode::Sequence) {
            gpuEncodePackedKmers<CucoBloomFixture::k>(
                thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
                fixture.sequenceLength,
                thrust::raw_pointer_cast(fixture.benchData->d_throughputPackedKmers.data())
            );
        }
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
    CUDA_CALL(cudaDeviceSynchronize());

    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        if (g_inputMode == InputMode::Sequence) {
            gpuEncodePackedKmers<CucoBloomFixture::k>(
                thrust::raw_pointer_cast(fixture.benchData->d_zeroOverlapSequence.data()),
                fixture.sequenceLength,
                thrust::raw_pointer_cast(fixture.benchData->d_zeroOverlapPackedKmers.data())
            );
        }
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
    setFprCounters(state, falsePositives, fixture.numKmers);
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

#define DEFINE_SUPERBLOOM_BENCHMARK_SET_INSERT_QUERY_FPR(FixtureName) \
    BENCHMARK_DEFINE_F(FixtureName, Insert)(bm::State & state) {      \
        runSuperBloomInsertBenchmark(*this, state);                   \
    }                                                                 \
    BENCHMARK_DEFINE_F(FixtureName, Query)(bm::State & state) {       \
        runSuperBloomQueryBenchmark(*this, state);                    \
    }                                                                 \
    BENCHMARK_DEFINE_F(FixtureName, FPR)(bm::State & state) {         \
        runSuperBloomFprBenchmark(*this, state);                      \
    }

#define DEFINE_SUPERBLOOM_BENCHMARK_SET_FPR_ONLY(FixtureName) \
    BENCHMARK_DEFINE_F(FixtureName, FPR)(bm::State & state) { \
        runSuperBloomFprBenchmark(*this, state);              \
    }

#define DEFINE_SUPERBLOOM_INSERT_QUERY_FPR_BENCHMARKS(K, S, M, H) \
    DEFINE_SUPERBLOOM_BENCHMARK_SET_INSERT_QUERY_FPR(SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

#define DEFINE_SUPERBLOOM_FPR_ONLY_BENCHMARKS(K, S, M, H) \
    DEFINE_SUPERBLOOM_BENCHMARK_SET_FPR_ONLY(SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR(DEFINE_SUPERBLOOM_INSERT_QUERY_FPR_BENCHMARKS)
SUPERBLOOM_CONFIGS_FPR_ONLY(DEFINE_SUPERBLOOM_FPR_ONLY_BENCHMARKS)

#undef DEFINE_SUPERBLOOM_FPR_ONLY_BENCHMARKS
#undef DEFINE_SUPERBLOOM_INSERT_QUERY_FPR_BENCHMARKS
#undef DEFINE_SUPERBLOOM_BENCHMARK_SET_FPR_ONLY
#undef DEFINE_SUPERBLOOM_BENCHMARK_SET_INSERT_QUERY_FPR

#define REGISTER_SUPERBLOOM_BENCHMARK_SET_INSERT_QUERY_FPR(FixtureName) \
    REGISTER_BENCHMARK(FixtureName, Insert);                            \
    REGISTER_BENCHMARK(FixtureName, Query);                             \
    REGISTER_BENCHMARK(FixtureName, FPR);

#define REGISTER_SUPERBLOOM_BENCHMARK_SET_FPR_ONLY(FixtureName) \
    REGISTER_BENCHMARK(FixtureName, FPR);

#define REGISTER_SUPERBLOOM_INSERT_QUERY_FPR_BENCHMARKS(K, S, M, H) \
    REGISTER_SUPERBLOOM_BENCHMARK_SET_INSERT_QUERY_FPR(SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

#define REGISTER_SUPERBLOOM_FPR_ONLY_BENCHMARKS(K, S, M, H) \
    REGISTER_SUPERBLOOM_BENCHMARK_SET_FPR_ONLY(SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR(REGISTER_SUPERBLOOM_INSERT_QUERY_FPR_BENCHMARKS)
SUPERBLOOM_CONFIGS_FPR_ONLY(REGISTER_SUPERBLOOM_FPR_ONLY_BENCHMARKS)

#undef REGISTER_SUPERBLOOM_FPR_ONLY_BENCHMARKS
#undef REGISTER_SUPERBLOOM_INSERT_QUERY_FPR_BENCHMARKS
#undef REGISTER_SUPERBLOOM_BENCHMARK_SET_FPR_ONLY
#undef REGISTER_SUPERBLOOM_BENCHMARK_SET_INSERT_QUERY_FPR

REGISTER_BENCHMARK(CucoBloomFixture, Insert);
REGISTER_BENCHMARK(CucoBloomFixture, Query);
REGISTER_BENCHMARK(CucoBloomFixture, FPR);

#undef FOR_EACH_SUPERBLOOM_CONFIG
#undef SUPERBLOOM_CONFIGS_FPR_ONLY
#undef SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR
#undef SUPERBLOOM_FIRST_INSERT_QUERY_FPR_CONFIG
#undef SUPERBLOOM_FIXTURE_SYMBOL
#undef SUPERBLOOM_CONFIG_SYMBOL

void parseCustomArgs(int argc, char** argv, std::vector<char*>& benchmarkArgv) {
    benchmarkArgv.clear();
    benchmarkArgv.reserve(argc);
    benchmarkArgv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--input=packed") {
            g_inputMode = InputMode::Packed;
            continue;
        }
        if (arg == "--input=sequence") {
            g_inputMode = InputMode::Sequence;
            continue;
        }
        constexpr const char* inputPrefix = "--input=";
        if (std::strncmp(arg.c_str(), inputPrefix, std::strlen(inputPrefix)) == 0) {
            std::string value = arg.substr(std::strlen(inputPrefix));
            if (value == "packed") {
                g_inputMode = InputMode::Packed;
            } else if (value == "sequence") {
                g_inputMode = InputMode::Sequence;
            } else {
                std::cerr << "Unknown input mode: " << value << " (use 'packed' or 'sequence')"
                          << std::endl;
                std::exit(1);
            }
            continue;
        }
        if (arg == "--input") {
            if (i + 1 < argc) {
                ++i;
                std::string value = argv[i];
                if (value == "packed") {
                    g_inputMode = InputMode::Packed;
                } else if (value == "sequence") {
                    g_inputMode = InputMode::Sequence;
                } else {
                    std::cerr << "Unknown input mode: " << value << " (use 'packed' or 'sequence')"
                              << std::endl;
                    std::exit(1);
                }
            } else {
                std::cerr << "Missing value for --input" << std::endl;
                std::exit(1);
            }
            continue;
        }

        benchmarkArgv.push_back(argv[i]);
    }
}

int main(int argc, char** argv) {
    std::vector<char*> benchmarkArgv;
    parseCustomArgs(argc, argv, benchmarkArgv);

    int benchmarkArgc = static_cast<int>(benchmarkArgv.size());
    ::benchmark::Initialize(&benchmarkArgc, benchmarkArgv.data());
    if (::benchmark::ReportUnrecognizedArguments(benchmarkArgc, benchmarkArgv.data())) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    fflush(stdout);
    std::_Exit(0);
}
