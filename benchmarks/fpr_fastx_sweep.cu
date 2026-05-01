#include <benchmark/benchmark.h>
#include <cuda/__cmath/ceil_div.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <bloom/BloomFilter.cuh>
#include <bloom/device_span.cuh>
#include <bloom/helpers.cuh>
#include <cuco/bloom_filter.cuh>

#include "benchmark_common.cuh"

namespace bm = benchmark;

struct FastxData {
    thrust::device_vector<char> d_insertSequence;
    uint64_t insertKmers = 0;

    thrust::device_vector<char> d_querySequence;
    uint64_t queryKmers = 0;

    // Pre-encoded packed k-mers for Cuco
    thrust::device_vector<uint64_t> d_insertPackedKmers;
    thrust::device_vector<uint64_t> d_queryPackedKmers;
};

static std::unique_ptr<FastxData> g_fastxData;
static std::string g_insertFastxPath;

static constexpr uint64_t kQueryLength = 1'000'000'000ULL;
static constexpr uint64_t kQuerySeed = 0xDEADBEEF;

using CucoBloom = cuco::bloom_filter<uint64_t>;

static void prepareFastxData() {
    if (g_fastxData) {
        return;
    }
    if (g_insertFastxPath.empty()) {
        std::cerr << "Error: --insert-fastx is required" << std::endl;
        std::exit(1);
    }

    g_fastxData = std::make_unique<FastxData>();

    // Read insert FASTX
    std::vector<char> hostInsert = benchmark_common::readFastxConcatenated(g_insertFastxPath);
    if (hostInsert.empty()) {
        std::cerr << "Error: FASTX file is empty or contains no sequences" << std::endl;
        std::exit(1);
    }

    g_fastxData->d_insertSequence.resize(hostInsert.size());
    BLOOM_CUDA_CALL(cudaMemcpy(
        thrust::raw_pointer_cast(g_fastxData->d_insertSequence.data()),
        hostInsert.data(),
        hostInsert.size(),
        cudaMemcpyHostToDevice
    ));
    g_fastxData->insertKmers = hostInsert.size() >= 31 ? hostInsert.size() - 31 + 1 : 0;

    // Generate query sequence on device
    benchmark_common::gpuGenerateDna(g_fastxData->d_querySequence, kQueryLength, kQuerySeed);
    g_fastxData->queryKmers = kQueryLength >= 31 ? kQueryLength - 31 + 1 : 0;

    // Pre-encode packed k-mers for Cuco
    g_fastxData->d_insertPackedKmers.resize(g_fastxData->insertKmers);
    benchmark_common::gpuEncodePackedKmers<31>(
        thrust::raw_pointer_cast(g_fastxData->d_insertSequence.data()),
        hostInsert.size(),
        thrust::raw_pointer_cast(g_fastxData->d_insertPackedKmers.data())
    );

    g_fastxData->d_queryPackedKmers.resize(g_fastxData->queryKmers);
    benchmark_common::gpuEncodePackedKmers<31>(
        thrust::raw_pointer_cast(g_fastxData->d_querySequence.data()),
        kQueryLength,
        thrust::raw_pointer_cast(g_fastxData->d_queryPackedKmers.data())
    );

    BLOOM_CUDA_CALL(cudaDeviceSynchronize());
}

static void setFprFastxCounters(
    bm::State& state,
    uint64_t filterBits,
    uint64_t memoryBytes,
    uint64_t insertKmers,
    uint64_t queryKmers
) {
    state.counters["filter_bits"] = bm::Counter(static_cast<double>(filterBits));
    state.counters["memory_bytes"] =
        bm::Counter(static_cast<double>(memoryBytes), bm::Counter::kDefaults, bm::Counter::kIs1024);
    state.counters["insert_kmers"] = bm::Counter(static_cast<double>(insertKmers));
    state.counters["query_kmers"] = bm::Counter(static_cast<double>(queryKmers));
    state.counters["bits_per_item"] = bm::Counter(
        insertKmers > 0 ? static_cast<double>(filterBits) / static_cast<double>(insertKmers) : 0.0
    );
}

template <typename Config>
class SuperBloomFprFastxFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        prepareFastxData();

        filterBits = static_cast<uint64_t>(state.range(0));
        filter = std::make_unique<bloom::Filter<Config>>(filterBits);
        filterMemory = filter->filterBits() / 8;

        d_output.resize(g_fastxData->queryKmers);
    }

    void TearDown(const bm::State&) override {
        filter.reset();
        d_output.clear();
        d_output.shrink_to_fit();
    }

    uint64_t filterBits = 0;
    uint64_t filterMemory = 0;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<bloom::Filter<Config>> filter;
    benchmark_common::GPUTimer timer;
};

#define SUPERBLOOM_FPR_FASTX_CONFIG_SYMBOL(S) SuperBloom_K31_S##S##_M21_H4_FprFastxConfig
#define SUPERBLOOM_FPR_FASTX_FIXTURE_SYMBOL(S) SuperBloom_K31_S##S##_M21_H4_FprFastxFixture

#define DEFINE_SUPERBLOOM_FPR_FASTX_CONFIG_AND_FIXTURE(S)                           \
    using SUPERBLOOM_FPR_FASTX_CONFIG_SYMBOL(S) = bloom::Config<31, S, 21, 4, 256>; \
    using SUPERBLOOM_FPR_FASTX_FIXTURE_SYMBOL(S) =                                  \
        SuperBloomFprFastxFixture<SUPERBLOOM_FPR_FASTX_CONFIG_SYMBOL(S)>;

#define FOR_EACH_SUPERBLOOM_FPR_FASTX_CONFIG(X) \
    X(20)                                       \
    X(22)                                       \
    X(24)                                       \
    X(26)                                       \
    X(28)                                       \
    X(30)                                       \
    X(31)

FOR_EACH_SUPERBLOOM_FPR_FASTX_CONFIG(DEFINE_SUPERBLOOM_FPR_FASTX_CONFIG_AND_FIXTURE)

#undef DEFINE_SUPERBLOOM_FPR_FASTX_CONFIG_AND_FIXTURE

class CucoBloomFprFastxFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        prepareFastxData();

        filterBits = static_cast<uint64_t>(state.range(0));
        constexpr auto bitsPerBlock =
            CucoBloom::words_per_block * sizeof(typename CucoBloom::word_type) * 8;
        uint64_t blocks = cuda::ceil_div(filterBits, bitsPerBlock);
        if (blocks == 0) {
            blocks = 1;
        }
        filter = std::make_unique<CucoBloom>(blocks);
        filterMemory = filter->block_extent() * CucoBloom::words_per_block *
                       sizeof(typename CucoBloom::word_type);
        actualFilterBits = filterMemory * 8;

        d_output.resize(g_fastxData->queryKmers);
    }

    void TearDown(const bm::State&) override {
        filter.reset();
        d_output.clear();
        d_output.shrink_to_fit();
    }

    uint64_t filterBits = 0;
    uint64_t actualFilterBits = 0;
    uint64_t filterMemory = 0;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<CucoBloom> filter;
    benchmark_common::GPUTimer timer;
};

template <typename Fixture>
void runSuperBloomFprFastxBenchmark(Fixture& fixture, bm::State& state) {
    fixture.filter->clear();
    benchmark::DoNotOptimize(fixture.filter->insertSequenceDevice(
        bloom::device_span<const char>{
            thrust::raw_pointer_cast(g_fastxData->d_insertSequence.data()),
            g_fastxData->d_insertSequence.size()
        }
    ));
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fixture.timer.start();
        fixture.filter->containsSequenceDevice(
            bloom::device_span<const char>{
                thrust::raw_pointer_cast(g_fastxData->d_querySequence.data()),
                g_fastxData->d_querySequence.size()
            },
            bloom::device_span<uint8_t>{
                thrust::raw_pointer_cast(fixture.d_output.data()), fixture.d_output.size()
            }
        );
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }

    const auto falsePositives = static_cast<uint64_t>(
        thrust::count(fixture.d_output.begin(), fixture.d_output.end(), uint8_t{1})
    );

    setFprFastxCounters(
        state,
        fixture.filter->filterBits(),
        fixture.filterMemory,
        g_fastxData->insertKmers,
        g_fastxData->queryKmers
    );
    benchmark_common::setFprCounters(state, falsePositives, g_fastxData->queryKmers);
}

void runCucoFprFastxBenchmark(CucoBloomFprFastxFixture& fixture, bm::State& state) {
    fixture.filter->clear();
    fixture.filter->add(
        g_fastxData->d_insertPackedKmers.begin(), g_fastxData->d_insertPackedKmers.end()
    );
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fixture.timer.start();
        fixture.filter->contains(
            g_fastxData->d_queryPackedKmers.begin(),
            g_fastxData->d_queryPackedKmers.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(fixture.d_output.data()))
        );
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }

    const uint64_t falsePositives = static_cast<uint64_t>(
        thrust::count(fixture.d_output.begin(), fixture.d_output.end(), uint8_t{1})
    );

    setFprFastxCounters(
        state,
        fixture.actualFilterBits,
        fixture.filterMemory,
        g_fastxData->insertKmers,
        g_fastxData->queryKmers
    );
    benchmark_common::setFprCounters(state, falsePositives, g_fastxData->queryKmers);
}

#define DEFINE_SUPERBLOOM_FPR_FASTX_BENCHMARK(S)                    \
    BENCHMARK_DEFINE_F(SUPERBLOOM_FPR_FASTX_FIXTURE_SYMBOL(S), FPR) \
    (bm::State & state) {                                           \
        runSuperBloomFprFastxBenchmark(*this, state);               \
    }

FOR_EACH_SUPERBLOOM_FPR_FASTX_CONFIG(DEFINE_SUPERBLOOM_FPR_FASTX_BENCHMARK)

#undef DEFINE_SUPERBLOOM_FPR_FASTX_BENCHMARK

BENCHMARK_DEFINE_F(CucoBloomFprFastxFixture, FPR)(bm::State& state) {
    runCucoFprFastxBenchmark(*this, state);
}

#define REGISTER_SUPERBLOOM_FPR_FASTX_BENCHMARK(S) \
    REGISTER_BENCHMARK_FPR_FASTX_SWEEP(SUPERBLOOM_FPR_FASTX_FIXTURE_SYMBOL(S), FPR);

FOR_EACH_SUPERBLOOM_FPR_FASTX_CONFIG(REGISTER_SUPERBLOOM_FPR_FASTX_BENCHMARK)

#undef REGISTER_SUPERBLOOM_FPR_FASTX_BENCHMARK

REGISTER_BENCHMARK_FPR_FASTX_SWEEP(CucoBloomFprFastxFixture, FPR);

#undef FOR_EACH_SUPERBLOOM_FPR_FASTX_CONFIG
#undef SUPERBLOOM_FPR_FASTX_FIXTURE_SYMBOL
#undef SUPERBLOOM_FPR_FASTX_CONFIG_SYMBOL

void parseCustomArgs(int argc, char** argv, std::vector<char*>& benchmarkArgv) {
    benchmarkArgv.clear();
    benchmarkArgv.reserve(argc);
    benchmarkArgv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        constexpr const char* fastxPrefix = "--insert-fastx=";
        if (std::strncmp(arg.c_str(), fastxPrefix, std::strlen(fastxPrefix)) == 0) {
            g_insertFastxPath = arg.substr(std::strlen(fastxPrefix));
            continue;
        }
        if (arg == "--insert-fastx") {
            if (i + 1 < argc) {
                ++i;
                g_insertFastxPath = argv[i];
            } else {
                std::cerr << "Missing value for --insert-fastx" << std::endl;
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
