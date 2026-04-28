#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <bloom/BloomFilter.cuh>
#include <bloom/device_span.cuh>
#include <bloom/helpers.cuh>

#include "benchmark_common.cuh"

namespace bm = benchmark;

struct FastxInputData {
    thrust::device_vector<char> d_insertSequence;
    uint64_t insertKmers = 0;

    // Optional user-provided query sequence (used by Query benchmark)
    thrust::device_vector<char> d_querySequence;
    uint64_t queryKmers = 0;

    // Fixed 1B random DNA for FPR benchmark (always independent of insert)
    thrust::device_vector<char> d_fprSequence;
    uint64_t fprKmers = 0;
};

static constexpr uint64_t kFprQueryLength = 1'000'000'000ULL;
static constexpr uint64_t kFprQuerySeed = 0xDEADBEEF;

static std::unique_ptr<FastxInputData> g_fastxData;
static std::string g_insertFastxPath;
static std::string g_queryFastxPath;
static uint64_t g_filterBits = 0;

static std::vector<char> readFastxConcatenated(std::string_view path) {
    auto input = bloom::detail::openFastxFile(path);
    bloom::detail::FastxReader reader(input, path);
    bloom::detail::FastxRecord record;

    std::vector<char> sequence;
    bool firstRecord = true;

    while (reader.nextRecord(record)) {
        if (!firstRecord) {
            sequence.push_back('N');
        }
        firstRecord = false;
        sequence.insert(sequence.end(), record.sequence.begin(), record.sequence.end());
    }

    return sequence;
}

static void prepareFastxData() {
    if (g_fastxData) {
        return;
    }
    if (g_insertFastxPath.empty()) {
        std::cerr << "Error: --insert-fastx is required" << std::endl;
        std::exit(1);
    }

    g_fastxData = std::make_unique<FastxInputData>();

    // Read insert FASTX
    std::vector<char> hostInsert = readFastxConcatenated(g_insertFastxPath);
    if (hostInsert.empty()) {
        std::cerr << "Error: Insert FASTX file is empty or contains no sequences" << std::endl;
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

    // Query sequence (throughput benchmark)
    if (!g_queryFastxPath.empty()) {
        std::vector<char> hostQuery = readFastxConcatenated(g_queryFastxPath);
        if (hostQuery.empty()) {
            std::cerr << "Error: Query FASTX file is empty" << std::endl;
            std::exit(1);
        }
        g_fastxData->d_querySequence.resize(hostQuery.size());
        BLOOM_CUDA_CALL(cudaMemcpy(
            thrust::raw_pointer_cast(g_fastxData->d_querySequence.data()),
            hostQuery.data(),
            hostQuery.size(),
            cudaMemcpyHostToDevice
        ));
        g_fastxData->queryKmers = hostQuery.size() >= 31 ? hostQuery.size() - 31 + 1 : 0;
    } else {
        // GPU-generated random DNA of same length as insert
        benchmark_common::gpuGenerateDna(g_fastxData->d_querySequence, hostInsert.size(), 1337);
        g_fastxData->queryKmers = g_fastxData->insertKmers;
    }

    // FPR sequence: fixed 1B random DNA, independent of insert
    benchmark_common::gpuGenerateDna(g_fastxData->d_fprSequence, kFprQueryLength, kFprQuerySeed);
    g_fastxData->fprKmers = kFprQueryLength >= 31 ? kFprQueryLength - 31 + 1 : 0;

    // Compute filter size: 16 bits per insert k-mer, rounded up to next power-of-two shards
    g_filterBits = cuda::std::bit_ceil(g_fastxData->insertKmers * 16);
    if (g_filterBits == 0) {
        g_filterBits = 256;
    }

    BLOOM_CUDA_CALL(cudaDeviceSynchronize());
}

template <typename Config>
class ShSweepFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& /*state*/) override {
        prepareFastxData();
        filter = std::make_unique<bloom::Filter<Config>>(g_filterBits);
        filterMemory = filter->filterBits() / 8;
        d_output.resize(std::max(g_fastxData->queryKmers, g_fastxData->fprKmers));
    }

    void TearDown(const bm::State&) override {
        filter.reset();
        d_output.clear();
        d_output.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) const {
        state.SetItemsProcessed(
            static_cast<int64_t>(state.iterations() * g_fastxData->insertKmers)
        );
        state.counters["sequence_bases"] =
            benchmark::Counter(static_cast<double>(g_fastxData->d_insertSequence.size()));
        state.counters["memory_bytes"] = benchmark::Counter(
            static_cast<double>(filterMemory),
            benchmark::Counter::kDefaults,
            benchmark::Counter::kIs1024
        );
        state.counters["bits_per_item"] = benchmark::Counter(
            static_cast<double>(filterMemory * 8) / static_cast<double>(g_fastxData->insertKmers),
            benchmark::Counter::kDefaults,
            benchmark::Counter::kIs1024
        );
        state.counters["num_kmers"] =
            benchmark::Counter(static_cast<double>(g_fastxData->insertKmers));
        state.counters["s"] = benchmark::Counter(static_cast<double>(Config::s));
        state.counters["m"] = benchmark::Counter(static_cast<double>(Config::m));
        state.counters["hashes"] = benchmark::Counter(static_cast<double>(Config::hashCount));
        // Always emit these so the CSV contains the column for every operation.
        state.counters["fpr_percentage"] = 0.0;
        state.counters["false_positives"] = 0.0;
    }

    std::unique_ptr<bloom::Filter<Config>> filter;
    uint64_t filterMemory = 0;
    thrust::device_vector<uint8_t> d_output;
    benchmark_common::GPUTimer timer;
};

// Benchmark runners
template <typename Fixture>
void runShSweepInsert(Fixture& fixture, benchmark::State& state) {
    for (auto _ : state) {
        fixture.filter->clear();
        BLOOM_CUDA_CALL(cudaDeviceSynchronize());

        fixture.timer.start();
        benchmark::DoNotOptimize(fixture.filter->insertSequenceDevice(
            bloom::device_span<const char>{
                thrust::raw_pointer_cast(g_fastxData->d_insertSequence.data()),
                g_fastxData->d_insertSequence.size()
            }
        ));
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
    }
    fixture.setCounters(state);
}

template <typename Fixture>
void runShSweepQuery(Fixture& fixture, benchmark::State& state) {
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
    fixture.setCounters(state);
}

template <typename Fixture>
void runShSweepFpr(Fixture& fixture, benchmark::State& state) {
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
                thrust::raw_pointer_cast(g_fastxData->d_fprSequence.data()),
                g_fastxData->d_fprSequence.size()
            },
            bloom::device_span<uint8_t>{
                thrust::raw_pointer_cast(fixture.d_output.data()), g_fastxData->fprKmers
            }
        );
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }

    const auto falsePositives = static_cast<uint64_t>(thrust::count(
        fixture.d_output.begin(),
        fixture.d_output.begin() + static_cast<int64_t>(g_fastxData->fprKmers),
        uint8_t{1}
    ));
    fixture.setCounters(state);
    benchmark_common::setFprCounters(state, falsePositives, g_fastxData->fprKmers);
}

// Macros for config / fixture / benchmark definition and registration
#define PARAM_SWEEP_DEFINE_CONFIG_AND_FIXTURE(K, S, M, H)                                  \
    using BENCHMARK_SUPERBLOOM_CONFIG_SYMBOL(K, S, M, H) = bloom::Config<K, S, M, H, 256>; \
    using BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H) =                                \
        ShSweepFixture<BENCHMARK_SUPERBLOOM_CONFIG_SYMBOL(K, S, M, H)>;

#define PARAM_SWEEP_DEFINE_ALL(FixtureName)                             \
    BENCHMARK_DEFINE_F(FixtureName, Insert)(benchmark::State & state) { \
        runShSweepInsert(*this, state);                                 \
    }                                                                   \
    BENCHMARK_DEFINE_F(FixtureName, Query)(benchmark::State & state) {  \
        runShSweepQuery(*this, state);                                  \
    }                                                                   \
    BENCHMARK_DEFINE_F(FixtureName, FPR)(benchmark::State & state) {    \
        runShSweepFpr(*this, state);                                    \
    }

#define PARAM_SWEEP_BENCHMARK_CONFIG \
    ->Unit(benchmark::kMillisecond)  \
        ->UseManualTime()            \
        ->Iterations(10)             \
        ->Repetitions(5)             \
        ->ReportAggregatesOnly(true)

#define REGISTER_PARAM_SWEEP_BENCHMARK(FixtureName, BenchName) \
    BENCHMARK_REGISTER_F(FixtureName, BenchName)               \
    PARAM_SWEEP_BENCHMARK_CONFIG

#define REGISTER_PARAM_SWEEP_ALL(FixtureName)            \
    REGISTER_PARAM_SWEEP_BENCHMARK(FixtureName, Insert); \
    REGISTER_PARAM_SWEEP_BENCHMARK(FixtureName, Query);  \
    REGISTER_PARAM_SWEEP_BENCHMARK(FixtureName, FPR);

// X-macro helpers: one H-list, reused for every (S,M) pair.
#define PARAM_SWEEP_H_SPARSE(MACRO, K, S, M) \
    MACRO(K, S, M, 4)                        \
    MACRO(K, S, M, 8)                        \
    MACRO(K, S, M, 12)                       \
    MACRO(K, S, M, 16)

#define PARAM_SWEEP_H_DENSE(MACRO, K, S, M) \
    MACRO(K, S, M, 4)                       \
    MACRO(K, S, M, 5)                       \
    MACRO(K, S, M, 6)                       \
    MACRO(K, S, M, 7)                       \
    MACRO(K, S, M, 8)                       \
    MACRO(K, S, M, 9)                       \
    MACRO(K, S, M, 10)                      \
    MACRO(K, S, M, 11)                      \
    MACRO(K, S, M, 12)                      \
    MACRO(K, S, M, 13)                      \
    MACRO(K, S, M, 14)                      \
    MACRO(K, S, M, 15)                      \
    MACRO(K, S, M, 16)

// Sparse 3-D grid (default): SxMxH = 4x4x4 = 64 configs
#define PARAM_SWEEP_APPLY_SPARSE(MACRO)     \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 20, 16) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 20, 21) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 20, 26) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 20, 31) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 24, 16) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 24, 21) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 24, 26) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 24, 31) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 28, 16) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 28, 21) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 28, 26) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 28, 31) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 31, 16) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 31, 21) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 31, 26) \
    PARAM_SWEEP_H_SPARSE(MACRO, 31, 31, 31)

// Dense 2-D grid (PARAM_SWEEP_DENSE): SxH with fixed M=21
#ifdef PARAM_SWEEP_DENSE
    #define PARAM_SWEEP_APPLY_DENSE(MACRO)     \
        PARAM_SWEEP_H_DENSE(MACRO, 31, 20, 21) \
        PARAM_SWEEP_H_DENSE(MACRO, 31, 21, 21) \
        PARAM_SWEEP_H_DENSE(MACRO, 31, 22, 21) \
        PARAM_SWEEP_H_DENSE(MACRO, 31, 23, 21) \
        PARAM_SWEEP_H_DENSE(MACRO, 31, 24, 21) \
        PARAM_SWEEP_H_DENSE(MACRO, 31, 25, 21) \
        PARAM_SWEEP_H_DENSE(MACRO, 31, 26, 21) \
        PARAM_SWEEP_H_DENSE(MACRO, 31, 27, 21) \
        PARAM_SWEEP_H_DENSE(MACRO, 31, 28, 21) \
        PARAM_SWEEP_H_DENSE(MACRO, 31, 29, 21) \
        PARAM_SWEEP_H_DENSE(MACRO, 31, 30, 21) \
        PARAM_SWEEP_H_DENSE(MACRO, 31, 31, 21)
#endif

// Dense 3-D grid (PARAM_SWEEP_DENSE3D): SxMxH = 5x4x5 = 100 configs
#ifdef PARAM_SWEEP_DENSE3D
    #define PARAM_SWEEP_H_DENSE3D(MACRO, K, S, M) \
        MACRO(K, S, M, 4)                         \
        MACRO(K, S, M, 7)                         \
        MACRO(K, S, M, 10)                        \
        MACRO(K, S, M, 13)                        \
        MACRO(K, S, M, 16)

    #define PARAM_SWEEP_APPLY_DENSE3D(MACRO)     \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 20, 16) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 20, 21) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 20, 26) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 20, 31) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 23, 16) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 23, 21) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 23, 26) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 23, 31) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 26, 16) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 26, 21) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 26, 26) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 26, 31) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 28, 16) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 28, 21) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 28, 26) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 28, 31) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 31, 16) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 31, 21) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 31, 26) \
        PARAM_SWEEP_H_DENSE3D(MACRO, 31, 31, 31)
#endif

// Instantiate configs, fixtures, benchmarks, and registrations
#define PARAM_SWEEP_DEFINE_CONFIG_AND_FIXTURE_WRAPPER(K, S, M, H) \
    PARAM_SWEEP_DEFINE_CONFIG_AND_FIXTURE(K, S, M, H)

#define PARAM_SWEEP_DEFINE_ALL_WRAPPER(K, S, M, H) \
    PARAM_SWEEP_DEFINE_ALL(BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

#define PARAM_SWEEP_REGISTER_ALL_WRAPPER(K, S, M, H) \
    REGISTER_PARAM_SWEEP_ALL(BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

#if defined(PARAM_SWEEP_DENSE3D)
PARAM_SWEEP_APPLY_DENSE3D(PARAM_SWEEP_DEFINE_CONFIG_AND_FIXTURE_WRAPPER)
PARAM_SWEEP_APPLY_DENSE3D(PARAM_SWEEP_DEFINE_ALL_WRAPPER)
PARAM_SWEEP_APPLY_DENSE3D(PARAM_SWEEP_REGISTER_ALL_WRAPPER)
#elif defined(PARAM_SWEEP_DENSE)
PARAM_SWEEP_APPLY_DENSE(PARAM_SWEEP_DEFINE_CONFIG_AND_FIXTURE_WRAPPER)
PARAM_SWEEP_APPLY_DENSE(PARAM_SWEEP_DEFINE_ALL_WRAPPER)
PARAM_SWEEP_APPLY_DENSE(PARAM_SWEEP_REGISTER_ALL_WRAPPER)
#else
PARAM_SWEEP_APPLY_SPARSE(PARAM_SWEEP_DEFINE_CONFIG_AND_FIXTURE_WRAPPER)
PARAM_SWEEP_APPLY_SPARSE(PARAM_SWEEP_DEFINE_ALL_WRAPPER)
PARAM_SWEEP_APPLY_SPARSE(PARAM_SWEEP_REGISTER_ALL_WRAPPER)
#endif

static void parseCustomArgs(int argc, char** argv, std::vector<char*>& benchmarkArgv) {
    benchmarkArgv.clear();
    benchmarkArgv.reserve(argc);
    benchmarkArgv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        constexpr const char* insertPrefix = "--insert-fastx=";
        if (std::strncmp(arg.c_str(), insertPrefix, std::strlen(insertPrefix)) == 0) {
            g_insertFastxPath = arg.substr(std::strlen(insertPrefix));
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

        constexpr const char* queryPrefix = "--query-fastx=";
        if (std::strncmp(arg.c_str(), queryPrefix, std::strlen(queryPrefix)) == 0) {
            g_queryFastxPath = arg.substr(std::strlen(queryPrefix));
            continue;
        }
        if (arg == "--query-fastx") {
            if (i + 1 < argc) {
                ++i;
                g_queryFastxPath = argv[i];
            } else {
                std::cerr << "Missing value for --query-fastx" << std::endl;
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
