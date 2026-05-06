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

#ifndef PARAM_SWEEP_K
    #define PARAM_SWEEP_K 31
#endif
#ifndef PARAM_SWEEP_ALPHABET
    #define PARAM_SWEEP_ALPHABET bloom::DnaAlphabet
#endif

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
    bloom::detail::FastxReader reader(*input, path);
    bloom::detail::FastxRecord record;

    std::vector<char> sequence;
    bool firstRecord = true;

    while (reader.nextRecord(record)) {
        if (!firstRecord) {
            sequence.push_back(static_cast<char>(PARAM_SWEEP_ALPHABET::separator));
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
    g_fastxData->insertKmers =
        hostInsert.size() >= PARAM_SWEEP_K ? hostInsert.size() - PARAM_SWEEP_K + 1 : 0;

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
        g_fastxData->queryKmers =
            hostQuery.size() >= PARAM_SWEEP_K ? hostQuery.size() - PARAM_SWEEP_K + 1 : 0;
    } else {
        // GPU-generated random sequence of same length as insert
#ifdef PARAM_SWEEP_PROTEIN
        benchmark_common::gpuGenerateProtein(g_fastxData->d_querySequence, hostInsert.size(), 1337);
#else
        benchmark_common::gpuGenerateDna(g_fastxData->d_querySequence, hostInsert.size(), 1337);
#endif
        g_fastxData->queryKmers = g_fastxData->insertKmers;
    }

    // FPR sequence: fixed 1B random sequence, independent of insert
#ifdef PARAM_SWEEP_PROTEIN
    benchmark_common::gpuGenerateProtein(
        g_fastxData->d_fprSequence, kFprQueryLength, kFprQuerySeed
    );
#else
    benchmark_common::gpuGenerateDna(g_fastxData->d_fprSequence, kFprQueryLength, kFprQuerySeed);
#endif
    g_fastxData->fprKmers =
        kFprQueryLength >= PARAM_SWEEP_K ? kFprQueryLength - PARAM_SWEEP_K + 1 : 0;

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
#define PARAM_SWEEP_DEFINE_CONFIG_AND_FIXTURE(K, S, M, H)     \
    using BENCHMARK_SUPERBLOOM_CONFIG_SYMBOL(K, S, M, H) =    \
        bloom::Config<K, S, M, H, 256, PARAM_SWEEP_ALPHABET>; \
    using BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H) =   \
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

// H-list: 4 hash values for dense coverage.
#define PARAM_SWEEP_H_DEFAULT(MACRO, K, S, M) \
    MACRO(K, S, M, 4)                         \
    MACRO(K, S, M, 8)                         \
    MACRO(K, S, M, 12)                        \
    MACRO(K, S, M, 16)

// 3-D parameter grid:  SxMxH = 16x13x7 = 1456 configs
// S values: {16..31} (16 values, every integer)
// M values: {8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31} (13 values, step 2)
// H values: {4, 8, 12, 16} (4 values)
// Split into 208 groups (one per S,M pair), each with 4 configs.
// Group index = s_idx * 13 + m_idx

#ifndef PARAM_SWEEP_PROTEIN

    // S=16 (groups 0-12)
    #define PARAM_SWEEP_SM_0(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 16, 8)
    #define PARAM_SWEEP_SM_1(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 16, 10)
    #define PARAM_SWEEP_SM_2(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 16, 12)
    #define PARAM_SWEEP_SM_3(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 16, 14)
    #define PARAM_SWEEP_SM_4(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 16, 16)
    #define PARAM_SWEEP_SM_5(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 16, 18)
    #define PARAM_SWEEP_SM_6(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 16, 20)
    #define PARAM_SWEEP_SM_7(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 16, 22)
    #define PARAM_SWEEP_SM_8(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 16, 24)
    #define PARAM_SWEEP_SM_9(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 16, 26)
    #define PARAM_SWEEP_SM_10(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 16, 28)
    #define PARAM_SWEEP_SM_11(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 16, 30)
    #define PARAM_SWEEP_SM_12(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 16, 31)

    // S=17 (groups 13-25)
    #define PARAM_SWEEP_SM_13(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 17, 8)
    #define PARAM_SWEEP_SM_14(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 17, 10)
    #define PARAM_SWEEP_SM_15(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 17, 12)
    #define PARAM_SWEEP_SM_16(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 17, 14)
    #define PARAM_SWEEP_SM_17(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 17, 16)
    #define PARAM_SWEEP_SM_18(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 17, 18)
    #define PARAM_SWEEP_SM_19(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 17, 20)
    #define PARAM_SWEEP_SM_20(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 17, 22)
    #define PARAM_SWEEP_SM_21(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 17, 24)
    #define PARAM_SWEEP_SM_22(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 17, 26)
    #define PARAM_SWEEP_SM_23(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 17, 28)
    #define PARAM_SWEEP_SM_24(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 17, 30)
    #define PARAM_SWEEP_SM_25(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 17, 31)

    // S=18 (groups 26-38)
    #define PARAM_SWEEP_SM_26(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 18, 8)
    #define PARAM_SWEEP_SM_27(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 18, 10)
    #define PARAM_SWEEP_SM_28(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 18, 12)
    #define PARAM_SWEEP_SM_29(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 18, 14)
    #define PARAM_SWEEP_SM_30(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 18, 16)
    #define PARAM_SWEEP_SM_31(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 18, 18)
    #define PARAM_SWEEP_SM_32(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 18, 20)
    #define PARAM_SWEEP_SM_33(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 18, 22)
    #define PARAM_SWEEP_SM_34(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 18, 24)
    #define PARAM_SWEEP_SM_35(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 18, 26)
    #define PARAM_SWEEP_SM_36(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 18, 28)
    #define PARAM_SWEEP_SM_37(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 18, 30)
    #define PARAM_SWEEP_SM_38(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 18, 31)

    // S=19 (groups 39-51)
    #define PARAM_SWEEP_SM_39(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 19, 8)
    #define PARAM_SWEEP_SM_40(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 19, 10)
    #define PARAM_SWEEP_SM_41(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 19, 12)
    #define PARAM_SWEEP_SM_42(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 19, 14)
    #define PARAM_SWEEP_SM_43(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 19, 16)
    #define PARAM_SWEEP_SM_44(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 19, 18)
    #define PARAM_SWEEP_SM_45(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 19, 20)
    #define PARAM_SWEEP_SM_46(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 19, 22)
    #define PARAM_SWEEP_SM_47(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 19, 24)
    #define PARAM_SWEEP_SM_48(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 19, 26)
    #define PARAM_SWEEP_SM_49(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 19, 28)
    #define PARAM_SWEEP_SM_50(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 19, 30)
    #define PARAM_SWEEP_SM_51(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 19, 31)

    // S=20 (groups 52-64)
    #define PARAM_SWEEP_SM_52(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 20, 8)
    #define PARAM_SWEEP_SM_53(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 20, 10)
    #define PARAM_SWEEP_SM_54(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 20, 12)
    #define PARAM_SWEEP_SM_55(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 20, 14)
    #define PARAM_SWEEP_SM_56(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 20, 16)
    #define PARAM_SWEEP_SM_57(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 20, 18)
    #define PARAM_SWEEP_SM_58(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 20, 20)
    #define PARAM_SWEEP_SM_59(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 20, 22)
    #define PARAM_SWEEP_SM_60(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 20, 24)
    #define PARAM_SWEEP_SM_61(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 20, 26)
    #define PARAM_SWEEP_SM_62(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 20, 28)
    #define PARAM_SWEEP_SM_63(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 20, 30)
    #define PARAM_SWEEP_SM_64(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 20, 31)

    // S=21 (groups 65-77)
    #define PARAM_SWEEP_SM_65(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 21, 8)
    #define PARAM_SWEEP_SM_66(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 21, 10)
    #define PARAM_SWEEP_SM_67(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 21, 12)
    #define PARAM_SWEEP_SM_68(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 21, 14)
    #define PARAM_SWEEP_SM_69(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 21, 16)
    #define PARAM_SWEEP_SM_70(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 21, 18)
    #define PARAM_SWEEP_SM_71(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 21, 20)
    #define PARAM_SWEEP_SM_72(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 21, 22)
    #define PARAM_SWEEP_SM_73(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 21, 24)
    #define PARAM_SWEEP_SM_74(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 21, 26)
    #define PARAM_SWEEP_SM_75(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 21, 28)
    #define PARAM_SWEEP_SM_76(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 21, 30)
    #define PARAM_SWEEP_SM_77(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 21, 31)

    // S=22 (groups 78-90)
    #define PARAM_SWEEP_SM_78(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 22, 8)
    #define PARAM_SWEEP_SM_79(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 22, 10)
    #define PARAM_SWEEP_SM_80(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 22, 12)
    #define PARAM_SWEEP_SM_81(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 22, 14)
    #define PARAM_SWEEP_SM_82(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 22, 16)
    #define PARAM_SWEEP_SM_83(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 22, 18)
    #define PARAM_SWEEP_SM_84(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 22, 20)
    #define PARAM_SWEEP_SM_85(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 22, 22)
    #define PARAM_SWEEP_SM_86(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 22, 24)
    #define PARAM_SWEEP_SM_87(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 22, 26)
    #define PARAM_SWEEP_SM_88(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 22, 28)
    #define PARAM_SWEEP_SM_89(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 22, 30)
    #define PARAM_SWEEP_SM_90(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 22, 31)

    // S=23 (groups 91-103)
    #define PARAM_SWEEP_SM_91(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 23, 8)
    #define PARAM_SWEEP_SM_92(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 23, 10)
    #define PARAM_SWEEP_SM_93(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 23, 12)
    #define PARAM_SWEEP_SM_94(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 23, 14)
    #define PARAM_SWEEP_SM_95(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 23, 16)
    #define PARAM_SWEEP_SM_96(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 23, 18)
    #define PARAM_SWEEP_SM_97(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 23, 20)
    #define PARAM_SWEEP_SM_98(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 23, 22)
    #define PARAM_SWEEP_SM_99(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 23, 24)
    #define PARAM_SWEEP_SM_100(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 23, 26)
    #define PARAM_SWEEP_SM_101(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 23, 28)
    #define PARAM_SWEEP_SM_102(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 23, 30)
    #define PARAM_SWEEP_SM_103(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 23, 31)

    // S=24 (groups 104-116)
    #define PARAM_SWEEP_SM_104(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 24, 8)
    #define PARAM_SWEEP_SM_105(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 24, 10)
    #define PARAM_SWEEP_SM_106(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 24, 12)
    #define PARAM_SWEEP_SM_107(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 24, 14)
    #define PARAM_SWEEP_SM_108(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 24, 16)
    #define PARAM_SWEEP_SM_109(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 24, 18)
    #define PARAM_SWEEP_SM_110(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 24, 20)
    #define PARAM_SWEEP_SM_111(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 24, 22)
    #define PARAM_SWEEP_SM_112(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 24, 24)
    #define PARAM_SWEEP_SM_113(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 24, 26)
    #define PARAM_SWEEP_SM_114(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 24, 28)
    #define PARAM_SWEEP_SM_115(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 24, 30)
    #define PARAM_SWEEP_SM_116(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 24, 31)

    // S=25 (groups 117-129)
    #define PARAM_SWEEP_SM_117(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 25, 8)
    #define PARAM_SWEEP_SM_118(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 25, 10)
    #define PARAM_SWEEP_SM_119(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 25, 12)
    #define PARAM_SWEEP_SM_120(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 25, 14)
    #define PARAM_SWEEP_SM_121(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 25, 16)
    #define PARAM_SWEEP_SM_122(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 25, 18)
    #define PARAM_SWEEP_SM_123(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 25, 20)
    #define PARAM_SWEEP_SM_124(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 25, 22)
    #define PARAM_SWEEP_SM_125(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 25, 24)
    #define PARAM_SWEEP_SM_126(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 25, 26)
    #define PARAM_SWEEP_SM_127(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 25, 28)
    #define PARAM_SWEEP_SM_128(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 25, 30)
    #define PARAM_SWEEP_SM_129(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 25, 31)

    // S=26 (groups 130-142)
    #define PARAM_SWEEP_SM_130(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 26, 8)
    #define PARAM_SWEEP_SM_131(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 26, 10)
    #define PARAM_SWEEP_SM_132(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 26, 12)
    #define PARAM_SWEEP_SM_133(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 26, 14)
    #define PARAM_SWEEP_SM_134(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 26, 16)
    #define PARAM_SWEEP_SM_135(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 26, 18)
    #define PARAM_SWEEP_SM_136(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 26, 20)
    #define PARAM_SWEEP_SM_137(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 26, 22)
    #define PARAM_SWEEP_SM_138(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 26, 24)
    #define PARAM_SWEEP_SM_139(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 26, 26)
    #define PARAM_SWEEP_SM_140(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 26, 28)
    #define PARAM_SWEEP_SM_141(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 26, 30)
    #define PARAM_SWEEP_SM_142(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 26, 31)

    // S=27 (groups 143-155)
    #define PARAM_SWEEP_SM_143(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 27, 8)
    #define PARAM_SWEEP_SM_144(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 27, 10)
    #define PARAM_SWEEP_SM_145(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 27, 12)
    #define PARAM_SWEEP_SM_146(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 27, 14)
    #define PARAM_SWEEP_SM_147(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 27, 16)
    #define PARAM_SWEEP_SM_148(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 27, 18)
    #define PARAM_SWEEP_SM_149(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 27, 20)
    #define PARAM_SWEEP_SM_150(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 27, 22)
    #define PARAM_SWEEP_SM_151(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 27, 24)
    #define PARAM_SWEEP_SM_152(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 27, 26)
    #define PARAM_SWEEP_SM_153(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 27, 28)
    #define PARAM_SWEEP_SM_154(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 27, 30)
    #define PARAM_SWEEP_SM_155(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 27, 31)

    // S=28 (groups 156-168)
    #define PARAM_SWEEP_SM_156(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 28, 8)
    #define PARAM_SWEEP_SM_157(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 28, 10)
    #define PARAM_SWEEP_SM_158(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 28, 12)
    #define PARAM_SWEEP_SM_159(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 28, 14)
    #define PARAM_SWEEP_SM_160(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 28, 16)
    #define PARAM_SWEEP_SM_161(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 28, 18)
    #define PARAM_SWEEP_SM_162(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 28, 20)
    #define PARAM_SWEEP_SM_163(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 28, 22)
    #define PARAM_SWEEP_SM_164(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 28, 24)
    #define PARAM_SWEEP_SM_165(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 28, 26)
    #define PARAM_SWEEP_SM_166(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 28, 28)
    #define PARAM_SWEEP_SM_167(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 28, 30)
    #define PARAM_SWEEP_SM_168(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 28, 31)

    // S=29 (groups 169-181)
    #define PARAM_SWEEP_SM_169(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 29, 8)
    #define PARAM_SWEEP_SM_170(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 29, 10)
    #define PARAM_SWEEP_SM_171(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 29, 12)
    #define PARAM_SWEEP_SM_172(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 29, 14)
    #define PARAM_SWEEP_SM_173(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 29, 16)
    #define PARAM_SWEEP_SM_174(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 29, 18)
    #define PARAM_SWEEP_SM_175(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 29, 20)
    #define PARAM_SWEEP_SM_176(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 29, 22)
    #define PARAM_SWEEP_SM_177(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 29, 24)
    #define PARAM_SWEEP_SM_178(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 29, 26)
    #define PARAM_SWEEP_SM_179(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 29, 28)
    #define PARAM_SWEEP_SM_180(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 29, 30)
    #define PARAM_SWEEP_SM_181(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 29, 31)

    // S=30 (groups 182-194)
    #define PARAM_SWEEP_SM_182(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 30, 8)
    #define PARAM_SWEEP_SM_183(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 30, 10)
    #define PARAM_SWEEP_SM_184(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 30, 12)
    #define PARAM_SWEEP_SM_185(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 30, 14)
    #define PARAM_SWEEP_SM_186(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 30, 16)
    #define PARAM_SWEEP_SM_187(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 30, 18)
    #define PARAM_SWEEP_SM_188(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 30, 20)
    #define PARAM_SWEEP_SM_189(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 30, 22)
    #define PARAM_SWEEP_SM_190(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 30, 24)
    #define PARAM_SWEEP_SM_191(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 30, 26)
    #define PARAM_SWEEP_SM_192(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 30, 28)
    #define PARAM_SWEEP_SM_193(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 30, 30)
    #define PARAM_SWEEP_SM_194(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 30, 31)

    // S=31 (groups 195-207)
    #define PARAM_SWEEP_SM_195(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 31, 8)
    #define PARAM_SWEEP_SM_196(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 31, 10)
    #define PARAM_SWEEP_SM_197(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 31, 12)
    #define PARAM_SWEEP_SM_198(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 31, 14)
    #define PARAM_SWEEP_SM_199(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 31, 16)
    #define PARAM_SWEEP_SM_200(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 31, 18)
    #define PARAM_SWEEP_SM_201(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 31, 20)
    #define PARAM_SWEEP_SM_202(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 31, 22)
    #define PARAM_SWEEP_SM_203(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 31, 24)
    #define PARAM_SWEEP_SM_204(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 31, 26)
    #define PARAM_SWEEP_SM_205(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 31, 28)
    #define PARAM_SWEEP_SM_206(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 31, 30)
    #define PARAM_SWEEP_SM_207(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 31, 31, 31)

#else  // PARAM_SWEEP_PROTEIN

    // Protein parameter grid: K=12
    // S values: {6,7,8,9,10,11,12} (7 values)
    // M values: {4,6,8,10,12} (5 values)
    // H values: {4,8,12,16} (4 values)
    // Groups: 7*5 = 35

    // S=6 (groups 0-4)
    #define PARAM_SWEEP_SM_0(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 6, 4)
    #define PARAM_SWEEP_SM_1(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 6, 6)
    #define PARAM_SWEEP_SM_2(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 6, 8)
    #define PARAM_SWEEP_SM_3(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 6, 10)
    #define PARAM_SWEEP_SM_4(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 6, 12)

    // S=7 (groups 5-9)
    #define PARAM_SWEEP_SM_5(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 7, 4)
    #define PARAM_SWEEP_SM_6(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 7, 6)
    #define PARAM_SWEEP_SM_7(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 7, 8)
    #define PARAM_SWEEP_SM_8(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 7, 10)
    #define PARAM_SWEEP_SM_9(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 7, 12)

    // S=8 (groups 10-14)
    #define PARAM_SWEEP_SM_10(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 8, 4)
    #define PARAM_SWEEP_SM_11(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 8, 6)
    #define PARAM_SWEEP_SM_12(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 8, 8)
    #define PARAM_SWEEP_SM_13(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 8, 10)
    #define PARAM_SWEEP_SM_14(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 8, 12)

    // S=9 (groups 15-19)
    #define PARAM_SWEEP_SM_15(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 9, 4)
    #define PARAM_SWEEP_SM_16(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 9, 6)
    #define PARAM_SWEEP_SM_17(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 9, 8)
    #define PARAM_SWEEP_SM_18(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 9, 10)
    #define PARAM_SWEEP_SM_19(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 9, 12)

    // S=10 (groups 20-24)
    #define PARAM_SWEEP_SM_20(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 10, 4)
    #define PARAM_SWEEP_SM_21(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 10, 6)
    #define PARAM_SWEEP_SM_22(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 10, 8)
    #define PARAM_SWEEP_SM_23(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 10, 10)
    #define PARAM_SWEEP_SM_24(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 10, 12)

    // S=11 (groups 25-29)
    #define PARAM_SWEEP_SM_25(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 11, 4)
    #define PARAM_SWEEP_SM_26(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 11, 6)
    #define PARAM_SWEEP_SM_27(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 11, 8)
    #define PARAM_SWEEP_SM_28(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 11, 10)
    #define PARAM_SWEEP_SM_29(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 11, 12)

    // S=12 (groups 30-34)
    #define PARAM_SWEEP_SM_30(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 12, 4)
    #define PARAM_SWEEP_SM_31(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 12, 6)
    #define PARAM_SWEEP_SM_32(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 12, 8)
    #define PARAM_SWEEP_SM_33(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 12, 10)
    #define PARAM_SWEEP_SM_34(MACRO) PARAM_SWEEP_H_DEFAULT(MACRO, 12, 12, 12)

#endif  // PARAM_SWEEP_PROTEIN

// Token-paste selection: PARAM_SWEEP_APPLY_GROUP(N, MACRO) -> PARAM_SWEEP_SM_N(MACRO)
#define PARAM_SWEEP_APPLY_GROUP_IMPL(g, MACRO) PARAM_SWEEP_SM_##g(MACRO)
#define PARAM_SWEEP_APPLY_GROUP(g, MACRO) PARAM_SWEEP_APPLY_GROUP_IMPL(g, MACRO)

#if !defined(PARAM_SWEEP_GROUP)
    #error "PARAM_SWEEP_GROUP must be defined. Build via meson targets (param-sweep-group*)."
#else
    #define PARAM_SWEEP_APPLY_DEFAULT(MACRO) PARAM_SWEEP_APPLY_GROUP(PARAM_SWEEP_GROUP, MACRO)
#endif

// Instantiate configs, fixtures, benchmarks, and registrations
#define PARAM_SWEEP_DEFINE_CONFIG_AND_FIXTURE_WRAPPER(K, S, M, H) \
    PARAM_SWEEP_DEFINE_CONFIG_AND_FIXTURE(K, S, M, H)

#define PARAM_SWEEP_DEFINE_ALL_WRAPPER(K, S, M, H) \
    PARAM_SWEEP_DEFINE_ALL(BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

#define PARAM_SWEEP_REGISTER_ALL_WRAPPER(K, S, M, H) \
    REGISTER_PARAM_SWEEP_ALL(BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

PARAM_SWEEP_APPLY_DEFAULT(PARAM_SWEEP_DEFINE_CONFIG_AND_FIXTURE_WRAPPER)
PARAM_SWEEP_APPLY_DEFAULT(PARAM_SWEEP_DEFINE_ALL_WRAPPER)
PARAM_SWEEP_APPLY_DEFAULT(PARAM_SWEEP_REGISTER_ALL_WRAPPER)

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
