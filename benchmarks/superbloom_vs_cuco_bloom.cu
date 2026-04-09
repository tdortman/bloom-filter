#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <bloom/BloomFilter.cuh>
#include <bloom/helpers.cuh>
#include <cuco/bloom_filter.cuh>

#include "benchmark_common.cuh"

namespace bm = benchmark;

using CucoBloom = cuco::bloom_filter<uint64_t>;

// It's K - S - M - H

#define SUPERBLOOM_FIRST_INSERT_QUERY_FPR_CONFIG(X) X(31, 27, 21, 3)

#define SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR(X) SUPERBLOOM_FIRST_INSERT_QUERY_FPR_CONFIG(X)

#define SUPERBLOOM_CONFIGS_FPR_ONLY(X) \
    X(31, 31, 21, 3)                   \
    X(31, 30, 21, 3)                   \
    X(31, 28, 21, 3)                   \
    X(31, 24, 21, 3)                   \
    X(31, 20, 21, 3)                   \
    X(31, 16, 21, 3)

#define FOR_EACH_SUPERBLOOM_CONFIG(X)      \
    SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR(X) \
    SUPERBLOOM_CONFIGS_FPR_ONLY(X)

constexpr uint64_t kBitsPerItem = 16;

std::string generateRandomDNA(uint64_t length, uint32_t seed) {
    static constexpr char bases[] = {'A', 'C', 'G', 'T'};

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 3);

    std::string sequence(length, 'A');
    for (char& base : sequence) {
        base = bases[dist(rng)];
    }
    return sequence;
}

template <uint64_t K>
bool packWindow(std::string_view sequence, uint64_t start, uint64_t& packed) {
    packed = 0;
    for (uint64_t i = 0; i < K; ++i) {
        const uint8_t encoded = bloom::detail::encodeBase(sequence[start + i]);
        if (encoded > 3) {
            return false;
        }
        packed = (packed << 2) | static_cast<uint64_t>(encoded);
    }
    return true;
}

template <uint64_t K>
std::vector<uint64_t> collectPackedKmers(std::string_view sequence) {
    std::vector<uint64_t> kmers;
    if (sequence.size() < K) {
        return kmers;
    }

    kmers.reserve(sequence.size() - K + 1);
    for (uint64_t index = 0; index + K <= sequence.size(); ++index) {
        uint64_t packed = 0;
        if (packWindow<K>(sequence, index, packed)) {
            kmers.push_back(packed);
        }
    }
    return kmers;
}

template <uint64_t K>
std::unordered_set<uint64_t> collectPackedKmerSet(std::string_view sequence) {
    const auto kmers = collectPackedKmers<K>(sequence);
    return std::unordered_set<uint64_t>(kmers.begin(), kmers.end());
}

template <uint64_t K>
bool hasPackedKmerOverlap(std::string_view query, const std::unordered_set<uint64_t>& insertKmers) {
    if (query.size() < K) {
        return false;
    }

    for (uint64_t index = 0; index + K <= query.size(); ++index) {
        uint64_t packed = 0;
        if (packWindow<K>(query, index, packed) && insertKmers.contains(packed)) {
            return true;
        }
    }
    return false;
}

template <uint64_t K>
std::string generateZeroOverlapQuery(
    uint64_t length,
    uint32_t seed,
    const std::unordered_set<uint64_t>& insertKmers
) {
    for (uint32_t attempt = 0; attempt < 1024; ++attempt) {
        auto query = generateRandomDNA(length, seed + attempt);
        if (!hasPackedKmerOverlap<K>(query, insertKmers)) {
            return query;
        }
    }

    throw std::runtime_error("failed to generate zero-overlap query sequence");
}

struct BenchmarkKmers {
    std::vector<uint64_t> throughputPackedKmers;
    std::vector<uint64_t> fprInsertPackedKmers;
    std::vector<uint64_t> zeroOverlapPackedKmers;
};

template <uint64_t K>
const BenchmarkKmers& getCachedKmers(uint64_t length) {
    static std::unordered_map<uint64_t, BenchmarkKmers> cache;

    auto it = cache.find(length);
    if (it != cache.end()) {
        return it->second;
    }

    BenchmarkKmers sequences;
    const auto throughputSequence = generateRandomDNA(length, 42);
    sequences.throughputPackedKmers = collectPackedKmers<K>(throughputSequence);

    const auto fprInsertSequence = generateRandomDNA(length, 7);
    sequences.fprInsertPackedKmers = collectPackedKmers<K>(fprInsertSequence);

    const auto insertKmers = collectPackedKmerSet<K>(fprInsertSequence);
    const auto zeroOverlapQuery = generateZeroOverlapQuery<K>(length, 1337, insertKmers);
    sequences.zeroOverlapPackedKmers = collectPackedKmers<K>(zeroOverlapQuery);

    return cache.emplace(length, std::move(sequences)).first->second;
}

uint64_t cucoNumBlocks(uint64_t numItems) {
    constexpr auto bitsPerWord = sizeof(typename CucoBloom::word_type) * 8;
    return SDIV(numItems * kBitsPerItem, CucoBloom::words_per_block * bitsPerWord);
}

void setPackedKmerCounters(
    bm::State& state,
    uint64_t memoryBytes,
    uint64_t sequenceLength,
    uint64_t numKmers
) {
    benchmark_common::setCommonCounters(state, memoryBytes, numKmers, sequenceLength);
    state.counters["num_kmers"] = bm::Counter(static_cast<double>(numKmers));
    state.counters["packed_input"] = bm::Counter(1.0);
}

template <typename Config>
class SuperBloomFixtureBase : public bm::Fixture {
   public:
    void setupCommon(const bm::State& state) {
        sequenceLength = static_cast<uint64_t>(state.range(0));
        cachedKmers = &getCachedKmers<Config::k>(sequenceLength);
        numKmers = cachedKmers->throughputPackedKmers.size();
        numSmers = sequenceLength - Config::s + 1;

        const uint64_t requestedFilterBits = bloom::detail::nextPowerOfTwo(numKmers * kBitsPerItem);
        filter = std::make_unique<bloom::Filter<Config>>(requestedFilterBits);
        filterMemory = filter->filterBits() / 8;
        d_output.resize(numKmers);
    }

    void tearDownCommon() {
        filter.reset();
        cachedKmers = nullptr;
        d_output.clear();
        d_output.shrink_to_fit();
    }

    void setPackedCounters(bm::State& state) const {
        setPackedKmerCounters(state, filterMemory, sequenceLength, numKmers);
        state.counters["s"] = bm::Counter(static_cast<double>(Config::s));
        state.counters["hashes"] = bm::Counter(static_cast<double>(Config::hashCount));
    }

    uint64_t sequenceLength{};
    uint64_t numKmers{};
    uint64_t numSmers{};
    uint64_t filterMemory{};
    const BenchmarkKmers* cachedKmers{};
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

#define DEFINE_SUPERBLOOM_CONFIG_AND_FIXTURE(K, S, M, H)                              \
    using SUPERBLOOM_CONFIG_SYMBOL(K, S, M, H) = bloom::Config<K, S, M, H, 512>; \
    using SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H) =                                     \
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
    void SetUp(const bm::State& state) override {
        sequenceLength = static_cast<uint64_t>(state.range(0));
        cachedKmers = &getCachedKmers<CucoReferenceConfig::k>(sequenceLength);
        numKmers = cachedKmers->throughputPackedKmers.size();

        d_kmerHashes.resize(numKmers);
        d_output.resize(numKmers);

        filter = std::make_unique<CucoBloom>(cucoNumBlocks(numKmers));
        filterMemory = filter->block_extent() * CucoBloom::words_per_block *
                       sizeof(typename CucoBloom::word_type);
    }

    void TearDown(const bm::State&) override {
        filter.reset();
        cachedKmers = nullptr;
        d_kmerHashes.clear();
        d_output.clear();
        d_kmerHashes.shrink_to_fit();
        d_output.shrink_to_fit();
    }

    void setPackedCounters(bm::State& state) const {
        setPackedKmerCounters(state, filterMemory, sequenceLength, numKmers);
    }

    void stagePackedKmers(const std::vector<uint64_t>& kmers, cudaStream_t stream = {}) {
        CUDA_CALL(cudaMemcpyAsync(
            thrust::raw_pointer_cast(d_kmerHashes.data()),
            kmers.data(),
            kmers.size() * sizeof(uint64_t),
            cudaMemcpyHostToDevice,
            stream
        ));
    }

    uint64_t sequenceLength{};
    uint64_t numKmers{};
    uint64_t filterMemory{};
    const BenchmarkKmers* cachedKmers{};
    thrust::device_vector<uint64_t> d_kmerHashes;
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
        benchmark::DoNotOptimize(
            fixture.filter->insertPackedKmers(fixture.cachedKmers->throughputPackedKmers)
        );
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
    }
    fixture.setPackedCounters(state);
}

template <typename Fixture>
void runSuperBloomQueryBenchmark(Fixture& fixture, bm::State& state) {
    fixture.filter->clear();
    benchmark::DoNotOptimize(
        fixture.filter->insertPackedKmers(fixture.cachedKmers->throughputPackedKmers)
    );
    CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fixture.timer.start();
        fixture.filter->containsPackedKmers(
            fixture.cachedKmers->throughputPackedKmers.data(),
            fixture.cachedKmers->throughputPackedKmers.size(),
            thrust::raw_pointer_cast(fixture.d_output.data())
        );
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }
    fixture.setPackedCounters(state);
}

template <typename Fixture>
void runSuperBloomFprBenchmark(Fixture& fixture, bm::State& state) {
    fixture.filter->clear();
    benchmark::DoNotOptimize(
        fixture.filter->insertPackedKmers(fixture.cachedKmers->fprInsertPackedKmers)
    );
    CUDA_CALL(cudaDeviceSynchronize());

    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        fixture.filter->containsPackedKmers(
            fixture.cachedKmers->zeroOverlapPackedKmers.data(),
            fixture.cachedKmers->zeroOverlapPackedKmers.size(),
            thrust::raw_pointer_cast(fixture.d_output.data())
        );
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }

    falsePositives = static_cast<uint64_t>(
        thrust::count(fixture.d_output.begin(), fixture.d_output.end(), uint8_t{1})
    );
    fixture.setPackedCounters(state);
    setFprCounters(state, falsePositives, fixture.numKmers);
}

void stageAndAddCuco(CucoBloomFixture& fixture, const std::vector<uint64_t>& kmers) {
    fixture.stagePackedKmers(kmers);
    fixture.filter->add(fixture.d_kmerHashes.begin(), fixture.d_kmerHashes.end());
}

void runCucoInsertBenchmark(CucoBloomFixture& fixture, bm::State& state) {
    for (auto _ : state) {
        fixture.filter->clear();
        CUDA_CALL(cudaDeviceSynchronize());

        fixture.timer.start();
        stageAndAddCuco(fixture, fixture.cachedKmers->throughputPackedKmers);
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
    }
    fixture.setPackedCounters(state);
}

void runCucoQueryBenchmark(CucoBloomFixture& fixture, bm::State& state) {
    fixture.filter->clear();
    stageAndAddCuco(fixture, fixture.cachedKmers->throughputPackedKmers);
    CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fixture.timer.start();
        fixture.stagePackedKmers(fixture.cachedKmers->throughputPackedKmers);
        fixture.filter->contains(
            fixture.d_kmerHashes.begin(),
            fixture.d_kmerHashes.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(fixture.d_output.data()))
        );
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }
    fixture.setPackedCounters(state);
}

void runCucoFprBenchmark(CucoBloomFixture& fixture, bm::State& state) {
    fixture.filter->clear();
    stageAndAddCuco(fixture, fixture.cachedKmers->fprInsertPackedKmers);
    CUDA_CALL(cudaDeviceSynchronize());

    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        fixture.stagePackedKmers(fixture.cachedKmers->zeroOverlapPackedKmers);
        fixture.filter->contains(
            fixture.d_kmerHashes.begin(),
            fixture.d_kmerHashes.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(fixture.d_output.data()))
        );
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(fixture.d_output.data()));
    }

    falsePositives = static_cast<uint64_t>(
        thrust::count(fixture.d_output.begin(), fixture.d_output.end(), uint8_t{1})
    );
    fixture.setPackedCounters(state);
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

STANDARD_BENCHMARK_MAIN();
