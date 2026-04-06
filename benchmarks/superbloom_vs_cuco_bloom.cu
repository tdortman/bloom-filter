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

#include <bloom/BloomFilter.cuh>
#include <bloom/hashutil.cuh>
#include <bloom/helpers.cuh>
#include <cuco/bloom_filter.cuh>

#include "benchmark_common.cuh"

namespace bm = benchmark;


using SuperConfig = bloom::Config<31, 21, 27, 6, 256, 512>;
using SuperS30Config = bloom::Config<31, 21, 30, 7, 256, 512>;
using SuperS28Config = bloom::Config<31, 21, 28, 6, 256, 512>;
using SuperS24Config = bloom::Config<31, 21, 24, 5, 256, 512>;
using SuperNoFindereConfig = bloom::Config<31, 21, 31, 8, 256, 512>;
using CucoBloom = cuco::bloom_filter<uint64_t>;

constexpr size_t kBitsPerItem = 16;

std::string generateRandomDNA(size_t length, uint32_t seed) {
    static constexpr char bases[] = {'A', 'C', 'G', 'T'};

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 3);

    std::string sequence(length, 'A');
    for (char& base : sequence) {
        base = bases[dist(rng)];
    }
    return sequence;
}

template <size_t K>
bool packWindow(std::string_view sequence, size_t start, uint64_t& packed) {
    packed = 0;
    for (size_t i = 0; i < K; ++i) {
        const uint8_t encoded = bloom::detail::encodeBase(sequence[start + i]);
        if (encoded > 3) {
            return false;
        }
        packed = (packed << 2) | static_cast<uint64_t>(encoded);
    }
    return true;
}

template <size_t K>
std::unordered_set<uint64_t> collectPackedKmers(std::string_view sequence) {
    std::unordered_set<uint64_t> kmers;
    if (sequence.size() < K) {
        return kmers;
    }

    kmers.reserve(sequence.size() - K + 1);
    for (size_t index = 0; index + K <= sequence.size(); ++index) {
        uint64_t packed = 0;
        if (packWindow<K>(sequence, index, packed)) {
            kmers.insert(packed);
        }
    }
    return kmers;
}

template <size_t K>
bool hasPackedKmerOverlap(std::string_view query, const std::unordered_set<uint64_t>& insertKmers) {
    if (query.size() < K) {
        return false;
    }

    for (size_t index = 0; index + K <= query.size(); ++index) {
        uint64_t packed = 0;
        if (packWindow<K>(query, index, packed) && insertKmers.contains(packed)) {
            return true;
        }
    }
    return false;
}

template <size_t K>
std::string generateZeroOverlapQuery(
    size_t length,
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

struct BenchmarkSequences {
    std::string throughputSequence;
    std::string fprInsertSequence;
    std::string zeroOverlapQuery;
};

template <size_t K>
const BenchmarkSequences& getCachedSequences(size_t length) {
    static std::unordered_map<size_t, BenchmarkSequences> cache;

    auto it = cache.find(length);
    if (it != cache.end()) {
        return it->second;
    }

    BenchmarkSequences sequences;
    sequences.throughputSequence = generateRandomDNA(length, 42);
    sequences.fprInsertSequence = generateRandomDNA(length, 7);
    const auto insertKmers = collectPackedKmers<K>(sequences.fprInsertSequence);
    sequences.zeroOverlapQuery = generateZeroOverlapQuery<K>(length, 1337, insertKmers);

    return cache.emplace(length, std::move(sequences)).first->second;
}

size_t cucoNumBlocks(size_t numItems) {
    constexpr auto bitsPerWord = sizeof(typename CucoBloom::word_type) * 8;
    return SDIV(numItems * kBitsPerItem, CucoBloom::words_per_block * bitsPerWord);
}

void setSequenceCounters(
    bm::State& state,
    size_t memoryBytes,
    size_t sequenceLength,
    size_t numKmers,
    size_t numSmers
) {
    benchmark_common::setCommonCounters(state, memoryBytes, numSmers, sequenceLength);
    state.counters["num_kmers"] = bm::Counter(static_cast<double>(numKmers));
    state.counters["num_smers"] = bm::Counter(static_cast<double>(numSmers));
}

template <size_t WindowLength>
__global__ void hashWindowsKernel(const char* sequence, size_t length, uint64_t* hashes) {
    const size_t index = bloom::detail::globalThreadId();
    if (index + WindowLength > length) {
        return;
    }

    uint64_t packed = 0;
    _Pragma("unroll")
    for (size_t i = 0; i < WindowLength; ++i) {
        packed = (packed << 2) |
                 static_cast<uint64_t>(bloom::detail::encodeBase(sequence[index + i]));
    }
    hashes[index] = xxhash::xxhash64(packed);
}

template <typename Config>
class SuperBloomFixtureBase : public bm::Fixture {
   public:
    void setupCommon(const bm::State& state) {
        sequenceLength = static_cast<size_t>(state.range(0));
        const auto& sequences = getCachedSequences<Config::k>(sequenceLength);
        hostSequence = sequences.throughputSequence;
        numKmers = sequenceLength - Config::k + 1;
        numSmers = sequenceLength - Config::s + 1;

        const size_t requestedFilterBits = bloom::detail::nextPowerOfTwo(numSmers * kBitsPerItem);
        filter = std::make_unique<bloom::Filter<Config>>(requestedFilterBits);
        filterMemory = filter->filterBits() / 8;
        dOutput.resize(numKmers);

        hostFprInsertSequence = sequences.fprInsertSequence;
        hostZeroOverlapQuery = sequences.zeroOverlapQuery;
    }

    void tearDownCommon() {
        filter.reset();
        dOutput.clear();
        dOutput.shrink_to_fit();
    }

    void setCounters(bm::State& state) const {
        setSequenceCounters(state, filterMemory, sequenceLength, numKmers, numSmers);
        state.counters["s"] = bm::Counter(static_cast<double>(Config::s));
        state.counters["hashes"] = bm::Counter(static_cast<double>(Config::hashCount));
    }

    size_t sequenceLength{};
    size_t numKmers{};
    size_t numSmers{};
    size_t filterMemory{};
    std::string hostSequence;
    std::string hostFprInsertSequence;
    std::string hostZeroOverlapQuery;
    thrust::device_vector<uint8_t> dOutput;
    std::unique_ptr<bloom::Filter<Config>> filter;
    benchmark_common::GPUTimer timer;
};

class SuperBloomFixture : public SuperBloomFixtureBase<SuperConfig> {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        setupCommon(state);
    }

    void TearDown(const bm::State&) override {
        tearDownCommon();
    }
};

class SuperBloomNoFindereFixture : public SuperBloomFixtureBase<SuperNoFindereConfig> {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        setupCommon(state);
    }

    void TearDown(const bm::State&) override {
        tearDownCommon();
    }
};

class SuperBloomS30Fixture : public SuperBloomFixtureBase<SuperS30Config> {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        setupCommon(state);
    }

    void TearDown(const bm::State&) override {
        tearDownCommon();
    }
};

class SuperBloomS28Fixture : public SuperBloomFixtureBase<SuperS28Config> {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        setupCommon(state);
    }

    void TearDown(const bm::State&) override {
        tearDownCommon();
    }
};

class SuperBloomS24Fixture : public SuperBloomFixtureBase<SuperS24Config> {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        setupCommon(state);
    }

    void TearDown(const bm::State&) override {
        tearDownCommon();
    }
};

class CucoBloomFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        sequenceLength = static_cast<size_t>(state.range(0));
        const auto& sequences = getCachedSequences<SuperConfig::k>(sequenceLength);
        hostSequence = sequences.throughputSequence;
        numKmers = sequenceLength - SuperConfig::k + 1;
        numSmers = sequenceLength - SuperConfig::s + 1;

        dSequence.resize(sequenceLength);
        dKmerHashes.resize(numKmers);
        dOutput.resize(numKmers);

        filter = std::make_unique<CucoBloom>(cucoNumBlocks(numKmers));
        filterMemory = filter->block_extent() * CucoBloom::words_per_block *
                       sizeof(typename CucoBloom::word_type);

        hostFprInsertSequence = sequences.fprInsertSequence;
        hostZeroOverlapQuery = sequences.zeroOverlapQuery;
    }

    void TearDown(const bm::State&) override {
        filter.reset();
        dSequence.clear();
        dKmerHashes.clear();
        dOutput.clear();
        dSequence.shrink_to_fit();
        dKmerHashes.shrink_to_fit();
        dOutput.shrink_to_fit();
    }

    void setCounters(bm::State& state) const {
        setSequenceCounters(state, filterMemory, sequenceLength, numKmers, numSmers);
    }

    void stageSequence(const std::string& sequence, cudaStream_t stream = {}) {
        CUDA_CALL(cudaMemcpyAsync(
            thrust::raw_pointer_cast(dSequence.data()),
            sequence.data(),
            sequenceLength * sizeof(char),
            cudaMemcpyHostToDevice,
            stream
        ));
    }

    void hashKmers(cudaStream_t stream = {}) {
        const size_t blockSize = 256;
        const size_t gridSize = SDIV(numKmers, blockSize);
        hashWindowsKernel<SuperConfig::k><<<gridSize, blockSize, 0, stream>>>(
            thrust::raw_pointer_cast(dSequence.data()),
            sequenceLength,
            thrust::raw_pointer_cast(dKmerHashes.data())
        );
        CUDA_CALL(cudaGetLastError());
    }

    size_t sequenceLength{};
    size_t numKmers{};
    size_t numSmers{};
    size_t filterMemory{};
    std::string hostSequence;
    std::string hostFprInsertSequence;
    std::string hostZeroOverlapQuery;
    thrust::device_vector<char> dSequence;
    thrust::device_vector<uint64_t> dKmerHashes;
    thrust::device_vector<uint8_t> dOutput;
    std::unique_ptr<CucoBloom> filter;
    benchmark_common::GPUTimer timer;
};

BENCHMARK_DEFINE_F(SuperBloomFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        CUDA_CALL(cudaDeviceSynchronize());

        timer.start();
        benchmark::DoNotOptimize(filter->insertSequence(hostSequence));
        const double elapsed = timer.elapsed();
        state.SetIterationTime(elapsed);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(SuperBloomFixture, Query)(bm::State& state) {
    filter->clear();
    benchmark::DoNotOptimize(filter->insertSequence(hostSequence));
    CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        timer.start();
        filter->containsSequence(
            hostSequence.data(), hostSequence.size(), thrust::raw_pointer_cast(dOutput.data())
        );
        const double elapsed = timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(dOutput.data()));
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(SuperBloomFixture, FPR)(bm::State& state) {
    filter->clear();
    benchmark::DoNotOptimize(filter->insertSequence(hostFprInsertSequence));
    CUDA_CALL(cudaDeviceSynchronize());

    size_t falsePositives = 0;
    for (auto _ : state) {
        timer.start();
        filter->containsSequence(
            hostZeroOverlapQuery.data(),
            hostZeroOverlapQuery.size(),
            thrust::raw_pointer_cast(dOutput.data())
        );
        const double elapsed = timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(dOutput.data()));
    }

    falsePositives = static_cast<size_t>(thrust::count(dOutput.begin(), dOutput.end(), uint8_t{1}));
    setCounters(state);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["fpr_percentage"] =
        bm::Counter(100.0 * static_cast<double>(falsePositives) / static_cast<double>(numKmers));
}

BENCHMARK_DEFINE_F(SuperBloomNoFindereFixture, FPR)(bm::State& state) {
    filter->clear();
    benchmark::DoNotOptimize(filter->insertSequence(hostFprInsertSequence));
    CUDA_CALL(cudaDeviceSynchronize());

    size_t falsePositives = 0;
    for (auto _ : state) {
        timer.start();
        filter->containsSequence(
            hostZeroOverlapQuery.data(),
            hostZeroOverlapQuery.size(),
            thrust::raw_pointer_cast(dOutput.data())
        );
        const double elapsed = timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(dOutput.data()));
    }

    falsePositives = static_cast<size_t>(thrust::count(dOutput.begin(), dOutput.end(), uint8_t{1}));
    setCounters(state);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["fpr_percentage"] =
        bm::Counter(100.0 * static_cast<double>(falsePositives) / static_cast<double>(numKmers));
}

BENCHMARK_DEFINE_F(SuperBloomS30Fixture, FPR)(bm::State& state) {
    filter->clear();
    benchmark::DoNotOptimize(filter->insertSequence(hostFprInsertSequence));
    CUDA_CALL(cudaDeviceSynchronize());

    size_t falsePositives = 0;
    for (auto _ : state) {
        timer.start();
        filter->containsSequence(
            hostZeroOverlapQuery.data(),
            hostZeroOverlapQuery.size(),
            thrust::raw_pointer_cast(dOutput.data())
        );
        const double elapsed = timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(dOutput.data()));
    }

    falsePositives = static_cast<size_t>(thrust::count(dOutput.begin(), dOutput.end(), uint8_t{1}));
    setCounters(state);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["fpr_percentage"] =
        bm::Counter(100.0 * static_cast<double>(falsePositives) / static_cast<double>(numKmers));
}

BENCHMARK_DEFINE_F(SuperBloomS28Fixture, FPR)(bm::State& state) {
    filter->clear();
    benchmark::DoNotOptimize(filter->insertSequence(hostFprInsertSequence));
    CUDA_CALL(cudaDeviceSynchronize());

    size_t falsePositives = 0;
    for (auto _ : state) {
        timer.start();
        filter->containsSequence(
            hostZeroOverlapQuery.data(),
            hostZeroOverlapQuery.size(),
            thrust::raw_pointer_cast(dOutput.data())
        );
        const double elapsed = timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(dOutput.data()));
    }

    falsePositives = static_cast<size_t>(thrust::count(dOutput.begin(), dOutput.end(), uint8_t{1}));
    setCounters(state);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["fpr_percentage"] =
        bm::Counter(100.0 * static_cast<double>(falsePositives) / static_cast<double>(numKmers));
}

BENCHMARK_DEFINE_F(SuperBloomS24Fixture, FPR)(bm::State& state) {
    filter->clear();
    benchmark::DoNotOptimize(filter->insertSequence(hostFprInsertSequence));
    CUDA_CALL(cudaDeviceSynchronize());

    size_t falsePositives = 0;
    for (auto _ : state) {
        timer.start();
        filter->containsSequence(
            hostZeroOverlapQuery.data(),
            hostZeroOverlapQuery.size(),
            thrust::raw_pointer_cast(dOutput.data())
        );
        const double elapsed = timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(dOutput.data()));
    }

    falsePositives = static_cast<size_t>(thrust::count(dOutput.begin(), dOutput.end(), uint8_t{1}));
    setCounters(state);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["fpr_percentage"] =
        bm::Counter(100.0 * static_cast<double>(falsePositives) / static_cast<double>(numKmers));
}

BENCHMARK_DEFINE_F(CucoBloomFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        CUDA_CALL(cudaDeviceSynchronize());

        timer.start();
        stageSequence(hostSequence);
        hashKmers();
        filter->add(dKmerHashes.begin(), dKmerHashes.end());
        const double elapsed = timer.elapsed();
        state.SetIterationTime(elapsed);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(CucoBloomFixture, Query)(bm::State& state) {
    filter->clear();
    stageSequence(hostSequence);
    hashKmers();
    filter->add(dKmerHashes.begin(), dKmerHashes.end());
    CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        timer.start();
        stageSequence(hostSequence);
        hashKmers();
        filter->contains(
            dKmerHashes.begin(),
            dKmerHashes.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(dOutput.data()))
        );
        const double elapsed = timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(dOutput.data()));
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(CucoBloomFixture, FPR)(bm::State& state) {
    filter->clear();
    stageSequence(hostFprInsertSequence);
    hashKmers();
    filter->add(dKmerHashes.begin(), dKmerHashes.end());
    CUDA_CALL(cudaDeviceSynchronize());

    size_t falsePositives = 0;
    for (auto _ : state) {
        timer.start();
        stageSequence(hostZeroOverlapQuery);
        hashKmers();
        filter->contains(
            dKmerHashes.begin(),
            dKmerHashes.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(dOutput.data()))
        );
        const double elapsed = timer.elapsed();
        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(thrust::raw_pointer_cast(dOutput.data()));
    }

    falsePositives = static_cast<size_t>(thrust::count(dOutput.begin(), dOutput.end(), uint8_t{1}));
    setCounters(state);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["fpr_percentage"] =
        bm::Counter(100.0 * static_cast<double>(falsePositives) / static_cast<double>(numKmers));
}

REGISTER_BENCHMARK(SuperBloomFixture, Insert);
REGISTER_BENCHMARK(SuperBloomFixture, Query);
REGISTER_BENCHMARK(SuperBloomFixture, FPR);
REGISTER_BENCHMARK(SuperBloomNoFindereFixture, FPR);
REGISTER_BENCHMARK(SuperBloomS30Fixture, FPR);
REGISTER_BENCHMARK(SuperBloomS28Fixture, FPR);
REGISTER_BENCHMARK(SuperBloomS24Fixture, FPR);
REGISTER_BENCHMARK(CucoBloomFixture, Insert);
REGISTER_BENCHMARK(CucoBloomFixture, Query);
REGISTER_BENCHMARK(CucoBloomFixture, FPR);


STANDARD_BENCHMARK_MAIN();
