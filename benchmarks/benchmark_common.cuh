#pragma once

#include <benchmark/benchmark.h>
#include <cuda/__cmath/ceil_div.h>
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
#include <memory>
#include <unordered_map>

#include <bloom/BloomFilter.cuh>
#include <bloom/device_span.cuh>
#include <bloom/helpers.cuh>

namespace benchmark_common {

class GPUTimer {
   public:
    GPUTimer() {
        BLOOM_CUDA_CALL(cudaEventCreate(&start_));
        BLOOM_CUDA_CALL(cudaEventCreate(&stop_));
    }

    ~GPUTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    GPUTimer(const GPUTimer&) = delete;
    GPUTimer& operator=(const GPUTimer&) = delete;

    void start(cudaStream_t stream = {}) {
        BLOOM_CUDA_CALL(cudaEventRecord(start_, stream));
    }

    [[nodiscard]] double elapsed(cudaStream_t stream = {}) {
        BLOOM_CUDA_CALL(cudaEventRecord(stop_, stream));
        BLOOM_CUDA_CALL(cudaEventSynchronize(stop_));

        float milliseconds = 0.0f;
        BLOOM_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return static_cast<double>(milliseconds) / 1000.0;
    }

   private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

// Concatenate all records in a FASTA/FASTQ file into a single sequence,
// with 'N' as a separatorbetween records
std::vector<char> readFastxConcatenated(std::string_view path) {
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

inline void setBenchmarkCounters(
    benchmark::State& state,
    uint64_t memoryBytes,
    uint64_t sequenceLength,
    uint64_t numKmers
) {
    setCommonCounters(state, memoryBytes, numKmers, sequenceLength);
    state.counters["num_kmers"] = benchmark::Counter(static_cast<double>(numKmers));
}

namespace detail {

__device__ __forceinline__ char randomDnaBase(uint64_t idx, uint32_t seed) {
    thrust::default_random_engine rng(seed);
    thrust::uniform_int_distribution<uint32_t> dist(0, 3);
    rng.discard(idx);
    constexpr char bases[] = {'A', 'C', 'G', 'T'};
    return bases[dist(rng)];
}

__device__ __forceinline__ char randomProteinSymbol(uint64_t idx, uint32_t seed) {
    thrust::default_random_engine rng(seed);
    thrust::uniform_int_distribution<uint32_t> dist(0, 19);
    rng.discard(idx);
    constexpr char symbols[] = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'};
    return symbols[dist(rng)];
}

}  // namespace detail

inline void
gpuGenerateDna(thrust::device_vector<char>& d_seq, uint64_t length, uint32_t seed = 42) {
    d_seq.resize(length);
    thrust::transform(
        thrust::counting_iterator<uint64_t>(0),
        thrust::counting_iterator<uint64_t>(length),
        d_seq.begin(),
        [seed] __device__(uint64_t idx) { return detail::randomDnaBase(idx, seed); }
    );
}

inline void
gpuGenerateProtein(thrust::device_vector<char>& d_seq, uint64_t length, uint32_t seed = 42) {
    d_seq.resize(length);
    thrust::transform(
        thrust::counting_iterator<uint64_t>(0),
        thrust::counting_iterator<uint64_t>(length),
        d_seq.begin(),
        [seed] __device__(uint64_t idx) { return detail::randomProteinSymbol(idx, seed); }
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

template <uint64_t K, typename Alphabet = bloom::DnaAlphabet>
__global__ void encodePackedKmersKernel(const char* sequence, uint64_t numKmers, uint64_t* output) {
    constexpr uint64_t symbolBits = cuda::std::bit_width(Alphabet::symbolCount - 1);
    constexpr uint64_t symbolMask = (uint64_t{1} << symbolBits) - 1;
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numKmers) {
        return;
    }

    uint64_t packed = 0;
    for (uint64_t i = 0; i < K; ++i) {
        const uint8_t encoded = Alphabet::encode(sequence[idx + i]);
        packed = (packed << symbolBits) | (encoded & symbolMask);
    }
    output[idx] = packed;
}

template <uint64_t K, typename Alphabet = bloom::DnaAlphabet>
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
    const uint64_t gridSize = cuda::ceil_div(numKmers, blockSize);
    encodePackedKmersKernel<K, Alphabet>
        <<<gridSize, blockSize, 0, stream>>>(d_sequence, numKmers, d_output);
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
        d_throughputPackedKmers.resize(numKmers);
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

        BLOOM_CUDA_CALL(cudaDeviceSynchronize());

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

    cache.clear();

    BenchmarkData<K> data;
    data.sequenceLength = length;
    data.generateThroughputData();
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());

    return cache.emplace(length, std::move(data)).first->second;
}

template <typename Config>
class SuperBloomFixtureBase : public benchmark::Fixture {
   public:
    static constexpr uint64_t k = Config::k;

    void setupCommon(const benchmark::State& state) {
        sequenceLength = static_cast<uint64_t>(state.range(0));
        benchData = &getBenchmarkData<Config::k>(sequenceLength);

        numKmers = benchData->numKmers;
        numSmers = sequenceLength - Config::s + 1;

        const uint64_t requestedFilterBits = cuda::std::bit_ceil(numKmers * 16);
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

    void setCounters(benchmark::State& state) const {
        setBenchmarkCounters(state, filterMemory, sequenceLength, numKmers);
        state.counters["s"] = benchmark::Counter(static_cast<double>(Config::s));
        state.counters["hashes"] = benchmark::Counter(static_cast<double>(Config::hashCount));
    }

    uint64_t sequenceLength{};
    uint64_t numKmers{};
    uint64_t numSmers{};
    uint64_t filterMemory{};
    BenchmarkData<Config::k>* benchData{};
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<bloom::Filter<Config>> filter;
    GPUTimer timer;
};

template <typename Config>
class SuperBloomConfigFixture : public SuperBloomFixtureBase<Config> {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        this->setupCommon(state);
    }

    void TearDown(const benchmark::State&) override {
        this->tearDownCommon();
    }
};

template <typename Fixture>
void runSuperBloomInsert(Fixture& fixture, benchmark::State& state) {
    for (auto _ : state) {
        fixture.filter->clear();
        BLOOM_CUDA_CALL(cudaDeviceSynchronize());

        fixture.timer.start();
        benchmark::DoNotOptimize(fixture.filter->insertSequenceDevice(
            bloom::device_span<const char>{
                thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
                fixture.sequenceLength
            }
        ));
        const double elapsed = fixture.timer.elapsed();
        state.SetIterationTime(elapsed);
    }
    fixture.setCounters(state);
}

template <typename Fixture>
void runSuperBloomQuery(Fixture& fixture, benchmark::State& state) {
    fixture.filter->clear();
    benchmark::DoNotOptimize(fixture.filter->insertSequenceDevice(
        bloom::device_span<const char>{
            thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
            fixture.sequenceLength
        }
    ));
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());

    for (auto _ : state) {
        fixture.timer.start();
        fixture.filter->containsSequenceDevice(
            bloom::device_span<const char>{
                thrust::raw_pointer_cast(fixture.benchData->d_throughputSequence.data()),
                fixture.sequenceLength
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
void runSuperBloomFpr(Fixture& fixture, benchmark::State& state) {
    fixture.benchData->ensureFprData();

    fixture.filter->clear();
    benchmark::DoNotOptimize(fixture.filter->insertSequenceDevice(
        bloom::device_span<const char>{
            thrust::raw_pointer_cast(fixture.benchData->d_fprInsertSequence.data()),
            fixture.sequenceLength
        }
    ));
    BLOOM_CUDA_CALL(cudaDeviceSynchronize());

    uint64_t falsePositives = 0;
    for (auto _ : state) {
        fixture.timer.start();
        fixture.filter->containsSequenceDevice(
            bloom::device_span<const char>{
                thrust::raw_pointer_cast(fixture.benchData->d_zeroOverlapSequence.data()),
                fixture.sequenceLength
            },
            bloom::device_span<uint8_t>{
                thrust::raw_pointer_cast(fixture.d_output.data()), fixture.d_output.size()
            }
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

}  // namespace benchmark_common

#define BENCHMARK_SUPERBLOOM_CONFIG_SYMBOL(K, S, M, H) SuperBloom_K##K##_S##S##_M##M##_H##H##_Config
#define BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H) \
    SuperBloom_K##K##_S##S##_M##M##_H##H##_Fixture

#define BENCHMARK_DEFINE_SUPERBLOOM_CONFIG_AND_FIXTURE(K, S, M, H)                         \
    using BENCHMARK_SUPERBLOOM_CONFIG_SYMBOL(K, S, M, H) = bloom::Config<K, S, M, H, 256>; \
    using BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H) =                                \
        benchmark_common::SuperBloomConfigFixture<BENCHMARK_SUPERBLOOM_CONFIG_SYMBOL(K, S, M, H)>;

#define BENCHMARK_DEFINE_SUPERBLOOM_ALL(FixtureName)                    \
    BENCHMARK_DEFINE_F(FixtureName, Insert)(benchmark::State & state) { \
        benchmark_common::runSuperBloomInsert(*this, state);            \
    }                                                                   \
    BENCHMARK_DEFINE_F(FixtureName, Query)(benchmark::State & state) {  \
        benchmark_common::runSuperBloomQuery(*this, state);             \
    }                                                                   \
    BENCHMARK_DEFINE_F(FixtureName, FPR)(benchmark::State & state) {    \
        benchmark_common::runSuperBloomFpr(*this, state);               \
    }

#define BENCHMARK_DEFINE_SUPERBLOOM_FPR_ONLY(FixtureName)            \
    BENCHMARK_DEFINE_F(FixtureName, FPR)(benchmark::State & state) { \
        benchmark_common::runSuperBloomFpr(*this, state);            \
    }

#define BENCHMARK_REGISTER_SUPERBLOOM_ALL(FixtureName) \
    REGISTER_BENCHMARK(FixtureName, Insert);           \
    REGISTER_BENCHMARK(FixtureName, Query);            \
    REGISTER_BENCHMARK(FixtureName, FPR);

#define BENCHMARK_REGISTER_SUPERBLOOM_FPR_ONLY(FixtureName) REGISTER_BENCHMARK(FixtureName, FPR);

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
