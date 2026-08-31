#include <benchmark/benchmark.h>
#include <cuda/__cmath/ceil_div.h>
#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <cub/warp/warp_reduce.cuh>
#include <cuda/std/bit>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <cusbf/config.cuh>
#include <cusbf/detail/filter_common.cuh>
#include <cusbf/detail/filter_impl.cuh>
#include <cusbf/detail/sequence_kmer.cuh>
#include <cusbf/filter_ref.cuh>
#include <cusbf/helpers.cuh>

#include "benchmark_common.cuh"

namespace bm = benchmark;
namespace detail = cusbf::detail;

using AblationConfig = cusbf::Config<31, 28, 16, 4, 256>;
using AblationBlock = cusbf::filter_block<AblationConfig>;

namespace {

template <bool Sectorized, bool SharedMemoryTiling, bool WarpShardSharing, bool SegmentedReduction>
struct DesignPolicy {
    static constexpr bool sectorized = Sectorized;
    static constexpr bool shared_memory_tiling = SharedMemoryTiling;
    static constexpr bool warp_shard_sharing = WarpShardSharing;
    static constexpr bool segmented_reduction = SegmentedReduction;
};

using BaselineInsertPolicy = DesignPolicy<false, false, false, false>;
using SectorizedInsertPolicy = DesignPolicy<true, false, false, false>;
using TiledInsertPolicy = DesignPolicy<false, true, false, false>;
using SectorizedTiledInsertPolicy = DesignPolicy<true, true, false, false>;
using SegmentedInsertPolicy = DesignPolicy<false, false, false, true>;
using SectorizedSegmentedInsertPolicy = DesignPolicy<true, false, false, true>;
using TiledSegmentedInsertPolicy = DesignPolicy<false, true, false, true>;
using FullInsertPolicy = DesignPolicy<true, true, false, true>;
using BaselineQueryPolicy = DesignPolicy<false, false, true, false>;
using SectorizedQueryPolicy = DesignPolicy<true, false, true, false>;
using TiledQueryPolicy = DesignPolicy<false, true, true, false>;
using FullQueryPolicy = DesignPolicy<true, true, true, false>;

template <typename Config>
__device__ __forceinline__ uint64_t
pack_kmer_from_global(const char* sequence, uint64_t start, bool& valid) {
    uint64_t packed = 0;
    valid = true;
    _Pragma("unroll")
    for (uint64_t i = 0; i < Config::k; ++i) {
        const uint8_t symbol =
            Config::Alphabet::encode(sequence + (start + i) * Config::symbolWidth);
        packed = (packed << Config::symbolBits) | (symbol & Config::symbolMask);
        valid &= symbol != Config::Alphabet::invalidSymbol;
    }
    return packed;
}

template <typename Policy, typename Config>
__device__ __forceinline__ void hash_to_masks(
    uint64_t base_hash,
    uint64_t& mask0,
    uint64_t& mask1,
    uint64_t& mask2,
    uint64_t& mask3
) {
    if constexpr (Policy::sectorized) {
        cusbf::filter_block<Config>::sectorizedHashToMasks(base_hash, mask0, mask1, mask2, mask3);
    } else {
        detail::forEachHashIndex<Config>(
            [&]<uint64_t HashIndex>(std::integral_constant<uint64_t, HashIndex>) {
                const uint64_t address =
                    (base_hash * detail::multiplicativeSaltLiteral<HashIndex>()) >> 56;
                const uint64_t bit = uint64_t{1} << (address & 63);
                switch (address >> 6) {
                    case 0:
                        mask0 |= bit;
                        break;
                    case 1:
                        mask1 |= bit;
                        break;
                    case 2:
                        mask2 |= bit;
                        break;
                    default:
                        mask3 |= bit;
                }
            }
        );
    }
}

template <typename Policy, typename Config>
__device__ __forceinline__ bool
contains_packed_kmer(uint64_t packed_kmer, const uint64_t* shard_words) {
    if constexpr (Policy::sectorized) {
        return cusbf::filter_ref<Config>::sectorized_contains_packed_kmer(packed_kmer, shard_words);
    } else {
        bool present = true;
        _Pragma("unroll")
        for (uint64_t offset = 0; offset < Config::findereSpan; ++offset) {
            const uint64_t base_hash = detail::packed_kmer_smer_hash<Config>(packed_kmer, offset);
            detail::forEachHashIndex<Config>(
                [&]<uint64_t HashIndex>(std::integral_constant<uint64_t, HashIndex>) {
                    const uint64_t address =
                        (base_hash * detail::multiplicativeSaltLiteral<HashIndex>()) >> 56;
                    present &= ((shard_words[address >> 6] >> (address & 63)) & 1) != 0;
                }
            );
            if (!present) {
                break;
            }
        }
        return present;
    }
}

__device__ __forceinline__ void apply_masks(
    AblationBlock* shards,
    uint32_t shard_idx,
    uint64_t mask0,
    uint64_t mask1,
    uint64_t mask2,
    uint64_t mask3
) {
    cusbf::filter_ref<AblationConfig> ref;
    ref.apply_word_masks(shards[shard_idx], mask0, mask1, mask2, mask3);
}

template <typename Policy>
__device__ __forceinline__ void insert_one_thread(
    const char* sequence,
    const uint8_t* sequence_tile,
    uint64_t block_start_kmer,
    uint64_t block_kmers,
    bool block_all_valid,
    AblationBlock* shards,
    uint64_t num_shards,
    cub::WarpReduce<uint64_t>::TempStorage* reduce_storage
) {
    const uint64_t local_idx = threadIdx.x;
    bool active = local_idx < block_kmers;
    uint64_t packed_kmer = 0;

    if (active) {
        if constexpr (Policy::shared_memory_tiling) {
            active =
                block_all_valid || detail::kmer_is_valid<AblationConfig>(sequence_tile, local_idx);
            if (active) {
                packed_kmer = detail::pack_kmer_from_tile<AblationConfig, AblationConfig::k>(
                    sequence_tile, local_idx
                );
            }
        } else {
            packed_kmer = pack_kmer_from_global<AblationConfig>(
                sequence, block_start_kmer + local_idx, active
            );
        }
    }

    uint64_t minimizer_hash = 0;
    uint64_t masks[4]{};
    if (active) {
        minimizer_hash = detail::packed_kmer_minimizer_hash<AblationConfig>(packed_kmer);
        _Pragma("unroll")
        for (uint64_t offset = 0; offset < AblationConfig::findereSpan; ++offset) {
            hash_to_masks<Policy, AblationConfig>(
                detail::packed_kmer_smer_hash<AblationConfig>(packed_kmer, offset),
                masks[0],
                masks[1],
                masks[2],
                masks[3]
            );
        }
    }

    const uint32_t shard_idx =
        static_cast<uint32_t>(active ? (minimizer_hash & (num_shards - 1)) : ~threadIdx.x);

    if constexpr (Policy::segmented_reduction) {
        constexpr uint32_t warp_size = 32;
        const uint32_t lane = threadIdx.x & (warp_size - 1);
        const uint32_t warp_idx = threadIdx.x / warp_size;
        const uint32_t previous_shard = __shfl_up_sync(0xffffffff, shard_idx, 1);
        const bool run_head = lane == 0 || shard_idx != previous_shard;
        const detail::BitwiseOr<uint64_t> bitwise_or{};

        _Pragma("unroll")
        for (uint32_t word = 0; word < 4; ++word) {
            masks[word] = cub::WarpReduce<uint64_t>(reduce_storage[warp_idx * 4 + word])
                              .HeadSegmentedReduce(masks[word], run_head, bitwise_or);
        }
        if (run_head && active) {
            apply_masks(shards, shard_idx, masks[0], masks[1], masks[2], masks[3]);
        }
    } else if (active) {
        apply_masks(shards, shard_idx, masks[0], masks[1], masks[2], masks[3]);
    }
}

template <typename Policy>
__global__ void insert_ablation_kernel(
    const char* sequence,
    uint64_t num_kmers,
    AblationBlock* shards,
    uint64_t num_shards
) {
    constexpr uint64_t tile_size = AblationConfig::cudaBlockSize + AblationConfig::k - 1;
    constexpr uint32_t warps_per_block = AblationConfig::cudaBlockSize / 32;
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * AblationConfig::cudaBlockSize;
    if (block_start >= num_kmers) {
        return;
    }
    const uint64_t block_kmers =
        min(static_cast<uint64_t>(AblationConfig::cudaBlockSize), num_kmers - block_start);

    if constexpr (Policy::shared_memory_tiling) {
        __shared__ uint8_t sequence_tile[tile_size];
        const bool all_valid = detail::prepare_sequence_hash_tiles<AblationConfig>(
            sequence, block_start, block_kmers, sequence_tile
        );
        if constexpr (Policy::segmented_reduction) {
            __shared__ cub::WarpReduce<uint64_t>::TempStorage reduce_storage[warps_per_block * 4];
            insert_one_thread<Policy>(
                sequence,
                sequence_tile,
                block_start,
                block_kmers,
                all_valid,
                shards,
                num_shards,
                reduce_storage
            );
        } else {
            insert_one_thread<Policy>(
                sequence,
                sequence_tile,
                block_start,
                block_kmers,
                all_valid,
                shards,
                num_shards,
                nullptr
            );
        }
    } else {
        if constexpr (Policy::segmented_reduction) {
            __shared__ cub::WarpReduce<uint64_t>::TempStorage reduce_storage[warps_per_block * 4];
            insert_one_thread<Policy>(
                sequence,
                nullptr,
                block_start,
                block_kmers,
                false,
                shards,
                num_shards,
                reduce_storage
            );
        } else {
            insert_one_thread<Policy>(
                sequence, nullptr, block_start, block_kmers, false, shards, num_shards, nullptr
            );
        }
    }
}

template <typename Policy>
__device__ __forceinline__ void query_thread(
    const char* sequence,
    const uint8_t* sequence_tile,
    uint64_t block_start,
    uint64_t block_kmers,
    bool block_all_valid,
    const AblationBlock* shards,
    uint64_t num_shards,
    uint8_t* output
) {
    constexpr uint32_t stride = detail::kContainsSequenceStride;
    const uint64_t thread_offset = static_cast<uint64_t>(threadIdx.x) * stride;
    if (thread_offset >= block_kmers) {
        return;
    }

    uint32_t valid_mask = 0;
    uint64_t packed_kmer = 0;
    if constexpr (Policy::shared_memory_tiling) {
        valid_mask = detail::build_stride_kmer_valid_mask<stride, AblationConfig>(
            thread_offset, block_kmers, block_all_valid, sequence_tile
        );
        packed_kmer = detail::pack_kmer_from_tile<AblationConfig, AblationConfig::k>(
            sequence_tile, thread_offset
        );
    }

    for (uint32_t step = 0; step < stride; ++step) {
        const uint64_t local_idx = thread_offset + step;
        if (local_idx >= block_kmers) {
            break;
        }

        bool valid = true;
        if constexpr (Policy::shared_memory_tiling) {
            valid = (valid_mask & (1u << step)) != 0;
            if (step > 0) {
                packed_kmer = detail::advance_packed_kmer<AblationConfig, AblationConfig::k>(
                    packed_kmer, sequence_tile[local_idx + AblationConfig::k - 1]
                );
            }
        } else {
            packed_kmer =
                pack_kmer_from_global<AblationConfig>(sequence, block_start + local_idx, valid);
        }

        const uint64_t output_idx = block_start + local_idx;
        if (!valid) {
            output[output_idx] = 0;
            continue;
        }

        const auto shard_idx = static_cast<uint32_t>(
            detail::packed_kmer_minimizer_hash<AblationConfig>(packed_kmer) & (num_shards - 1)
        );
        uint64_t words[4];
        if constexpr (Policy::warp_shard_sharing) {
            const uint32_t active_lanes = __activemask();
            const uint32_t peers = __match_any_sync(active_lanes, shard_idx);
            const int leader = __ffs(static_cast<int>(peers)) - 1;
            if (static_cast<int>(threadIdx.x & 31u) == leader) {
                detail::load_shard_words4<AblationConfig>(shards, shard_idx, words);
            }
            _Pragma("unroll")
            for (auto& word : words) {
                word = __shfl_sync(peers, word, leader);
            }
        } else {
            detail::load_shard_words4<AblationConfig>(shards, shard_idx, words);
        }
        output[output_idx] = contains_packed_kmer<Policy, AblationConfig>(packed_kmer, words);
    }
}

template <typename Policy>
__global__ __launch_bounds__(AblationConfig::cudaBlockSize, 6) void query_ablation_kernel(
    const char* sequence,
    uint64_t num_kmers,
    const AblationBlock* shards,
    uint64_t num_shards,
    uint8_t* output
) {
    constexpr uint32_t stride = detail::kContainsSequenceStride;
    constexpr uint64_t tile_size = AblationConfig::cudaBlockSize * stride + AblationConfig::k - 1;
    const uint64_t block_start =
        static_cast<uint64_t>(blockIdx.x) * AblationConfig::cudaBlockSize * stride;
    if (block_start >= num_kmers) {
        return;
    }
    const uint64_t block_kmers =
        min(static_cast<uint64_t>(AblationConfig::cudaBlockSize * stride), num_kmers - block_start);

    if constexpr (Policy::shared_memory_tiling) {
        __shared__ uint8_t sequence_tile[tile_size];
        const bool all_valid = detail::prepare_sequence_hash_tiles<AblationConfig>(
            sequence, block_start, block_kmers, sequence_tile
        );
        query_thread<Policy>(
            sequence, sequence_tile, block_start, block_kmers, all_valid, shards, num_shards, output
        );
    } else {
        query_thread<Policy>(
            sequence, nullptr, block_start, block_kmers, false, shards, num_shards, output
        );
    }
}

template <typename Policy>
void launch_insert(
    const thrust::device_vector<char>& sequence,
    uint64_t num_kmers,
    thrust::device_vector<AblationBlock>& shards
) {
    const uint64_t grid = cuda::ceil_div(num_kmers, AblationConfig::cudaBlockSize);
    insert_ablation_kernel<Policy><<<grid, AblationConfig::cudaBlockSize>>>(
        thrust::raw_pointer_cast(sequence.data()),
        num_kmers,
        thrust::raw_pointer_cast(shards.data()),
        shards.size()
    );
    CUSBF_CUDA_CALL(cudaGetLastError());
}

template <typename Policy>
void launch_query(
    const thrust::device_vector<char>& sequence,
    uint64_t num_kmers,
    const thrust::device_vector<AblationBlock>& shards,
    thrust::device_vector<uint8_t>& output
) {
    constexpr uint64_t block_kmers =
        AblationConfig::cudaBlockSize * detail::kContainsSequenceStride;
    const uint64_t grid = cuda::ceil_div(num_kmers, block_kmers);
    query_ablation_kernel<Policy><<<grid, AblationConfig::cudaBlockSize>>>(
        thrust::raw_pointer_cast(sequence.data()),
        num_kmers,
        thrust::raw_pointer_cast(shards.data()),
        shards.size(),
        thrust::raw_pointer_cast(output.data())
    );
    CUSBF_CUDA_CALL(cudaGetLastError());
}

class DesignAblationFixture : public bm::Fixture {
    using bm::Fixture::SetUp;
    using bm::Fixture::TearDown;

   public:
    void SetUp(const bm::State& state) override {
        num_symbols = static_cast<uint64_t>(state.range(0));
        num_kmers = num_symbols - AblationConfig::k + 1;
        const uint64_t filter_bits =
            cuda::std::bit_ceil(num_kmers * benchmark_common::g_fastxBitsPerItem);
        shards.resize(filter_bits / AblationConfig::filterBlockBits);
        sequence.resize(num_symbols);
        output.resize(num_kmers);
        benchmark_common::gpuGenerateDna(sequence, num_symbols);
        clear();
    }

    void TearDown(const bm::State&) override {
        thrust::device_vector<char>{}.swap(sequence);
        thrust::device_vector<AblationBlock>{}.swap(shards);
        thrust::device_vector<uint8_t>{}.swap(output);
    }

    void clear() {
        CUSBF_CUDA_CALL(cudaMemset(
            thrust::raw_pointer_cast(shards.data()), 0, shards.size() * sizeof(AblationBlock)
        ));
    }

    template <typename Policy>
    bool validate() {
        clear();
        launch_insert<Policy>(sequence, num_kmers, shards);
        launch_query<Policy>(sequence, num_kmers, shards, output);
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
        return static_cast<uint64_t>(thrust::count(output.begin(), output.end(), uint8_t{1})) ==
               num_kmers;
    }

    void counters(bm::State& state) const {
        benchmark_common::filter_benchmark::setFilterBenchmarkCounters(
            state, shards.size() * sizeof(AblationBlock), num_kmers
        );
        state.counters["num_symbols"] = bm::Counter(static_cast<double>(num_symbols));
    }

    uint64_t num_symbols{};
    uint64_t num_kmers{};
    thrust::device_vector<char> sequence;
    thrust::device_vector<AblationBlock> shards;
    thrust::device_vector<uint8_t> output;
    benchmark_common::GPUTimer timer;
};

template <typename Policy>
void run_insert(DesignAblationFixture& fixture, bm::State& state) {
    if (!fixture.validate<Policy>()) {
        state.SkipWithError("ablation insert/query self-check failed");
        return;
    }
    for (auto _ : state) {
        fixture.clear();
        CUSBF_CUDA_CALL(cudaDeviceSynchronize());
        fixture.timer.start();
        launch_insert<Policy>(fixture.sequence, fixture.num_kmers, fixture.shards);
        state.SetIterationTime(fixture.timer.elapsed());
    }
    fixture.counters(state);
}

template <typename Policy>
void run_query(DesignAblationFixture& fixture, bm::State& state) {
    if (!fixture.validate<Policy>()) {
        state.SkipWithError("ablation insert/query self-check failed");
        return;
    }
    for (auto _ : state) {
        fixture.timer.start();
        launch_query<Policy>(fixture.sequence, fixture.num_kmers, fixture.shards, fixture.output);
        state.SetIterationTime(fixture.timer.elapsed());
        bm::DoNotOptimize(thrust::raw_pointer_cast(fixture.output.data()));
    }
    fixture.counters(state);
}

#define DEFINE_ABLATION(Name, Policy, operation)                         \
    BENCHMARK_DEFINE_F(DesignAblationFixture, Name)(bm::State & state) { \
        auto& fixture = *static_cast<DesignAblationFixture*>(this);      \
        run_##operation<Policy>(fixture, state);                         \
    }

DEFINE_ABLATION(BaselineInsert, BaselineInsertPolicy, insert)
DEFINE_ABLATION(SectorizedInsert, SectorizedInsertPolicy, insert)
DEFINE_ABLATION(TiledInsert, TiledInsertPolicy, insert)
DEFINE_ABLATION(SectorizedTiledInsert, SectorizedTiledInsertPolicy, insert)
DEFINE_ABLATION(SegmentedInsert, SegmentedInsertPolicy, insert)
DEFINE_ABLATION(SectorizedSegmentedInsert, SectorizedSegmentedInsertPolicy, insert)
DEFINE_ABLATION(TiledSegmentedInsert, TiledSegmentedInsertPolicy, insert)
DEFINE_ABLATION(FullInsert, FullInsertPolicy, insert)
DEFINE_ABLATION(BaselineQuery, BaselineQueryPolicy, query)
DEFINE_ABLATION(SectorizedQuery, SectorizedQueryPolicy, query)
DEFINE_ABLATION(TiledQuery, TiledQueryPolicy, query)
DEFINE_ABLATION(FullQuery, FullQueryPolicy, query)

#define REGISTER_ABLATION(Name)                       \
    BENCHMARK_REGISTER_F(DesignAblationFixture, Name) \
        ->Arg(1ULL << 22)                             \
        ->Arg(1ULL << 28)                             \
        ->Unit(bm::kMillisecond)                      \
        ->UseManualTime()                             \
        ->Iterations(10)                              \
        ->Repetitions(5)                              \
        ->ReportAggregatesOnly(true)

REGISTER_ABLATION(BaselineInsert);
REGISTER_ABLATION(SectorizedInsert);
REGISTER_ABLATION(TiledInsert);
REGISTER_ABLATION(SectorizedTiledInsert);
REGISTER_ABLATION(SegmentedInsert);
REGISTER_ABLATION(SectorizedSegmentedInsert);
REGISTER_ABLATION(TiledSegmentedInsert);
REGISTER_ABLATION(FullInsert);
REGISTER_ABLATION(BaselineQuery);
REGISTER_ABLATION(SectorizedQuery);
REGISTER_ABLATION(TiledQuery);
REGISTER_ABLATION(FullQuery);

#undef REGISTER_ABLATION
#undef DEFINE_ABLATION

}  // namespace

STANDARD_BENCHMARK_MAIN()
