#pragma once

#include <cuda/__cmath/ceil_div.h>
#include <cuda_runtime.h>

#include <cub/warp/warp_reduce.cuh>

#include <cstdint>

#include <cusbf/config.cuh>
#include <cusbf/detail/dense_packed.cuh>
#include <cusbf/detail/filter_common.cuh>
#include <cusbf/detail/filter_impl.cuh>
#include <cusbf/detail/sequence_kmer.cuh>
#include <cusbf/filter_ref.cuh>

namespace cusbf::detail {

/// @brief Maximum @c uint64_t words loaded for a dense-packed insert block tile.
template <typename Config>
[[nodiscard]] constexpr uint64_t dense_packed_insert_word_tile_capacity() {
    constexpr uint64_t tile_symbols = Config::cudaBlockSize + Config::k - 1;
    return cuda::ceil_div(tile_symbols, dense_packed_symbols_per_word<Config>());
}

/// @brief Maximum @c uint64_t words loaded for a dense-packed query block tile.
template <typename Config>
[[nodiscard]] constexpr uint64_t dense_packed_query_word_tile_capacity() {
    constexpr uint64_t tile_symbols =
        Config::cudaBlockSize * kContainsSequenceStride + Config::k - 1;
    return cuda::ceil_div(tile_symbols, dense_packed_symbols_per_word<Config>());
}

/**
 * @brief Shared query path after a block symbol tile has been prepared.
 *
 * Used by both byte-sequence and dense-packed symbol kernels.
 */
template <typename Config, uint32_t k_stride>
__device__ __forceinline__ void contains_kmers_from_symbol_tile(
    const uint8_t* sequence_tile,
    uint64_t block_start_kmer,
    uint64_t block_kmers,
    bool block_all_valid,
    const filter_block<Config>* shards,
    uint64_t num_shards,
    uint8_t* output
) {
    const uint64_t thread_offset = static_cast<uint64_t>(threadIdx.x) * k_stride;
    if (thread_offset >= block_kmers) {
        return;
    }

    const uint32_t kmer_valid_mask = build_stride_kmer_valid_mask<k_stride, Config>(
        thread_offset, block_kmers, block_all_valid, sequence_tile
    );

    uint64_t packed_kmer = pack_kmer_from_tile<Config, Config::k>(sequence_tile, thread_offset);
    filter_ref<Config> ref;

    for (uint32_t s = 0; s < k_stride; ++s) {
        const uint64_t local_idx = thread_offset + s;
        if (local_idx >= block_kmers) {
            break;
        }

        const uint64_t kmer_index = block_start_kmer + local_idx;

        if (s > 0) {
            packed_kmer = advance_packed_kmer<Config, Config::k>(
                packed_kmer, sequence_tile[local_idx + Config::k - 1]
            );
        }

        if (!(kmer_valid_mask & (1u << s))) {
            output[kmer_index] = 0;
            continue;
        }

        const uint64_t minimizer_hash = packed_kmer_minimizer_hash<Config>(packed_kmer);

        const auto shard_idx =
            static_cast<uint32_t>(filter_ref<Config>::shard_index(minimizer_hash, num_shards));
        const uint32_t peers = __match_any_sync(0xFFFFFFFFu, shard_idx);
        const int leader = __ffs(static_cast<int>(peers)) - 1;

        uint64_t w[4];
        if (static_cast<int>(threadIdx.x & 31u) == leader) {
            load_shard_words4<Config>(shards, shard_idx, w);
        }
        w[0] = __shfl_sync(peers, w[0], leader);
        w[1] = __shfl_sync(peers, w[1], leader);
        w[2] = __shfl_sync(peers, w[2], leader);
        w[3] = __shfl_sync(peers, w[3], leader);

        const bool present = ref.sectorized_contains_packed_kmer(packed_kmer, w);
        output[kmer_index] = present;
    }
}

/**
 * @brief Shared insert path after a block symbol tile has been prepared.
 *
 * Used by both byte-sequence and dense-packed symbol kernels.
 */
template <typename Config, uint32_t warps_per_block>
__device__ __forceinline__ void insert_kmers_from_symbol_tile(
    const uint8_t* sequence_tile,
    uint64_t block_start_kmer,
    uint64_t block_kmers,
    bool block_all_valid,
    filter_block<Config>* shards,
    uint64_t num_shards,
    cub::WarpReduce<uint64_t>::TempStorage reduce_storage[warps_per_block][4]
) {
    constexpr uint32_t warp_size = 32;

    const auto local_kmer_index = static_cast<uint64_t>(threadIdx.x);
    const bool in_range = local_kmer_index < block_kmers;

    bool active = in_range;
    if (active && !block_all_valid) {
        active = kmer_is_valid<Config>(sequence_tile, local_kmer_index);
    }

    uint64_t minimizer_hash = 0;
    uint64_t word_mask0 = 0;
    uint64_t word_mask1 = 0;
    uint64_t word_mask2 = 0;
    uint64_t word_mask3 = 0;

    if (active) {
        const uint64_t packed_kmer =
            pack_kmer_from_tile<Config, Config::k>(sequence_tile, local_kmer_index);
        minimizer_hash = packed_kmer_minimizer_hash<Config>(packed_kmer);

        uint64_t h_s = packed_kmer_smer_hash<Config>(packed_kmer, 0);
        filter_block<Config>::sectorizedHashToMasks(
            h_s, word_mask0, word_mask1, word_mask2, word_mask3
        );
        _Pragma("unroll")
        for (uint64_t smer_offset = 1; smer_offset < Config::findereSpan; ++smer_offset) {
            h_s = packed_kmer_smer_hash<Config>(packed_kmer, smer_offset);
            filter_block<Config>::sectorizedHashToMasks(
                h_s, word_mask0, word_mask1, word_mask2, word_mask3
            );
        }
    }

    const auto shard_idx =
        static_cast<uint32_t>(active ? (minimizer_hash & (num_shards - 1)) : ~threadIdx.x);

    const uint32_t lane = threadIdx.x & (warp_size - 1);
    const uint32_t warp_idx = threadIdx.x / warp_size;
    const uint32_t prev_shard_idx = __shfl_up_sync(0xffffffff, shard_idx, 1);
    const bool run_head = (lane == 0) || (shard_idx != prev_shard_idx);
    const BitwiseOr<uint64_t> bitwise_or{};

    using WarpReduceWord = cub::WarpReduce<uint64_t>;
    word_mask0 = WarpReduceWord(reduce_storage[warp_idx][0])
                     .HeadSegmentedReduce(word_mask0, run_head, bitwise_or);
    word_mask1 = WarpReduceWord(reduce_storage[warp_idx][1])
                     .HeadSegmentedReduce(word_mask1, run_head, bitwise_or);
    word_mask2 = WarpReduceWord(reduce_storage[warp_idx][2])
                     .HeadSegmentedReduce(word_mask2, run_head, bitwise_or);
    word_mask3 = WarpReduceWord(reduce_storage[warp_idx][3])
                     .HeadSegmentedReduce(word_mask3, run_head, bitwise_or);

    if (run_head && active) {
        filter_ref<Config> ref;
        ref.apply_word_masks(shards[shard_idx], word_mask0, word_mask1, word_mask2, word_mask3);
    }
}

/**
 * @brief Query kernel: one byte per k-mer (1 = present, 0 = absent or invalid).
 *
 * Threads stride @c kContainsSequenceStride k-mers, warps sharing a shard load it once.
 */
template <typename Config>
__global__ __launch_bounds__(Config::cudaBlockSize, 6) void contains_sequence_kmers_kernel(
    const char* sequence,
    uint64_t num_kmers,
    const filter_block<Config>* shards,
    uint64_t num_shards,
    uint8_t* output
) {
    constexpr uint32_t k_stride = kContainsSequenceStride;
    constexpr uint64_t sequence_tile_bases = Config::cudaBlockSize * k_stride + Config::k - 1;

    __shared__ uint8_t sequence_tile[sequence_tile_bases];

    const uint64_t block_start_kmer =
        static_cast<uint64_t>(blockIdx.x) * Config::cudaBlockSize * k_stride;
    if (block_start_kmer >= num_kmers) {
        return;
    }

    const uint64_t block_kmers =
        min(Config::cudaBlockSize * k_stride, num_kmers - block_start_kmer);

    const bool block_all_valid = prepare_sequence_hash_tiles<Config>(
        sequence, block_start_kmer, block_kmers, sequence_tile
    );

    contains_kmers_from_symbol_tile<Config, k_stride>(
        sequence_tile,
        block_start_kmer,
        block_kmers,
        block_all_valid,
        shards,
        num_shards,
        output
    );
}

/**
 * @brief Insert kernel: sectorized Bloom updates grouped by minimizer shard.
 *
 * Warp-local segmented reduction merges consecutive k-mers targeting the same shard.
 */
template <typename Config>
__global__ void insert_sequence_kmers_kernel(
    const char* sequence,
    uint64_t num_kmers,
    filter_block<Config>* shards,
    uint64_t num_shards
) {
    constexpr uint64_t sequence_tile_bases = Config::cudaBlockSize + Config::k - 1;
    constexpr uint32_t warps_per_block = Config::cudaBlockSize / 32;

    using WarpReduceWord = cub::WarpReduce<uint64_t>;

    __shared__ uint8_t sequence_tile[sequence_tile_bases];
    __shared__ typename WarpReduceWord::TempStorage reduce_storage[warps_per_block][4];

    const uint64_t block_start_kmer = static_cast<uint64_t>(blockIdx.x) * Config::cudaBlockSize;
    if (block_start_kmer >= num_kmers) {
        return;
    }

    const uint64_t block_kmers = min(Config::cudaBlockSize, num_kmers - block_start_kmer);

    const bool block_all_valid = prepare_sequence_hash_tiles<Config>(
        sequence, block_start_kmer, block_kmers, sequence_tile
    );

    insert_kmers_from_symbol_tile<Config, warps_per_block>(
        sequence_tile,
        block_start_kmer,
        block_kmers,
        block_all_valid,
        shards,
        num_shards,
        reduce_storage
    );
}

/**
 * @brief Query kernel for a dense packed symbol buffer (@ref DensePackedKmerInput).
 */
template <typename Config>
__global__ __launch_bounds__(Config::cudaBlockSize, 6) void contains_dense_packed_kmers_kernel(
    const uint64_t* words,
    uint64_t num_kmers,
    const filter_block<Config>* shards,
    uint64_t num_shards,
    uint8_t* output
) {
    constexpr uint32_t k_stride = kContainsSequenceStride;
    constexpr uint64_t sequence_tile_bases = Config::cudaBlockSize * k_stride + Config::k - 1;
    constexpr uint64_t word_tile_capacity = dense_packed_query_word_tile_capacity<Config>();

    __shared__ uint64_t word_tile[word_tile_capacity];
    __shared__ uint8_t sequence_tile[sequence_tile_bases];

    const uint64_t block_start_kmer =
        static_cast<uint64_t>(blockIdx.x) * Config::cudaBlockSize * k_stride;
    if (block_start_kmer >= num_kmers) {
        return;
    }

    const uint64_t block_kmers =
        min(Config::cudaBlockSize * k_stride, num_kmers - block_start_kmer);

    const bool block_all_valid = prepare_dense_packed_tiles<Config>(
        words, block_start_kmer, block_kmers, word_tile, sequence_tile
    );

    contains_kmers_from_symbol_tile<Config, k_stride>(
        sequence_tile,
        block_start_kmer,
        block_kmers,
        block_all_valid,
        shards,
        num_shards,
        output
    );
}

/**
 * @brief Insert kernel for a dense packed symbol buffer (@ref DensePackedKmerInput).
 */
template <typename Config>
__global__ void insert_dense_packed_kmers_kernel(
    const uint64_t* words,
    uint64_t num_kmers,
    filter_block<Config>* shards,
    uint64_t num_shards
) {
    constexpr uint64_t sequence_tile_bases = Config::cudaBlockSize + Config::k - 1;
    constexpr uint32_t warps_per_block = Config::cudaBlockSize / 32;
    constexpr uint64_t word_tile_capacity = dense_packed_insert_word_tile_capacity<Config>();

    using WarpReduceWord = cub::WarpReduce<uint64_t>;

    __shared__ uint64_t word_tile[word_tile_capacity];
    __shared__ uint8_t sequence_tile[sequence_tile_bases];
    __shared__ typename WarpReduceWord::TempStorage reduce_storage[warps_per_block][4];

    const uint64_t block_start_kmer = static_cast<uint64_t>(blockIdx.x) * Config::cudaBlockSize;
    if (block_start_kmer >= num_kmers) {
        return;
    }

    const uint64_t block_kmers = min(Config::cudaBlockSize, num_kmers - block_start_kmer);

    const bool block_all_valid = prepare_dense_packed_tiles<Config>(
        words, block_start_kmer, block_kmers, word_tile, sequence_tile
    );

    insert_kmers_from_symbol_tile<Config, warps_per_block>(
        sequence_tile,
        block_start_kmer,
        block_kmers,
        block_all_valid,
        shards,
        num_shards,
        reduce_storage
    );
}

}  // namespace cusbf::detail
