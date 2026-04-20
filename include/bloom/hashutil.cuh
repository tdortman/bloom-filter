#pragma once

#include <cuda/std/bit>

#include <cstddef>
#include <cstdint>
#include <cstring>

/**
 * XXHash_64 implementation from
 * https://github.com/Cyan4973/xxHash
 * -----------------------------------------------------------------------------
 * xxHash - Extremely Fast Hash algorithm
 * Header File
 * Copyright (C) 2012-2021 Yann Collet
 *
 * BSD 2-Clause License (https://www.opensource.org/licenses/bsd-license.php)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following disclaimer
 *      in the documentation and/or other materials provided with the
 *      distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * You can contact the author at:
 *   - xxHash homepage: https://www.xxhash.com
 *   - xxHash source repository: https://github.com/Cyan4973/xxHash
 */
namespace xxhash {

constexpr uint64_t PRIME64_1 = 11400714785074694791ULL;
constexpr uint64_t PRIME64_2 = 14029467366897019727ULL;
constexpr uint64_t PRIME64_3 = 1609587929392839161ULL;
constexpr uint64_t PRIME64_4 = 9650029242287828579ULL;
constexpr uint64_t PRIME64_5 = 2870177450012600261ULL;

/// @brief Rotates @p x left by @p r bits.
constexpr __host__ __device__ __forceinline__ uint64_t rotl64(uint64_t x, int8_t r) {
    return cuda::std::rotl(x, static_cast<int>(r));
}

/// @brief Loads a chunk of type @p T from @p data at byte offset @c index*sizeof(T).
template <typename T>
__host__ __device__ __forceinline__ T load_chunk(const uint8_t* data, uint64_t index) {
    T chunk;
    memcpy(&chunk, data + index * sizeof(T), sizeof(T));
    return chunk;
}

/// @brief Applies the xxHash-64 final mixing (avalanche) step.
constexpr __host__ __device__ __forceinline__ uint64_t finalize(uint64_t h) {
    h ^= h >> 33;
    h *= PRIME64_2;
    h ^= h >> 29;
    h *= PRIME64_3;
    h ^= h >> 32;
    return h;
}

/**
 * @brief Computes the xxHash-64 digest of a value.
 *
 * @tparam T     Type of the value; hashed as raw bytes.
 * @param  key   Value to hash.
 * @param  seed  Optional seed (default 0).
 * @return 64-bit hash digest.
 */
template <typename T>
__host__ __device__ inline uint64_t xxhash64(const T& key, uint64_t seed = 0) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(&key);
    uint64_t size = sizeof(T);
    uint64_t offset = 0;
    uint64_t h64;

    // Process 32-byte chunks
    if (size >= 32) {
        uint64_t limit = size - 32;
        uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t v2 = seed + PRIME64_2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - PRIME64_1;

        do {
            const uint64_t pipeline_offset = offset / 8;
            v1 += load_chunk<uint64_t>(bytes, pipeline_offset + 0) * PRIME64_2;
            v1 = rotl64(v1, 31);
            v1 *= PRIME64_1;
            v2 += load_chunk<uint64_t>(bytes, pipeline_offset + 1) * PRIME64_2;
            v2 = rotl64(v2, 31);
            v2 *= PRIME64_1;
            v3 += load_chunk<uint64_t>(bytes, pipeline_offset + 2) * PRIME64_2;
            v3 = rotl64(v3, 31);
            v3 *= PRIME64_1;
            v4 += load_chunk<uint64_t>(bytes, pipeline_offset + 3) * PRIME64_2;
            v4 = rotl64(v4, 31);
            v4 *= PRIME64_1;
            offset += 32;
        } while (offset <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);

        v1 *= PRIME64_2;
        v1 = rotl64(v1, 31);
        v1 *= PRIME64_1;
        h64 ^= v1;
        h64 = h64 * PRIME64_1 + PRIME64_4;

        v2 *= PRIME64_2;
        v2 = rotl64(v2, 31);
        v2 *= PRIME64_1;
        h64 ^= v2;
        h64 = h64 * PRIME64_1 + PRIME64_4;

        v3 *= PRIME64_2;
        v3 = rotl64(v3, 31);
        v3 *= PRIME64_1;
        h64 ^= v3;
        h64 = h64 * PRIME64_1 + PRIME64_4;

        v4 *= PRIME64_2;
        v4 = rotl64(v4, 31);
        v4 *= PRIME64_1;
        h64 ^= v4;
        h64 = h64 * PRIME64_1 + PRIME64_4;
    } else {
        h64 = seed + PRIME64_5;
    }

    h64 += size;

    // Process remaining 8-byte chunks
    if ((size % 32) >= 8) {
        for (; offset <= size - 8; offset += 8) {
            uint64_t k1 = load_chunk<uint64_t>(bytes, offset / 8) * PRIME64_2;
            k1 = rotl64(k1, 31) * PRIME64_1;
            h64 ^= k1;
            h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        }
    }

    // Process remaining 4-byte chunks
    if ((size % 8) >= 4) {
        for (; offset <= size - 4; offset += 4) {
            h64 ^= (load_chunk<uint32_t>(bytes, offset / 4) & 0xffffffffULL) * PRIME64_1;
            h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        }
    }

    // Process remaining bytes
    if (size % 4) {
        while (offset < size) {
            h64 ^= (bytes[offset] & 0xff) * PRIME64_5;
            h64 = rotl64(h64, 11) * PRIME64_1;
            ++offset;
        }
    }

    return finalize(h64);
}

}  // namespace xxhash

namespace bloom::detail {

/**
 * @brief Fast 64-bit integer hash (non-cryptographic).
 *
 * One multiplicative step followed by an xorshift. Used to hash s-mer packed
 * representations for Bloom bit-position selection.
 *
 * @param key Input value.
 * @return Hashed value.
 */
constexpr __host__ __device__ __forceinline__ uint64_t hash64(uint64_t key) {
    key *= 0x9e3779b97f4a7c15ULL;
    key ^= key >> 33;
    return key;
}

/**
 * @brief Fast 64-bit hash sufficient for uniform minimizer selection.
 *
 * A single Knuth multiplicative step — provides enough uniformity for
 * shard selection without the full avalanche quality of @ref hash64.
 *
 * @param key Packed m-mer input.
 * @return Hash value used to select the minimum (minimizer).
 */
// sufficient for minimizer (shard) selection where only uniformity matters,
// not full avalanche quality.
constexpr __host__ __device__ __forceinline__ uint64_t minimizerHash64(uint64_t key) {
    return key * 0x9E3779B97F4A7C15ULL;
}

namespace nthash {

/// @brief ntHash seed for adenine (A).
constexpr uint64_t SEED_A = 0x3c8bfbb395c60474ULL;
/// @brief ntHash seed for cytosine (C).
constexpr uint64_t SEED_C = 0x3193c18562a02b4cULL;
/// @brief ntHash seed for guanine (G).
constexpr uint64_t SEED_G = 0x20323ed082572324ULL;
/// @brief ntHash seed for thymine (T).
constexpr uint64_t SEED_T = 0x295549f54be24456ULL;

/// @brief Rotates @p x left by one bit.
constexpr __host__ __device__ __forceinline__ uint64_t rol1(uint64_t x) {
    return cuda::std::rotl(x, 1);
}

/**
 * @brief Returns @p x rotated left by @p D bits (compile-time rotation amount).
 *
 * @tparam D Rotation amount.
 */
template <uint64_t D>
constexpr uint64_t rolnValue(uint64_t x) {
    return cuda::std::rotl(x, static_cast<int>(D % 64));
}

/**
 * @brief Four-element seed lookup table for ntHash rolling hash.
 *
 * Entries are indexed by the 2-bit base encoding (A=0, C=1, G=2, T=3).
 */
struct SeedTable {
    uint64_t v[4];
};

/// Thread-0 writes seeds + rolled-seed LUTs into shared memory.
/// Caller must barrier (e.g. __syncthreads) before reading.
template <uint64_t W_m, uint64_t W_s>
__device__ __forceinline__ void
initSeedTables(SeedTable& seeds, SeedTable& rolledM, SeedTable& rolledS) {
    if (threadIdx.x == 0) {
        seeds.v[0] = SEED_A;
        seeds.v[1] = SEED_C;
        seeds.v[2] = SEED_G;
        seeds.v[3] = SEED_T;
        rolledM.v[0] = rolnValue<W_m>(SEED_A);
        rolledM.v[1] = rolnValue<W_m>(SEED_C);
        rolledM.v[2] = rolnValue<W_m>(SEED_G);
        rolledM.v[3] = rolnValue<W_m>(SEED_T);
        rolledS.v[0] = rolnValue<W_s>(SEED_A);
        rolledS.v[1] = rolnValue<W_s>(SEED_C);
        rolledS.v[2] = rolnValue<W_s>(SEED_G);
        rolledS.v[3] = rolnValue<W_s>(SEED_T);
    }
}

/**
 * @brief Computes the initial ntHash value for a fixed-length window of pre-encoded bases.
 *
 * @tparam WindowLength  Window size in bases.
 * @param  encodedBases  Pre-encoded base array.
 * @param  start         Start position in @p encodedBases.
 * @param  seeds         Seed lookup table.
 * @return ntHash of the window.
 */
template <uint64_t WindowLength>
__device__ __forceinline__ uint64_t
baseHash(const uint8_t* encodedBases, uint64_t start, const SeedTable& seeds) {
    uint64_t h = 0;
    _Pragma("unroll")
    for (uint64_t i = 0; i < WindowLength; ++i) {
        h = rol1(h) ^ seeds.v[encodedBases[start + i] & 0x3u];
    }
    return h;
}

/**
 * @brief Incrementally updates an ntHash by one base step.
 *
 * Slides the window forward: removes @p baseOut from the front and adds @p
 * baseIn at the back.
 *
 * @tparam WindowLength  Window size in bases.
 * @param  h             Current hash value.
 * @param  baseOut       Pre-encoded base leaving the window (front).
 * @param  baseIn        Pre-encoded base entering the window (back).
 * @param  seeds         Seed lookup table.
 * @param  rolledSeeds   Seed table pre-rotated by @p WindowLength positions.
 * @return Updated ntHash.
 */
template <uint64_t WindowLength>
__device__ __forceinline__ uint64_t rollHash(
    uint64_t h,
    uint8_t baseOut,
    uint8_t baseIn,
    const SeedTable& seeds,
    const SeedTable& rolledSeeds
) {
    return rol1(h) ^ seeds.v[baseIn & 0x3u] ^ rolledSeeds.v[baseOut & 0x3u];
}

}  // namespace nthash

}  // namespace bloom::detail