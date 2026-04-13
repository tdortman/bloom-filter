#pragma once

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

__host__ __device__ __forceinline__ uint64_t rotl64(uint64_t x, int8_t r) {
    return (x << r) | (x >> (64 - r));
}

template <typename T>
__host__ __device__ __forceinline__ T load_chunk(const uint8_t* data, uint64_t index) {
    T chunk;
    memcpy(&chunk, data + index * sizeof(T), sizeof(T));
    return chunk;
}

__host__ __device__ __forceinline__ uint64_t finalize(uint64_t h) {
    h ^= h >> 33;
    h *= PRIME64_2;
    h ^= h >> 29;
    h *= PRIME64_3;
    h ^= h >> 32;
    return h;
}

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

__host__ __device__ __forceinline__ uint64_t hash64(uint64_t key) {
    key ^= key >> 30;
    key *= 0xbf58476d1ce4e5b9ULL;
    key ^= key >> 27;
    key *= 0x94d049bb133111ebULL;
    key ^= key >> 31;
    return key;
}

namespace nthash {

constexpr uint64_t SEED_A = 0x3c8bfbb395c60474ULL;
constexpr uint64_t SEED_C = 0x3193c18562a02b4cULL;
constexpr uint64_t SEED_G = 0x20323ed082572324ULL;
constexpr uint64_t SEED_T = 0x295549f54be24456ULL;

constexpr uint64_t SEED_TAB[4] = {SEED_A, SEED_C, SEED_G, SEED_T};

__host__ __device__ __forceinline__ uint64_t seedForBase(uint8_t base) {
    // clang-format off
    switch (base & 0x3u) {
        case 0: return SEED_A;
        case 1: return SEED_C;
        case 2: return SEED_G;
        default: return SEED_T;
    }
    // clang-format on
}

constexpr uint64_t srolValue(uint64_t x) {
    uint64_t m = ((x & 0x8000000000000000ULL) >> 30) | ((x & 0x100000000ULL) >> 32);
    return ((x << 1) & 0xFFFFFFFDFFFFFFFFULL) | m;
}

template <uint64_t D>
constexpr uint64_t srolnValue(uint64_t x) {
    uint64_t r = x;
    for (uint64_t i = 0; i < D; ++i) {
        r = srolValue(r);
    }
    return r;
}

__host__ __device__ __forceinline__ uint64_t srol(uint64_t x) {
    uint64_t m = ((x & 0x8000000000000000ULL) >> 30) | ((x & 0x100000000ULL) >> 32);
    return ((x << 1) & 0xFFFFFFFDFFFFFFFFULL) | m;
}

template <uint64_t WindowLength>
__device__ __forceinline__ uint64_t rolledSeed(uint8_t base) {
    constexpr uint64_t rs0 = srolnValue<WindowLength>(SEED_TAB[0]);
    constexpr uint64_t rs1 = srolnValue<WindowLength>(SEED_TAB[1]);
    constexpr uint64_t rs2 = srolnValue<WindowLength>(SEED_TAB[2]);
    constexpr uint64_t rs3 = srolnValue<WindowLength>(SEED_TAB[3]);

    // clang-format off
    switch (base & 0x3u) {
        case 0: return rs0;
        case 1: return rs1;
        case 2: return rs2;
        default: return rs3;
    }
    // clang-format on
}

template <uint64_t WindowLength>
__device__ __forceinline__ uint64_t baseHash(const uint8_t* encodedBases, uint64_t start) {
    uint64_t h = 0;
    _Pragma("unroll")
    for (uint64_t i = 0; i < WindowLength; ++i) {
        h = srol(h) ^ seedForBase(encodedBases[start + i]);
    }
    return h;
}

template <uint64_t WindowLength>
__device__ __forceinline__ uint64_t rollHash(uint64_t h, uint8_t baseOut, uint8_t baseIn) {
    return srol(h) ^ seedForBase(baseIn) ^ rolledSeed<WindowLength>(baseOut);
}

}  // namespace nthash

}  // namespace bloom::detail