#pragma once

#include <cuda_runtime.h>

#include <concepts>
#include <cstdint>

namespace bloom {

template <typename T>
concept Alphabet = requires(uint8_t byte) {
    { T::symbolCount } -> std::convertible_to<uint64_t>;
    { T::separator } -> std::convertible_to<uint8_t>;
    { T::encode(byte) } -> std::same_as<uint8_t>;
} && T::symbolCount > 0 && T::symbolCount <= 255;  // 0xFF reserved for invalid bytes

/**
 * @brief An alphabet for encoding DNA sequences, consisting of the symbols A, C, G, and T.
 * Each symbol is encoded as a 2-bit value: A=0, C=1, T=2, G=3. Invalid bytes are encoded as 0xFF.
 */
struct DnaAlphabet {
    static constexpr uint64_t symbolCount = 4;
    static constexpr uint8_t separator = 'N';

    [[nodiscard]] constexpr __host__ __device__ __forceinline__ static uint8_t encode(
        uint8_t byte
    ) {
        const uint8_t upper = byte & 0xDFu;   // force upper for validation only
        const uint8_t x = (byte >> 1u) & 3u;  // A=0, C=1, T=2, G=3
        const uint8_t valid = (upper == 'A') | (upper == 'C') | (upper == 'G') | (upper == 'T');
        const uint8_t mask = -valid;
        return (x & mask) | (0xFFu & ~mask);
    }
};

/**
 * @brief An alphabet for encoding protein sequences, consisting of the 20 standard amino acids
 * plus common ambiguous and rare residue symbols:
 *
 *  A through Z.
 *
 *  Each symbol is encoded as a unique 5-bit value from 0 to 25. Invalid bytes are encoded as
 *  0xFF.
 */
struct ProteinAlphabet {
    static constexpr uint64_t symbolCount = 26;
    static constexpr uint8_t separator = '*';

    [[nodiscard]] constexpr __host__ __device__ __forceinline__ static uint8_t encode(
        uint8_t byte
    ) {
        const uint8_t upper = byte & 0xDFu;
        const uint8_t letterIndex = upper - 'A';
        const uint8_t valid = letterIndex < 26;
        const uint8_t mask = -valid;
        return (letterIndex & mask) | (0xFFu & ~mask);
    }
};

}  // namespace bloom
