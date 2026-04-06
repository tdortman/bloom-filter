#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

#include "hashutil.cuh"
#include "helpers.cuh"

namespace bloom {

template <
    uint16_t K_,
    uint16_t M_,
    uint16_t S_,
    size_t HashCount_,
    size_t FilterBlockBits_ = 256,
    size_t CudaBlockSize_ = 256,
    typename WordType_ = uint64_t>
struct Config {
    using WordType = WordType_;

    static constexpr uint16_t k = K_;
    static constexpr uint16_t m = M_;
    static constexpr uint16_t s = S_;
    static constexpr size_t hashCount = HashCount_;
    static constexpr size_t filterBlockBits = FilterBlockBits_;
    static constexpr size_t cudaBlockSize = CudaBlockSize_;

    static constexpr size_t wordBits = sizeof(WordType) * 8;
    static constexpr size_t blockWordCount = filterBlockBits / wordBits;
    static constexpr size_t minimizerSpan = k - m + 1;
    static constexpr size_t findereSpan = k - s + 1;
    static constexpr size_t packedPositionWords = (hashCount + 7) / 8;
    static constexpr size_t insertGroupSize = blockWordCount;
    static constexpr size_t queryGroupSize = detail::nextPowerOfTwo(findereSpan);
    static constexpr size_t maxRunKmers = cudaBlockSize;

    static_assert(k > 0, "k must be positive");
    static_assert(m > 0 && m <= k, "m must satisfy 0 < m <= k");
    static_assert(s > 0 && s <= k, "s must satisfy 0 < s <= k");
    static_assert(k <= 32, "This first implementation supports k <= 32");
    static_assert(m <= 32, "This first implementation supports m <= 32");
    static_assert(s <= 32, "This first implementation supports s <= 32");
    static_assert(hashCount > 0, "At least one Bloom hash is required");
    static_assert(hashCount <= 16, "This implementation provides 16 multiplicative salts");
    static_assert(
        std::is_same_v<WordType, uint32_t> || std::is_same_v<WordType, uint64_t>,
        "WordType must be uint32_t or uint64_t"
    );
    static_assert(filterBlockBits >= wordBits, "Filter block must contain at least one word");
    static_assert(detail::powerOfTwo(filterBlockBits), "Filter block size must be a power of two");
    static_assert(filterBlockBits % wordBits == 0, "Filter block size must align to the word size");
    static_assert(blockWordCount <= 32, "At most one warp may cooperate on a filter block");
    static_assert(detail::powerOfTwo(blockWordCount), "blockWordCount must be a power of two");
    static_assert(insertGroupSize <= 32, "insertGroupSize must fit in one warp");
    static_assert(queryGroupSize <= 32, "queryGroupSize must fit in one warp");
    static_assert(detail::powerOfTwo(insertGroupSize), "insertGroupSize must be a power of two");
    static_assert(detail::powerOfTwo(queryGroupSize), "queryGroupSize must be a power of two");
    static_assert(cudaBlockSize % 32 == 0, "CUDA block size must be a multiple of one warp");
    static_assert(
        cudaBlockSize % insertGroupSize == 0,
        "cudaBlockSize must divide insertGroupSize"
    );
    static_assert(cudaBlockSize % queryGroupSize == 0, "cudaBlockSize must divide queryGroupSize");
};

template <typename Config>
class Filter;

namespace detail {

template <typename Config>
__global__ void preprocessSequenceKernel(
    const char* sequence,
    size_t sequenceLength,
    uint64_t* minimizerHashes,
    uint64_t* smerPackedPositions,
    uint8_t* validKmers,
    uint32_t* runLengths,
    size_t* leaderIndices,
    unsigned long long* leaderCount
);

template <typename Config>
__global__ void insertSequenceKernel(
    const uint64_t* minimizerHashes,
    const uint32_t* runLengths,
    const size_t* leaderIndices,
    size_t leaderCount,
    const uint64_t* smerPackedPositions,
    size_t numShards,
    typename Filter<Config>::Shard* shards
);

template <typename Config>
__global__ void containsSequenceKernel(
    const uint64_t* minimizerHashes,
    const uint32_t* runLengths,
    const size_t* leaderIndices,
    size_t leaderCount,
    const uint64_t* smerPackedPositions,
    size_t numShards,
    const typename Filter<Config>::Shard* shards,
    uint8_t* output
);

inline constexpr uint64_t kInvalidHash = std::numeric_limits<uint64_t>::max();
[[nodiscard]] __host__ __device__ __forceinline__ constexpr uint64_t multiplicativeSalt(
    size_t index
) {
    switch (index) {
        case 0:
            return 0x9E37'79B9'7F4A'7C15ULL;
        case 1:
            return 0xC2B2'AE3D'27D4'EB4FULL;
        case 2:
            return 0x1656'67B1'9E37'79F9ULL;
        case 3:
            return 0x85EB'CA77'C2B2'AE63ULL;
        case 4:
            return 0x27D4'EB2F'1656'67C5ULL;
        case 5:
            return 0x94D0'49BB'1331'11EFULL;
        case 6:
            return 0xBF58'476D'1CE4'E5B9ULL;
        case 7:
            return 0xD6E8'FEB8'6659'FD93ULL;
        case 8:
            return 0xA076'1D64'78BD'642FULL;
        case 9:
            return 0xE703'7ED1'A0B4'28DBULL;
        case 10:
            return 0x8EBC'6AF0'9C88'C6E3ULL;
        case 11:
            return 0x5899'65CC'7537'4CC3ULL;
        case 12:
            return 0x1D8E'4E27'C47D'124FULL;
        case 13:
            return 0xEB44'9C93'FBBE'A6B5ULL;
        case 14:
            return 0xDB4F'0B91'75AE'2165ULL;
        default:
            return 0xBBE0'56FD'ADE1'4B91ULL;
    }
}

constexpr int log2Pow2(size_t value) {
    int exponent = 0;
    while (value > 1) {
        value >>= 1;
        ++exponent;
    }
    return exponent;
}

__host__ __device__ __forceinline__ uint8_t encodeBase(char base) {
    switch (base) {
        case 'A':
        case 'a':
            return 0;
        case 'C':
        case 'c':
            return 1;
        case 'G':
        case 'g':
            return 2;
        case 'T':
        case 't':
            return 3;
        default:
            return 0xFF;
    }
}

template <size_t Length>
__device__ __forceinline__ bool encodeWindow(const char* sequence, size_t start, uint64_t& packed) {
    packed = 0;
    _Pragma("unroll")
    for (size_t i = 0; i < Length; ++i) {
        const uint8_t encoded = encodeBase(sequence[start + i]);
        if (encoded == 0xFF) {
            return false;
        }
        packed = (packed << 2) | static_cast<uint64_t>(encoded);
    }
    return true;
}

template <size_t Length, typename EncodedType>
__device__ __forceinline__ bool
encodeWindow(const EncodedType* sequence, size_t start, uint64_t& packed) {
    packed = 0;
    _Pragma("unroll")
    for (size_t i = 0; i < Length; ++i) {
        const uint8_t encoded = static_cast<uint8_t>(sequence[start + i]);
        if (encoded > 3) {
            return false;
        }
        packed = (packed << 2) | static_cast<uint64_t>(encoded);
    }
    return true;
}

template <size_t Length>
[[nodiscard]] __host__ __device__ __forceinline__ constexpr uint64_t packedWindowMask() {
    if constexpr (Length >= 32) {
        return std::numeric_limits<uint64_t>::max();
    } else {
        return (uint64_t{1} << (2 * Length)) - 1;
    }
}

template <size_t Length>
[[nodiscard]] __device__ __forceinline__ uint64_t
rollPackedWindow(uint64_t packed, uint8_t encodedBase) {
    return ((packed << 2) | static_cast<uint64_t>(encodedBase)) & packedWindowMask<Length>();
}

template <typename WordType>
__device__ __forceinline__ void atomicOrWord(WordType* ptr, WordType value) {
    if constexpr (std::is_same_v<WordType, uint64_t>) {
        atomicOr(
            reinterpret_cast<unsigned long long*>(ptr), static_cast<unsigned long long>(value)
        );
    } else {
        atomicOr(reinterpret_cast<unsigned int*>(ptr), static_cast<unsigned int>(value));
    }
}

template <typename WordType>
[[nodiscard]] __host__ __device__ __forceinline__ size_t popcountWord(WordType value) {
#if defined(__CUDA_ARCH__)
    if constexpr (std::is_same_v<WordType, uint64_t>) {
        return static_cast<size_t>(__popcll(static_cast<unsigned long long>(value)));
    } else {
        return static_cast<size_t>(__popc(static_cast<unsigned int>(value)));
    }
#else
    if constexpr (std::is_same_v<WordType, uint64_t>) {
        return static_cast<size_t>(__builtin_popcountll(static_cast<unsigned long long>(value)));
    } else {
        return static_cast<size_t>(__builtin_popcount(static_cast<unsigned int>(value)));
    }
#endif
}

template <typename Config>
[[nodiscard]] __device__ __forceinline__ uint64_t
minimizerHashDirect(const uint8_t* sequence, size_t kmerStart) {
    uint64_t packedMmer = 0;
    if (!encodeWindow<Config::m>(sequence, kmerStart, packedMmer)) {
        return kInvalidHash;
    }

    uint64_t minimumHash = xxhash::xxhash64(packedMmer);
    _Pragma("unroll")
    for (int offset = 1; offset < static_cast<int>(Config::minimizerSpan); ++offset) {
        const uint8_t encoded =
            encodeBase(sequence[kmerStart + static_cast<size_t>(offset) + Config::m - 1]);
        if (encoded > 3) {
            return kInvalidHash;
        }
        packedMmer = rollPackedWindow<Config::m>(packedMmer, encoded);
        const uint64_t candidate = xxhash::xxhash64(packedMmer);
        minimumHash = candidate < minimumHash ? candidate : minimumHash;
    }

    return minimumHash;
}

template <typename Config, typename Tile, typename EncodedType>
[[nodiscard]] __device__ __forceinline__ uint64_t
minimizerHashCooperative(const Tile& tile, const EncodedType* sequence, size_t kmerStart) {
    uint64_t minimizerHash = kInvalidHash;
    if (static_cast<size_t>(tile.thread_rank()) < Config::minimizerSpan) {
        uint64_t packedMmer = 0;
        encodeWindow<Config::m>(
            sequence, kmerStart + static_cast<size_t>(tile.thread_rank()), packedMmer
        );
        minimizerHash = xxhash::xxhash64(packedMmer);
    }

    for (int offset = static_cast<int>(tile.size()) / 2; offset > 0; offset /= 2) {
        const auto other = tile.shfl_down(minimizerHash, offset);
        minimizerHash = other < minimizerHash ? other : minimizerHash;
    }
    return tile.shfl(minimizerHash, 0);
}

template <typename Config>
[[nodiscard]] __device__ __forceinline__ bool
isRunLeader(const uint8_t* sequence, size_t kmerIndex, uint64_t minimizerHash) {
    if (kmerIndex == 0) {
        return true;
    }

    uint64_t packedPrev = 0;
    if (!encodeWindow<Config::k>(sequence, kmerIndex - 1, packedPrev)) {
        return true;
    }

    return minimizerHashDirect<Config>(sequence, kmerIndex - 1) != minimizerHash;
}

template <typename Config>
[[nodiscard]] __device__ __forceinline__ size_t
extendRunLength(const uint8_t* sequence, size_t runStart, size_t numKmers, uint64_t minimizerHash) {
    size_t runLength = 1;
    while (runStart + runLength < numKmers) {
        uint64_t packedKmer = 0;
        if (!encodeWindow<Config::k>(sequence, runStart + runLength, packedKmer)) {
            break;
        }
        if (minimizerHashDirect<Config>(sequence, runStart + runLength) != minimizerHash) {
            break;
        }
        ++runLength;
    }
    return runLength;
}

template <typename Config, typename Tile>
[[nodiscard]] __device__ __forceinline__ uint64_t
minimizerHashBatchReuse(const Tile& tile, uint64_t laneHash, const uint64_t* tailHashes) {
    uint64_t minimizerHash = kInvalidHash;
    const int lane = static_cast<int>(tile.thread_rank());
    const int tileSize = static_cast<int>(tile.size());
    _Pragma("unroll")
    for (int offset = 0; offset < static_cast<int>(Config::minimizerSpan); ++offset) {
        const int sourceLane = lane + offset;
        const auto candidate = sourceLane < tileSize ? tile.shfl(laneHash, sourceLane)
                                                     : tailHashes[sourceLane - tileSize];
        minimizerHash = candidate < minimizerHash ? candidate : minimizerHash;
    }
    return minimizerHash;
}

template <typename Config, typename Tile>
[[nodiscard]] __device__ __forceinline__ uint32_t countRunMatchesCooperative(
    const Tile& tile,
    const char* sequence,
    size_t batchStart,
    size_t batchKmers,
    uint64_t targetMinimizerHash,
    uint64_t* tailHashes
) {
    const auto tileSize = static_cast<size_t>(tile.size());
    const int lane = static_cast<int>(tile.thread_rank());
    const size_t mmerCount = batchKmers + Config::minimizerSpan - 1;
    const size_t inTileMMers = mmerCount < tileSize ? mmerCount : tileSize;

    uint64_t laneMmerHash = kInvalidHash;
    if (static_cast<size_t>(lane) < inTileMMers) {
        uint64_t packedMmer = 0;
        if (encodeWindow<Config::m>(sequence, batchStart + static_cast<size_t>(lane), packedMmer)) {
            laneMmerHash = xxhash::xxhash64(packedMmer);
        }
    }

    if constexpr (Config::minimizerSpan > 1) {
        if (static_cast<size_t>(lane) < Config::minimizerSpan - 1) {
            const size_t tailIndex = tileSize + static_cast<size_t>(lane);
            uint64_t tailHash = kInvalidHash;
            if (tailIndex < mmerCount) {
                uint64_t packedTail = 0;
                if (encodeWindow<Config::m>(sequence, batchStart + tailIndex, packedTail)) {
                    tailHash = xxhash::xxhash64(packedTail);
                }
            }
            tailHashes[lane] = tailHash;
        }
    }
    tile.sync();

    int kmerValid = 0;
    uint64_t minimizerHash = kInvalidHash;
    if (static_cast<size_t>(lane) < batchKmers) {
        uint64_t packedKmer = 0;
        kmerValid =
            encodeWindow<Config::k>(sequence, batchStart + static_cast<size_t>(lane), packedKmer)
                ? 1
                : 0;
        if (kmerValid != 0) {
            minimizerHash = minimizerHashBatchReuse<Config>(tile, laneMmerHash, tailHashes);
        }
    }

    uint32_t matches = 0;
    if (lane == 0) {
        for (size_t kmerOffset = 0; kmerOffset < batchKmers; ++kmerOffset) {
            if (tile.shfl(kmerValid, static_cast<int>(kmerOffset)) == 0) {
                break;
            }
            if (tile.shfl(minimizerHash, static_cast<int>(kmerOffset)) != targetMinimizerHash) {
                break;
            }
            ++matches;
        }
    }
    return tile.shfl(matches, 0);
}

}  // namespace detail

template <typename Config>
class Filter {
   public:
    using WordType = typename Config::WordType;

    struct alignas(32) Shard {
        static constexpr size_t wordCount = Config::blockWordCount;
        static constexpr size_t wordBits = Config::wordBits;
        static constexpr int blockShift = detail::log2Pow2(Config::filterBlockBits);

        WordType words[wordCount];

        [[nodiscard]] __host__ __device__ static size_t
        bitAddress(uint64_t baseHash, size_t hashIndex) {
            const uint64_t mixed = baseHash * detail::multiplicativeSalt(hashIndex);
            return static_cast<size_t>(mixed >> (64 - blockShift));
        }

        [[nodiscard]] __host__ __device__ __forceinline__ static size_t wordIndex(size_t address) {
            return address / wordBits;
        }

        [[nodiscard]] __host__ __device__ __forceinline__ static WordType bitMask(size_t address) {
            return WordType{1} << (address & (wordBits - 1));
        }

        template <typename PackedBuffer>
        [[nodiscard]] __host__ __device__ static WordType
        wordMaskForPackedPositions(const PackedBuffer& packedPositions, size_t ownedWord) {
            WordType mask = 0;
            for (size_t hashIndex = 0; hashIndex < Config::hashCount; ++hashIndex) {
                const size_t packIndex = hashIndex / 8;
                const size_t shift = (hashIndex & 7) * 8;
                const auto address =
                    static_cast<size_t>((packedPositions[packIndex] >> shift) & 0xFFu);
                if (wordIndex(address) == ownedWord) {
                    mask |= bitMask(address);
                }
            }
            return mask;
        }

        template <typename PackedBuffer>
        __host__ __device__ static void decodePackedPositionsToMasks4(
            const PackedBuffer& packedPositions,
            WordType& mask0,
            WordType& mask1,
            WordType& mask2,
            WordType& mask3
        ) {
            static_assert(wordCount == 4, "decodePackedPositionsToMasks4 requires four words");
            mask0 = 0;
            mask1 = 0;
            mask2 = 0;
            mask3 = 0;

            for (size_t hashIndex = 0; hashIndex < Config::hashCount; ++hashIndex) {
                const size_t packIndex = hashIndex / 8;
                const size_t shift = (hashIndex & 7) * 8;
                const auto address =
                    static_cast<size_t>((packedPositions[packIndex] >> shift) & 0xFFu);
                const WordType bit = bitMask(address);
                const size_t idx = wordIndex(address);
                const WordType sel0 = WordType{0} - static_cast<WordType>(idx == 0);
                const WordType sel1 = WordType{0} - static_cast<WordType>(idx == 1);
                const WordType sel2 = WordType{0} - static_cast<WordType>(idx == 2);
                const WordType sel3 = WordType{0} - static_cast<WordType>(idx == 3);
                mask0 |= bit & sel0;
                mask1 |= bit & sel1;
                mask2 |= bit & sel2;
                mask3 |= bit & sel3;
            }
        }

        template <typename WordBuffer>
        [[nodiscard]] __host__ __device__ static bool
        containsHashInWords(const WordBuffer& wordBuffer, uint64_t baseHash) {
            for (size_t hashIndex = 0; hashIndex < Config::hashCount; ++hashIndex) {
                const size_t address = bitAddress(baseHash, hashIndex);
                if ((wordBuffer[wordIndex(address)] & bitMask(address)) == 0) {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] __device__ static bool containsHashInRegisters4(
            WordType word0,
            WordType word1,
            WordType word2,
            WordType word3,
            uint64_t baseHash
        ) {
            static_assert(wordCount == 4, "containsHashInRegisters4 requires four words");
            for (size_t hashIndex = 0; hashIndex < Config::hashCount; ++hashIndex) {
                const size_t address = bitAddress(baseHash, hashIndex);
                const size_t idx = wordIndex(address);
                const WordType word =
                    idx == 0 ? word0 : (idx == 1 ? word1 : (idx == 2 ? word2 : word3));
                if ((word & bitMask(address)) == 0) {
                    return false;
                }
            }
            return true;
        }

        template <typename PackedBuffer>
        [[nodiscard]] __host__ __device__ static bool
        containsPackedPositionsInWords(const WordType* words, const PackedBuffer& packedPositions) {
            for (size_t hashIndex = 0; hashIndex < Config::hashCount; ++hashIndex) {
                const size_t packIndex = hashIndex / 8;
                const size_t shift = (hashIndex & 7) * 8;
                const auto address =
                    static_cast<size_t>((packedPositions[packIndex] >> shift) & 0xFFu);
                if ((words[wordIndex(address)] & bitMask(address)) == 0) {
                    return false;
                }
            }
            return true;
        }

        template <typename PackedBuffer>
        [[nodiscard]] __device__ static bool containsPackedPositionsInRegisters4(
            WordType word0,
            WordType word1,
            WordType word2,
            WordType word3,
            const PackedBuffer& packedPositions
        ) {
            static_assert(
                wordCount == 4, "containsPackedPositionsInRegisters4 requires four words"
            );
            for (size_t hashIndex = 0; hashIndex < Config::hashCount; ++hashIndex) {
                const size_t packIndex = hashIndex / 8;
                const size_t shift = (hashIndex & 7) * 8;
                const auto address =
                    static_cast<size_t>((packedPositions[packIndex] >> shift) & 0xFFu);
                const size_t idx = wordIndex(address);
                const WordType word =
                    idx == 0 ? word0 : (idx == 1 ? word1 : (idx == 2 ? word2 : word3));
                if ((word & bitMask(address)) == 0) {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] __device__ static bool containsMasksInRegisters4(
            WordType word0,
            WordType word1,
            WordType word2,
            WordType word3,
            WordType mask0,
            WordType mask1,
            WordType mask2,
            WordType mask3
        ) {
            return ((word0 & mask0) == mask0) && ((word1 & mask1) == mask1) &&
                   ((word2 & mask2) == mask2) && ((word3 & mask3) == mask3);
        }

        __device__ static void loadWordsVertical(const Shard* shard, WordType* out) {
#if __CUDA_ARCH__ >= 1000
            if constexpr (Config::filterBlockBits == 256) {
                detail::load256BitGlobalNC(shard->words, out);
            } else {
                _Pragma("unroll")
                for (size_t wordIdx = 0; wordIdx < wordCount; ++wordIdx) {
                    out[wordIdx] = shard->words[wordIdx];
                }
            }
#else
            _Pragma("unroll")
            for (size_t wordIdx = 0; wordIdx < wordCount; ++wordIdx) {
                out[wordIdx] = shard->words[wordIdx];
            }
#endif
        }

        [[nodiscard]] __device__ bool containsHash(uint64_t baseHash) const {
            return containsHashInWords(words, baseHash);
        }

        __device__ void atomicOrWordMask(size_t ownedWord, WordType wordMask) {
            detail::atomicOrWord(&words[ownedWord], wordMask);
        }

        __device__ void insertHash(uint64_t baseHash) {
            _Pragma("unroll")
            for (size_t hashIndex = 0; hashIndex < Config::hashCount; ++hashIndex) {
                const size_t address = bitAddress(baseHash, hashIndex);
                detail::atomicOrWord(&words[wordIndex(address)], bitMask(address));
            }
        }
    };

    static constexpr size_t defaultChunkBases = 1 << 20;

    explicit Filter(size_t requestedFilterBits, size_t chunkBases = defaultChunkBases)
        : numShards_(
              detail::nextPowerOfTwo(
                  std::max<size_t>(1, SDIV(requestedFilterBits, Config::filterBlockBits))
              )
          ),
          filterBits_(numShards_ * Config::filterBlockBits),
          chunkBases_(std::max<size_t>(chunkBases, Config::k)) {
        CUDA_CALL(cudaMalloc(&d_shards_, sizeBytes()));
        int currentDevice = 0;
        cudaDeviceProp props{};
        CUDA_CALL(cudaGetDevice(&currentDevice));
        CUDA_CALL(cudaGetDeviceProperties(&props, currentDevice));
        smCount_ = props.multiProcessorCount;
        clear();
    }

    Filter(const Filter&) = delete;
    Filter& operator=(const Filter&) = delete;

    ~Filter() {
        if (d_resultBuffer_ != nullptr) {
            cudaFree(d_resultBuffer_);
        }
        if (d_sequence_ != nullptr) {
            cudaFree(d_sequence_);
        }
        if (d_minimizerHashes_ != nullptr) {
            cudaFree(d_minimizerHashes_);
        }
        if (d_smerPackedPositions_ != nullptr) {
            cudaFree(d_smerPackedPositions_);
        }
        if (d_validKmers_ != nullptr) {
            cudaFree(d_validKmers_);
        }
        if (d_runLengths_ != nullptr) {
            cudaFree(d_runLengths_);
        }
        if (d_leaderIndices_ != nullptr) {
            cudaFree(d_leaderIndices_);
        }
        if (d_leaderCount_ != nullptr) {
            cudaFree(d_leaderCount_);
        }
        if (d_shards_ != nullptr) {
            cudaFree(d_shards_);
        }
    }

    [[nodiscard]] size_t
    insertSequence(const char* sequence, size_t length, cudaStream_t stream = {}) {
        if (sequence == nullptr || length < Config::k) {
            return 0;
        }

        stageSequence(sequence, length, stream);
        ensureMetadataCapacity(length - Config::k + 1);
        launchPreprocess(d_sequence_, length, stream);
        const size_t totalKmers = length - Config::k + 1;
        launchInsert(d_sequence_, length, stream);
        CUDA_CALL(cudaStreamSynchronize(stream));
        return totalKmers;
    }

    [[nodiscard]] size_t insertSequence(std::string_view sequence, cudaStream_t stream = {}) {
        return insertSequence(sequence.data(), sequence.size(), stream);
    }

    void containsSequence(
        const char* sequence,
        size_t length,
        uint8_t* dOutput,
        cudaStream_t stream = {}
    ) const {
        if (dOutput == nullptr || sequence == nullptr || length < Config::k) {
            return;
        }

        stageSequence(sequence, length, stream);
        ensureMetadataCapacity(length - Config::k + 1);
        launchPreprocess(d_sequence_, length, stream);
        launchContains(d_sequence_, length, dOutput, stream);
        CUDA_CALL(cudaStreamSynchronize(stream));
    }

    [[nodiscard]] std::vector<uint8_t>
    containsSequence(const char* sequence, size_t length, cudaStream_t stream = {}) const {
        if (sequence == nullptr || length < Config::k) {
            return {};
        }

        std::vector<uint8_t> output(length - Config::k + 1);

        stageSequence(sequence, length, stream);
        ensureMetadataCapacity(output.size());
        launchPreprocess(d_sequence_, length, stream);
        ensureResultCapacity(output.size());
        launchContains(d_sequence_, length, d_resultBuffer_, stream);
        CUDA_CALL(cudaMemcpyAsync(
            output.data(),
            d_resultBuffer_,
            output.size() * sizeof(uint8_t),
            cudaMemcpyDeviceToHost,
            stream
        ));

        CUDA_CALL(cudaStreamSynchronize(stream));
        return output;
    }

    [[nodiscard]] std::vector<uint8_t>
    containsSequence(std::string_view sequence, cudaStream_t stream = {}) const {
        return containsSequence(sequence.data(), sequence.size(), stream);
    }

    void clear(cudaStream_t stream = {}) {
        CUDA_CALL(cudaMemsetAsync(d_shards_, 0, sizeBytes(), stream));
        CUDA_CALL(cudaStreamSynchronize(stream));
    }

    [[nodiscard]] float loadFactor() const {
        std::vector<Shard> hostShards(shardCount());
        CUDA_CALL(cudaMemcpy(hostShards.data(), d_shards_, sizeBytes(), cudaMemcpyDeviceToHost));

        size_t setBits = 0;
        for (const Shard& shard : hostShards) {
            for (WordType word : shard.words) {
                setBits += detail::popcountWord(word);
            }
        }
        return static_cast<float>(setBits) / static_cast<float>(filterBits_);
    }

    [[nodiscard]] size_t filterBits() const {
        return filterBits_;
    }

    [[nodiscard]] size_t numBlocks() const {
        return numShards_;
    }

    [[nodiscard]] size_t numShards() const {
        return numShards_;
    }

    [[nodiscard]] size_t chunkSizeBases() const {
        return chunkBases_;
    }

   private:
    Shard* d_shards_{};
    mutable char* d_sequence_{};
    mutable uint64_t* d_minimizerHashes_{};
    mutable uint64_t* d_smerPackedPositions_{};
    mutable uint8_t* d_validKmers_{};
    mutable uint32_t* d_runLengths_{};
    mutable size_t* d_leaderIndices_{};
    mutable unsigned long long* d_leaderCount_{};
    mutable uint8_t* d_resultBuffer_{};
    mutable size_t sequenceCapacityBases_{};
    mutable size_t metadataCapacityKmers_{};
    mutable size_t resultCapacityKmers_{};
    mutable size_t leaderCountHost_{};
    int smCount_{};

    size_t numShards_{};
    size_t filterBits_{};
    size_t chunkBases_{};

    [[nodiscard]] size_t shardCount() const {
        return numShards_;
    }

    [[nodiscard]] size_t sizeBytes() const {
        return shardCount() * sizeof(Shard);
    }

    void ensureSequenceCapacity(size_t bases) const {
        if (bases <= sequenceCapacityBases_) {
            return;
        }
        if (d_sequence_ != nullptr) {
            CUDA_CALL(cudaFree(d_sequence_));
        }
        CUDA_CALL(cudaMalloc(&d_sequence_, bases * sizeof(char)));
        sequenceCapacityBases_ = bases;
    }

    void ensureResultCapacity(size_t kmers) const {
        if (kmers <= resultCapacityKmers_) {
            return;
        }
        if (d_resultBuffer_ != nullptr) {
            CUDA_CALL(cudaFree(d_resultBuffer_));
        }
        CUDA_CALL(cudaMalloc(&d_resultBuffer_, kmers * sizeof(uint8_t)));
        resultCapacityKmers_ = kmers;
    }

    void ensureMetadataCapacity(size_t kmers) const {
        if (kmers <= metadataCapacityKmers_) {
            return;
        }
        if (d_minimizerHashes_ != nullptr) {
            CUDA_CALL(cudaFree(d_minimizerHashes_));
        }
        if (d_smerPackedPositions_ != nullptr) {
            CUDA_CALL(cudaFree(d_smerPackedPositions_));
        }
        if (d_validKmers_ != nullptr) {
            CUDA_CALL(cudaFree(d_validKmers_));
        }
        if (d_runLengths_ != nullptr) {
            CUDA_CALL(cudaFree(d_runLengths_));
        }
        if (d_leaderIndices_ != nullptr) {
            CUDA_CALL(cudaFree(d_leaderIndices_));
        }
        CUDA_CALL(cudaMalloc(&d_minimizerHashes_, kmers * sizeof(uint64_t)));
        CUDA_CALL(cudaMalloc(
            &d_smerPackedPositions_,
            (kmers + Config::findereSpan - 1) * Config::packedPositionWords * sizeof(uint64_t)
        ));
        CUDA_CALL(cudaMalloc(&d_validKmers_, kmers * sizeof(uint8_t)));
        CUDA_CALL(cudaMalloc(&d_runLengths_, kmers * sizeof(uint32_t)));
        CUDA_CALL(cudaMalloc(&d_leaderIndices_, kmers * sizeof(size_t)));
        if (d_leaderCount_ == nullptr) {
            CUDA_CALL(cudaMalloc(&d_leaderCount_, sizeof(unsigned long long)));
        }
        metadataCapacityKmers_ = kmers;
    }

    void stageSequence(const char* sequence, size_t length, cudaStream_t stream) const {
        ensureSequenceCapacity(length);
        CUDA_CALL(cudaMemcpyAsync(
            d_sequence_, sequence, length * sizeof(char), cudaMemcpyHostToDevice, stream
        ));
    }

    void launchInsert(const char* dChunk, size_t chunkLength, cudaStream_t stream) {
        (void)dChunk;
        (void)chunkLength;
        if (leaderCountHost_ == 0) {
            return;
        }
        const size_t groupsPerBlock = Config::cudaBlockSize / Config::insertGroupSize;
        const size_t gridSize = SDIV(leaderCountHost_, groupsPerBlock);

        detail::insertSequenceKernel<Config><<<gridSize, Config::cudaBlockSize, 0, stream>>>(
            d_minimizerHashes_,
            d_runLengths_,
            d_leaderIndices_,
            leaderCountHost_,
            d_smerPackedPositions_,
            numShards_,
            d_shards_
        );
        CUDA_CALL(cudaGetLastError());
    }

    void launchContains(
        const char* dChunk,
        size_t chunkLength,
        uint8_t* dOutput,
        cudaStream_t stream
    ) const {
        (void)dChunk;
        const size_t chunkKmers = chunkLength - Config::k + 1;
        if (leaderCountHost_ == 0) {
            CUDA_CALL(cudaMemsetAsync(dOutput, 0, chunkKmers * sizeof(uint8_t), stream));
            return;
        }
        const size_t groupsPerBlock = Config::cudaBlockSize / Config::queryGroupSize;
        const size_t gridSize = SDIV(leaderCountHost_, groupsPerBlock);

        CUDA_CALL(cudaMemsetAsync(dOutput, 0, chunkKmers * sizeof(uint8_t), stream));

        detail::containsSequenceKernel<Config><<<gridSize, Config::cudaBlockSize, 0, stream>>>(
            d_minimizerHashes_,
            d_runLengths_,
            d_leaderIndices_,
            leaderCountHost_,
            d_smerPackedPositions_,
            numShards_,
            d_shards_,
            dOutput
        );
        CUDA_CALL(cudaGetLastError());
    }

    void launchPreprocess(const char* dSequence, size_t sequenceLength, cudaStream_t stream) const {
        const size_t numKmers = sequenceLength - Config::k + 1;
        const size_t gridSize = SDIV(numKmers, Config::cudaBlockSize);

        CUDA_CALL(cudaMemsetAsync(d_leaderCount_, 0, sizeof(unsigned long long), stream));

        detail::preprocessSequenceKernel<Config><<<gridSize, Config::cudaBlockSize, 0, stream>>>(
            dSequence,
            sequenceLength,
            d_minimizerHashes_,
            d_smerPackedPositions_,
            d_validKmers_,
            d_runLengths_,
            d_leaderIndices_,
            d_leaderCount_
        );
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaMemcpyAsync(
            &leaderCountHost_,
            d_leaderCount_,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost,
            stream
        ));
        CUDA_CALL(cudaStreamSynchronize(stream));
    }
};

namespace detail {

template <typename Config>
__global__ void preprocessSequenceKernel(
    const char* sequence,
    size_t sequenceLength,
    uint64_t* minimizerHashes,
    uint64_t* smerPackedPositions,
    uint8_t* validKmers,
    uint32_t* runLengths,
    size_t* leaderIndices,
    unsigned long long* leaderCount
) {
    constexpr size_t sequenceTileBases = Config::cudaBlockSize + Config::k - 1;
    constexpr size_t mmerTileCount = Config::cudaBlockSize + Config::minimizerSpan - 1;
    __shared__ uint8_t sequenceTile[sequenceTileBases];
    __shared__ uint64_t mmerHashes[mmerTileCount];
    __shared__ uint64_t hashTile[Config::cudaBlockSize];
    __shared__ uint8_t validTile[Config::cudaBlockSize];
    __shared__ size_t localLeaderIndices[Config::cudaBlockSize];
    __shared__ unsigned int blockLeaderCount;
    __shared__ unsigned long long blockLeaderBase;

    if (threadIdx.x == 0) {
        blockLeaderCount = 0;
        blockLeaderBase = 0;
    }
    __syncthreads();

    if (sequenceLength < Config::k) {
        return;
    }

    const size_t numKmers = sequenceLength - Config::k + 1;
    const size_t numSmers = sequenceLength - Config::s + 1;
    const size_t blockStartKmer = static_cast<size_t>(blockIdx.x) * Config::cudaBlockSize;
    if (blockStartKmer >= numKmers) {
        return;
    }
    const size_t blockKmers = min(static_cast<size_t>(Config::cudaBlockSize), numKmers - blockStartKmer);
    const size_t tileBases = blockKmers + Config::k - 1;

    for (size_t idx = threadIdx.x; idx < tileBases; idx += Config::cudaBlockSize) {
        sequenceTile[idx] = encodeBase(sequence[blockStartKmer + idx]);
    }
    __syncthreads();

    const size_t blockMmers = blockKmers + Config::minimizerSpan - 1;
    for (size_t idx = threadIdx.x; idx < blockMmers; idx += Config::cudaBlockSize) {
        uint64_t packedMmer = 0;
        if (!encodeWindow<Config::m>(sequenceTile, idx, packedMmer)) {
            mmerHashes[idx] = kInvalidHash;
        } else {
            mmerHashes[idx] = xxhash::xxhash64(packedMmer);
        }
    }
    __syncthreads();

    const size_t blockSmers = min(blockKmers + Config::findereSpan - 1, numSmers - blockStartKmer);
    for (size_t idx = threadIdx.x; idx < blockSmers; idx += Config::cudaBlockSize) {
        uint64_t packedSmer = 0;
        if (!encodeWindow<Config::s>(sequenceTile, idx, packedSmer)) {
            const size_t outputBase = (blockStartKmer + idx) * Config::packedPositionWords;
            for (size_t pack = 0; pack < Config::packedPositionWords; ++pack) {
                smerPackedPositions[outputBase + pack] = kInvalidHash;
            }
        } else {
            const uint64_t baseHash = xxhash::xxhash64(packedSmer);
            uint64_t packedPositions[Config::packedPositionWords] = {};
            for (size_t hashIndex = 0; hashIndex < Config::hashCount; ++hashIndex) {
                const size_t address = Filter<Config>::Shard::bitAddress(baseHash, hashIndex);
                const size_t packIndex = hashIndex / 8;
                const size_t shift = (hashIndex & 7) * 8;
                packedPositions[packIndex] |= static_cast<uint64_t>(address) << shift;
            }
            const size_t outputBase = (blockStartKmer + idx) * Config::packedPositionWords;
            for (size_t pack = 0; pack < Config::packedPositionWords; ++pack) {
                smerPackedPositions[outputBase + pack] = packedPositions[pack];
            }
        }
    }
    __syncthreads();

    const size_t localKmerIndex = static_cast<size_t>(threadIdx.x);
    const bool groupInRange = localKmerIndex < blockKmers;
    const size_t groupIndex = blockStartKmer + localKmerIndex;

    uint64_t minimizerHash = kInvalidHash;
    bool kmerValid = false;
    if (groupInRange) {
        kmerValid = true;
        for (size_t offset = 0; offset < Config::minimizerSpan; ++offset) {
            const uint64_t candidate = mmerHashes[localKmerIndex + offset];
            if (candidate == kInvalidHash) {
                kmerValid = false;
                break;
            }
            minimizerHash = candidate < minimizerHash ? candidate : minimizerHash;
        }
    }

    if (!kmerValid) {
        if (groupInRange) {
            minimizerHashes[groupIndex] = kInvalidHash;
            validKmers[groupIndex] = 0;
            runLengths[groupIndex] = 0;
            hashTile[localKmerIndex] = kInvalidHash;
            validTile[localKmerIndex] = 0;
        }
    } else {
        if (groupInRange) {
            minimizerHashes[groupIndex] = minimizerHash;
            validKmers[groupIndex] = 1;
            hashTile[localKmerIndex] = minimizerHash;
            validTile[localKmerIndex] = 1;
        }
    }
    __syncthreads();

    if (!groupInRange) {
        return;
    }

    if (kmerValid) {
        const bool isLocalLeader = localKmerIndex == 0 || validTile[localKmerIndex - 1] == 0 ||
                                   hashTile[localKmerIndex - 1] != minimizerHash;
        if (!isLocalLeader) {
            runLengths[groupIndex] = 0;
        } else {
            uint32_t runLength = 1;
            while (localKmerIndex + runLength < blockKmers &&
                   validTile[localKmerIndex + runLength] != 0 &&
                   hashTile[localKmerIndex + runLength] == minimizerHash) {
                ++runLength;
            }
            runLengths[groupIndex] = runLength;

            const unsigned int localSlot = atomicAdd(&blockLeaderCount, 1u);
            localLeaderIndices[localSlot] = groupIndex;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockLeaderCount != 0) {
        blockLeaderBase = atomicAdd(leaderCount, static_cast<unsigned long long>(blockLeaderCount));
    }
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < blockLeaderCount; idx += Config::cudaBlockSize) {
        leaderIndices[blockLeaderBase + idx] = localLeaderIndices[idx];
    }
}

template <typename Config>
__global__ void insertSequenceKernel(
    const uint64_t* minimizerHashes,
    const uint32_t* runLengths,
    const size_t* leaderIndices,
    size_t leaderCount,
    const uint64_t* smerPackedPositions,
    size_t numShards,
    typename Filter<Config>::Shard* shards
) {
    namespace cg = cooperative_groups;

    constexpr size_t tileSize = Config::insertGroupSize;
    constexpr size_t groupsPerBlock = Config::cudaBlockSize / tileSize;
    constexpr size_t chunkHashCount = 8;
    __shared__ typename Config::WordType runMasks[groupsPerBlock][chunkHashCount * Config::blockWordCount];

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<tileSize>(block);

    const int lane = static_cast<int>(tile.thread_rank());
    const size_t groupSlot = threadIdx.x / tileSize;
    const size_t leaderSlot = static_cast<size_t>(blockIdx.x) * groupsPerBlock + groupSlot;
    if (leaderSlot >= leaderCount) {
        return;
    }

    const size_t leaderKmerIndex = leaderIndices[leaderSlot];
    const size_t runLength = runLengths[leaderKmerIndex];
    const uint64_t minimizerHash = minimizerHashes[leaderKmerIndex];

    auto& shard = shards[static_cast<size_t>(minimizerHash) & (numShards - 1)];

    typename Config::WordType wordMask = 0;
    const size_t smerCount = static_cast<size_t>(runLength) + Config::findereSpan - 1;

    if constexpr (Config::blockWordCount == 4 && std::is_same_v<typename Config::WordType, uint64_t>) {
        for (size_t smerBase = 0; smerBase < smerCount; smerBase += chunkHashCount) {
            const size_t activeHashes = min(chunkHashCount, smerCount - smerBase);

            for (auto idx = static_cast<size_t>(lane); idx < activeHashes; idx += tile.size()) {
                const size_t sourceBase =
                    (leaderKmerIndex + smerBase + idx) * Config::packedPositionWords;
                typename Config::WordType mask0 = 0;
                typename Config::WordType mask1 = 0;
                typename Config::WordType mask2 = 0;
                typename Config::WordType mask3 = 0;
                Filter<Config>::Shard::decodePackedPositionsToMasks4(
                    &smerPackedPositions[sourceBase], mask0, mask1, mask2, mask3
                );
                const size_t targetBase = idx * Config::blockWordCount;
                runMasks[groupSlot][targetBase + 0] = mask0;
                runMasks[groupSlot][targetBase + 1] = mask1;
                runMasks[groupSlot][targetBase + 2] = mask2;
                runMasks[groupSlot][targetBase + 3] = mask3;
            }
            tile.sync();

            for (size_t smerOffset = 0; smerOffset < activeHashes; ++smerOffset) {
                wordMask |= runMasks[groupSlot][smerOffset * Config::blockWordCount + static_cast<size_t>(lane)];
            }
            tile.sync();
        }
    } else {
        for (size_t smerOffset = 0; smerOffset < smerCount; ++smerOffset) {
            const size_t sourceBase = (leaderKmerIndex + smerOffset) * Config::packedPositionWords;
            wordMask |= Filter<Config>::Shard::wordMaskForPackedPositions(
                &smerPackedPositions[sourceBase], static_cast<size_t>(lane)
            );
        }
    }

    if (wordMask != 0) {
        shard.atomicOrWordMask(static_cast<size_t>(lane), wordMask);
    }
}

template <typename Config>
__global__ void containsSequenceKernel(
    const uint64_t* minimizerHashes,
    const uint32_t* runLengths,
    const size_t* leaderIndices,
    size_t leaderCount,
    const uint64_t* smerPackedPositions,
    size_t numShards,
    const typename Filter<Config>::Shard* shards,
    uint8_t* output
) {
    namespace cg = cooperative_groups;

    constexpr size_t tileSize = Config::queryGroupSize;
    constexpr size_t groupsPerBlock = Config::cudaBlockSize / tileSize;
    constexpr size_t chunkHashCount = tileSize + Config::findereSpan - 1;
    __shared__ typename Config::WordType runMasks[groupsPerBlock][chunkHashCount * Config::blockWordCount];

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<tileSize>(block);

    const int lane = static_cast<int>(tile.thread_rank());
    const size_t groupSlot = threadIdx.x / tileSize;
    const size_t leaderSlot = static_cast<size_t>(blockIdx.x) * groupsPerBlock + groupSlot;
    if (leaderSlot >= leaderCount) {
        return;
    }

    const size_t leaderKmerIndex = leaderIndices[leaderSlot];
    const size_t runLength = runLengths[leaderKmerIndex];
    const uint64_t minimizerHash = minimizerHashes[leaderKmerIndex];
    if constexpr (Config::blockWordCount == 4 &&
                  std::is_same_v<typename Config::WordType, uint64_t>) {
        typename Config::WordType word0 = 0;
        typename Config::WordType word1 = 0;
        typename Config::WordType word2 = 0;
        typename Config::WordType word3 = 0;

        if (lane == 0) {
#if __CUDA_ARCH__ >= 1000
            detail::load256BitGlobalNC(
                shards[minimizerHash & (numShards - 1)].words, word0, word1, word2, word3
            );
#else
            const auto& shard = shards[minimizerHash & (numShards - 1)];
            word0 = shard.words[0];
            word1 = shard.words[1];
            word2 = shard.words[2];
            word3 = shard.words[3];
#endif
        }
        word0 = tile.shfl(word0, 0);
        word1 = tile.shfl(word1, 0);
        word2 = tile.shfl(word2, 0);
        word3 = tile.shfl(word3, 0);

        for (size_t kmerBase = 0; kmerBase < runLength; kmerBase += tile.size()) {
            const size_t activeKmers = min(static_cast<size_t>(tile.size()), runLength - kmerBase);
            const size_t activeHashes = activeKmers + Config::findereSpan - 1;

            for (size_t idx = static_cast<size_t>(lane); idx < activeHashes; idx += tile.size()) {
                const size_t sourceBase =
                    (leaderKmerIndex + kmerBase + idx) * Config::packedPositionWords;
                typename Config::WordType mask0 = 0;
                typename Config::WordType mask1 = 0;
                typename Config::WordType mask2 = 0;
                typename Config::WordType mask3 = 0;
                Filter<Config>::Shard::decodePackedPositionsToMasks4(
                    &smerPackedPositions[sourceBase], mask0, mask1, mask2, mask3
                );
                const size_t targetBase = idx * Config::blockWordCount;
                runMasks[groupSlot][targetBase + 0] = mask0;
                runMasks[groupSlot][targetBase + 1] = mask1;
                runMasks[groupSlot][targetBase + 2] = mask2;
                runMasks[groupSlot][targetBase + 3] = mask3;
            }
            tile.sync();

            const size_t kmerOffset = kmerBase + static_cast<size_t>(lane);
            if (kmerOffset < runLength) {
                uint32_t hitMask = 0;
                for (size_t smerOffset = 0; smerOffset < Config::findereSpan; ++smerOffset) {
                    const size_t maskBase =
                        (static_cast<size_t>(lane) + smerOffset) * Config::blockWordCount;
                    const bool present = Filter<Config>::Shard::containsMasksInRegisters4(
                        word0,
                        word1,
                        word2,
                        word3,
                        runMasks[groupSlot][maskBase + 0],
                        runMasks[groupSlot][maskBase + 1],
                        runMasks[groupSlot][maskBase + 2],
                        runMasks[groupSlot][maskBase + 3]
                    );
                    hitMask = (hitMask << 1) | static_cast<uint32_t>(present ? 1 : 0);
                }
                output[leaderKmerIndex + kmerOffset] =
                    hitMask == ((uint32_t{1} << Config::findereSpan) - 1) ? 1 : 0;
            }
            tile.sync();
        }
    } else if (lane == 0) {
        typename Config::WordType localWords[Config::blockWordCount];
        Filter<Config>::Shard::loadWordsVertical(
            &shards[minimizerHash & (numShards - 1)], localWords
        );

        for (size_t kmerBase = 0; kmerBase < runLength; kmerBase += tile.size()) {
            const size_t activeKmers = min(static_cast<size_t>(tile.size()), runLength - kmerBase);
            const size_t activeHashes = activeKmers + Config::findereSpan - 1;
            uint8_t ring[Config::findereSpan] = {};
            for (size_t smerOffset = 0; smerOffset < activeHashes; ++smerOffset) {
                const size_t sourceBase =
                    (leaderKmerIndex + kmerBase + smerOffset) * Config::packedPositionWords;
                ring[smerOffset % Config::findereSpan] = Filter<Config>::Shard::containsPackedPositionsInWords(
                    localWords, &smerPackedPositions[sourceBase]
                );

                if (smerOffset + 1 < Config::findereSpan) {
                    continue;
                }

                bool present = true;
                for (size_t ringOffset = 0; ringOffset < Config::findereSpan; ++ringOffset) {
                    if (ring[ringOffset] == 0) {
                        present = false;
                        break;
                    }
                }

                output[leaderKmerIndex + kmerBase + (smerOffset + 1 - Config::findereSpan)] =
                    present;
            }
        }
    }
}

}  // namespace detail

}  // namespace bloom
