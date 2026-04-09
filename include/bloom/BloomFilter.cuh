#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "Fastx.hpp"
#include "hashutil.cuh"
#include "helpers.cuh"

namespace bloom {

template <
    uint16_t K_,
    uint16_t S_,
    uint16_t M_,
    uint64_t HashCount_,
    uint64_t CudaBlockSize_ = 256,
    typename WordType_ = uint64_t>
struct Config {
    using WordType = WordType_;

    static constexpr uint16_t k = K_;
    static constexpr uint16_t m = M_;
    static constexpr uint16_t s = S_;
    static constexpr uint64_t hashCount = HashCount_;
    static constexpr uint64_t filterBlockBits = 256;
    static constexpr uint64_t cudaBlockSize = CudaBlockSize_;

    static constexpr uint64_t wordBits = sizeof(WordType) * 8;
    static constexpr uint64_t blockWordCount = filterBlockBits / wordBits;
    static constexpr uint64_t minimizerSpan = k - m + 1;
    static constexpr uint64_t findereSpan = k - s + 1;
    static constexpr uint64_t insertGroupSize = blockWordCount;
    static constexpr uint64_t queryGroupSize = filterBlockBits <= 256 ? 1 : (filterBlockBits / 256);
    static constexpr uint64_t maxRunKmers = cudaBlockSize;

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

template <typename Tile, typename WordType>
[[nodiscard]] __device__ __forceinline__ WordType tileOrReduce4(const Tile& tile, WordType value) {
    value |= tile.shfl_xor(value, 1);
    value |= tile.shfl_xor(value, 2);
    return value;
}

template <typename Config>
struct SequenceKmerInput;

template <typename Config>
struct PackedKmerInput;

template <typename Config, typename Input>
__global__ void containsKmersKernel(
    Input input,
    uint64_t numShards,
    const typename Filter<Config>::Shard* shards,
    uint8_t* output
);

template <typename Config>
__device__ __forceinline__ bool prepareSequenceHashTiles(
    const char* sequence,
    uint64_t blockStartKmer,
    uint64_t blockKmers,
    uint64_t numSmers,
    uint8_t* sequenceTile,
    uint64_t* mmerHashes,
    uint64_t* smerHashes
);

template <typename Config, typename Input>
__global__ void
insertKmersKernel(Input input, uint64_t numShards, typename Filter<Config>::Shard* shards);

inline constexpr uint64_t kInvalidHash = std::numeric_limits<uint64_t>::max();
template <uint64_t Index>
struct SaltLiteral;

template <>
struct SaltLiteral<0> {
    static constexpr uint64_t value = 0x9E37'79B9'7F4A'7C15ULL;
};
template <>
struct SaltLiteral<1> {
    static constexpr uint64_t value = 0xC2B2'AE3D'27D4'EB4FULL;
};
template <>
struct SaltLiteral<2> {
    static constexpr uint64_t value = 0x1656'67B1'9E37'79F9ULL;
};
template <>
struct SaltLiteral<3> {
    static constexpr uint64_t value = 0x85EB'CA77'C2B2'AE63ULL;
};
template <>
struct SaltLiteral<4> {
    static constexpr uint64_t value = 0x27D4'EB2F'1656'67C5ULL;
};
template <>
struct SaltLiteral<5> {
    static constexpr uint64_t value = 0x94D0'49BB'1331'11EFULL;
};
template <>
struct SaltLiteral<6> {
    static constexpr uint64_t value = 0xBF58'476D'1CE4'E5B9ULL;
};
template <>
struct SaltLiteral<7> {
    static constexpr uint64_t value = 0xD6E8'FEB8'6659'FD93ULL;
};
template <>
struct SaltLiteral<8> {
    static constexpr uint64_t value = 0xA076'1D64'78BD'642FULL;
};
template <>
struct SaltLiteral<9> {
    static constexpr uint64_t value = 0xE703'7ED1'A0B4'28DBULL;
};
template <>
struct SaltLiteral<10> {
    static constexpr uint64_t value = 0x8EBC'6AF0'9C88'C6E3ULL;
};
template <>
struct SaltLiteral<11> {
    static constexpr uint64_t value = 0x5899'65CC'7537'4CC3ULL;
};
template <>
struct SaltLiteral<12> {
    static constexpr uint64_t value = 0x1D8E'4E27'C47D'124FULL;
};
template <>
struct SaltLiteral<13> {
    static constexpr uint64_t value = 0xEB44'9C93'FBBE'A6B5ULL;
};
template <>
struct SaltLiteral<14> {
    static constexpr uint64_t value = 0xDB4F'0B91'75AE'2165ULL;
};
template <>
struct SaltLiteral<15> {
    static constexpr uint64_t value = 0xBBE0'56FD'ADE1'4B91ULL;
};

template <uint64_t Index>
[[nodiscard]] __host__ __device__ __forceinline__ constexpr uint64_t multiplicativeSaltLiteral() {
    static_assert(Index < 16, "Salt index out of range");
    return SaltLiteral<Index>::value;
}

template <typename Config, typename Fn, uint64_t... HashIndices>
__host__ __device__ __forceinline__ void
forEachHashIndexImpl(Fn&& fn, std::index_sequence<HashIndices...>) {
    (fn(std::integral_constant<uint64_t, HashIndices>{}), ...);
}

template <typename Config, typename Fn>
__host__ __device__ __forceinline__ void forEachHashIndex(Fn&& fn) {
    forEachHashIndexImpl<Config>(
        static_cast<Fn&&>(fn), std::make_index_sequence<Config::hashCount>{}
    );
}

constexpr int log2Pow2(uint64_t value) {
    int exponent = 0;
    while (value > 1) {
        value >>= 1;
        ++exponent;
    }
    return exponent;
}

__host__ __device__ __forceinline__ uint8_t encodeBase(uint8_t base) {
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

template <uint64_t Length>
__device__ __forceinline__ bool
encodeWindow(const uint8_t* encodedSequence, uint64_t start, uint64_t& packed) {
    packed = 0;
    uint8_t invalid = 0;
    _Pragma("unroll")
    for (uint64_t i = 0; i < Length; ++i) {
        const uint8_t encoded = encodedSequence[start + i];
        invalid |= static_cast<uint8_t>(encoded >> 2);
        packed = (packed << 2) | static_cast<uint64_t>(encoded & 0x3u);
    }
    return invalid == 0;
}

template <uint64_t Length, typename EncodedType>
[[nodiscard]] __device__ __forceinline__ uint64_t
packEncodedWindowUnchecked(const EncodedType* sequence, uint64_t start) {
    uint64_t packed = 0;
    // _Pragma("unroll")
    for (uint64_t i = 0; i < Length; ++i) {
        packed = (packed << 2) |
                 (static_cast<uint64_t>(static_cast<uint8_t>(sequence[start + i]) & 0x3u));
    }
    return packed;
}

template <uint64_t Length>
[[nodiscard]] __host__ __device__ __forceinline__ constexpr uint64_t packedWindowMask() {
    if constexpr (Length >= 32) {
        return std::numeric_limits<uint64_t>::max();
    } else {
        return (uint64_t{1} << (2 * Length)) - 1;
    }
}

template <uint64_t WindowLength, uint64_t K>
[[nodiscard]] __host__ __device__ __forceinline__ constexpr uint64_t
extractPackedSubwindow(uint64_t packedKmer, uint64_t start) {
    static_assert(WindowLength <= K, "WindowLength must not exceed K");
    return (packedKmer >> (2 * (K - (start + WindowLength)))) & packedWindowMask<WindowLength>();
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
[[nodiscard]] __host__ __device__ __forceinline__ uint64_t popcountWord(WordType value) {
#if defined(__CUDA_ARCH__)
    if constexpr (std::is_same_v<WordType, uint64_t>) {
        return static_cast<uint64_t>(__popcll(static_cast<unsigned long long>(value)));
    } else {
        return static_cast<uint64_t>(__popc(static_cast<unsigned int>(value)));
    }
#else
    if constexpr (std::is_same_v<WordType, uint64_t>) {
        return static_cast<uint64_t>(__builtin_popcountll(static_cast<unsigned long long>(value)));
    } else {
        return static_cast<uint64_t>(__builtin_popcount(static_cast<unsigned int>(value)));
    }
#endif
}

}  // namespace detail

template <typename Config>
class Filter {
   public:
    using WordType = typename Config::WordType;

    struct alignas(32) Shard {
        static constexpr uint64_t wordCount = Config::blockWordCount;
        static constexpr uint64_t wordBits = Config::wordBits;
        static constexpr int blockShift = detail::log2Pow2(Config::filterBlockBits);

        WordType words[wordCount];

        template <uint64_t HashIndex>
        [[nodiscard]] __host__ __device__ static uint64_t bitAddress(uint64_t baseHash) {
            static_assert(HashIndex < Config::hashCount, "Hash index out of range");
            const uint64_t mixed = baseHash * detail::multiplicativeSaltLiteral<HashIndex>();
            return static_cast<uint64_t>(mixed >> (64 - blockShift));
        }

        [[nodiscard]] __host__ __device__ __forceinline__ static uint64_t wordIndex(
            uint64_t address
        ) {
            return address / wordBits;
        }

        [[nodiscard]] __host__ __device__ __forceinline__ static WordType bitMask(
            uint64_t address
        ) {
            return WordType{1} << (address & (wordBits - 1));
        }

        [[nodiscard]] __device__ static bool containsHashInRegisters4(
            WordType word0,
            WordType word1,
            WordType word2,
            WordType word3,
            uint64_t baseHash
        ) {
            static_assert(wordCount == 4, "containsHashInRegisters4 requires four words");
            bool present = true;
            detail::forEachHashIndex<Config>(
                [&]<uint64_t HashIndex>(std::integral_constant<uint64_t, HashIndex>) {
                    if (!present) {
                        return;
                    }
                    const uint64_t address = bitAddress<HashIndex>(baseHash);
                    const uint64_t idx = wordIndex(address);
                    const WordType word = (idx == 0) * word0 + (idx == 1) * word1 +
                                          (idx == 2) * word2 + (idx == 3) * word3;
                    present = ((word & bitMask(address)) != 0);
                }
            );
            return present;
        }

        __device__ static void hashToWordMasks4(
            uint64_t baseHash,
            WordType activeMask,
            WordType& mask0,
            WordType& mask1,
            WordType& mask2,
            WordType& mask3
        ) {
            static_assert(wordCount == 4, "hashToWordMasks4 requires four words");
            detail::forEachHashIndex<Config>(
                [&]<uint64_t HashIndex>(std::integral_constant<uint64_t, HashIndex>) {
                    const uint64_t address = bitAddress<HashIndex>(baseHash);
                    const WordType bit = bitMask(address) & activeMask;
                    const uint64_t idx = wordIndex(address);

                    mask0 |= ((idx == 0) * bit);
                    mask1 |= ((idx == 1) * bit);
                    mask2 |= ((idx == 2) * bit);
                    mask3 |= ((idx == 3) * bit);
                }
            );
        }

        __device__ static void hashToWordMasks4(
            uint64_t baseHash,
            WordType& mask0,
            WordType& mask1,
            WordType& mask2,
            WordType& mask3
        ) {
            hashToWordMasks4(baseHash, ~WordType{0}, mask0, mask1, mask2, mask3);
        }
    };

    static_assert(
        Config::blockWordCount == 4 && std::is_same_v<WordType, uint64_t>,
        "Filter only supports the fused 256-bit uint64_t shard path"
    );

    explicit Filter(uint64_t requestedFilterBits)
        : numShards_(
              detail::nextPowerOfTwo(
                  std::max<uint64_t>(1, SDIV(requestedFilterBits, Config::filterBlockBits))
              )
          ),
          filterBits_(numShards_ * Config::filterBlockBits) {
        CUDA_CALL(cudaMalloc(&d_shards_, sizeBytes()));
        clear();
    }

    Filter(const Filter&) = delete;
    Filter& operator=(const Filter&) = delete;

    ~Filter() {
        if (d_resultBuffer_ != nullptr) {
            cudaFree(d_resultBuffer_);
        }
        if (d_packedKmers_ != nullptr) {
            cudaFree(d_packedKmers_);
        }
        if (d_sequence_ != nullptr) {
            cudaFree(d_sequence_);
        }
        if (d_shards_ != nullptr) {
            cudaFree(d_shards_);
        }
    }

    [[nodiscard]] uint64_t
    insertSequence(const char* sequence, uint64_t length, cudaStream_t stream = {}) {
        if (sequence == nullptr || length < Config::k) {
            return 0;
        }

        const uint64_t totalKmers = length - Config::k + 1;
        stageSequence(sequence, length, stream);
        launchInsertSequence(d_sequence_, length, stream);
        CUDA_CALL(cudaStreamSynchronize(stream));
        return totalKmers;
    }

    [[nodiscard]] uint64_t insertSequence(std::string_view sequence, cudaStream_t stream = {}) {
        return insertSequence(sequence.data(), sequence.size(), stream);
    }

    [[nodiscard]] uint64_t
    insertPackedKmers(const uint64_t* kmers, uint64_t count, cudaStream_t stream = {}) {
        if (kmers == nullptr || count == 0) {
            return 0;
        }

        stagePackedKmers(kmers, count, stream);
        launchInsertPacked(d_packedKmers_, count, stream);
        CUDA_CALL(cudaStreamSynchronize(stream));
        return count;
    }

    [[nodiscard]] uint64_t
    insertPackedKmers(const std::vector<uint64_t>& kmers, cudaStream_t stream = {}) {
        return insertPackedKmers(kmers.data(), kmers.size(), stream);
    }

    [[nodiscard]] FastxInsertReport insertFastx(std::istream& input, cudaStream_t stream = {}) {
        return insertFastxStream(input, "<stream>", stream);
    }

    [[nodiscard]] FastxInsertReport
    insertFastxFile(std::string_view path, cudaStream_t stream = {}) {
        auto input = detail::openFastxFile(path);
        return insertFastxStream(input, path, stream);
    }

    void containsSequence(
        const char* sequence,
        uint64_t length,
        uint8_t* d_output,
        cudaStream_t stream = {}
    ) const {
        if (d_output == nullptr || sequence == nullptr || length < Config::k) {
            return;
        }

        stageSequence(sequence, length, stream);
        launchContainsSequence(d_sequence_, length, d_output, stream);
        CUDA_CALL(cudaStreamSynchronize(stream));
    }

    [[nodiscard]] std::vector<uint8_t>
    containsSequence(const char* sequence, uint64_t length, cudaStream_t stream = {}) const {
        if (sequence == nullptr || length < Config::k) {
            return {};
        }

        std::vector<uint8_t> output(length - Config::k + 1);

        stageSequence(sequence, length, stream);
        ensureResultCapacity(output.size());
        launchContainsSequence(d_sequence_, length, d_resultBuffer_, stream);
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

    void containsPackedKmers(
        const uint64_t* kmers,
        uint64_t count,
        uint8_t* d_output,
        cudaStream_t stream = {}
    ) const {
        if (d_output == nullptr || kmers == nullptr || count == 0) {
            return;
        }

        stagePackedKmers(kmers, count, stream);
        launchContainsPacked(d_packedKmers_, count, d_output, stream);
        CUDA_CALL(cudaStreamSynchronize(stream));
    }

    [[nodiscard]] std::vector<uint8_t>
    containsPackedKmers(const uint64_t* kmers, uint64_t count, cudaStream_t stream = {}) const {
        if (kmers == nullptr || count == 0) {
            return {};
        }

        std::vector<uint8_t> output(count);
        stagePackedKmers(kmers, count, stream);
        ensureResultCapacity(count);
        launchContainsPacked(d_packedKmers_, count, d_resultBuffer_, stream);
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
    containsPackedKmers(const std::vector<uint64_t>& kmers, cudaStream_t stream = {}) const {
        return containsPackedKmers(kmers.data(), kmers.size(), stream);
    }

    [[nodiscard]] FastxQueryReport queryFastx(std::istream& input, cudaStream_t stream = {}) const {
        return queryFastxStream(input, "<stream>", stream);
    }

    [[nodiscard]] FastxQueryReport
    queryFastxFile(std::string_view path, cudaStream_t stream = {}) const {
        auto input = detail::openFastxFile(path);
        return queryFastxStream(input, path, stream);
    }

    void clear(cudaStream_t stream = {}) {
        CUDA_CALL(cudaMemsetAsync(d_shards_, 0, sizeBytes(), stream));
        CUDA_CALL(cudaStreamSynchronize(stream));
    }

    [[nodiscard]] float loadFactor() const {
        std::vector<Shard> hostShards(shardCount());
        CUDA_CALL(cudaMemcpy(hostShards.data(), d_shards_, sizeBytes(), cudaMemcpyDeviceToHost));

        uint64_t setBits = 0;
        for (const Shard& shard : hostShards) {
            for (WordType word : shard.words) {
                setBits += detail::popcountWord(word);
            }
        }
        return static_cast<float>(setBits) / static_cast<float>(filterBits_);
    }

    [[nodiscard]] uint64_t filterBits() const {
        return filterBits_;
    }

    [[nodiscard]] uint64_t numBlocks() const {
        return numShards_;
    }

    [[nodiscard]] uint64_t numShards() const {
        return numShards_;
    }

   private:
    Shard* d_shards_{};
    mutable char* d_sequence_{};
    mutable uint64_t* d_packedKmers_{};
    mutable uint8_t* d_resultBuffer_{};
    mutable uint64_t sequenceCapacityBases_{};
    mutable uint64_t packedKmerCapacity_{};
    mutable uint64_t resultCapacityKmers_{};

    uint64_t numShards_{};
    uint64_t filterBits_{};

    [[nodiscard]] uint64_t shardCount() const {
        return numShards_;
    }

    [[nodiscard]] uint64_t sizeBytes() const {
        return shardCount() * sizeof(Shard);
    }

    [[nodiscard]] FastxInsertReport
    insertFastxStream(std::istream& input, std::string_view sourceName, cudaStream_t stream) {
        detail::FastxReader reader(input, sourceName);
        detail::FastxRecord record;
        FastxInsertReport report;

        while (reader.nextRecord(record)) {
            ++report.recordsIndexed;
            report.indexedBases += record.sequence.size();
            report.insertedKmers += insertSequence(record.sequence, stream);
        }
        return report;
    }

    [[nodiscard]] FastxQueryReport
    queryFastxStream(std::istream& input, std::string_view sourceName, cudaStream_t stream) const {
        detail::FastxReader reader(input, sourceName);
        detail::FastxRecord record;
        FastxQueryReport report;

        while (reader.nextRecord(record)) {
            ++report.recordsQueried;
            report.queriedBases += record.sequence.size();

            const auto hits = containsSequence(record.sequence, stream);
            report.queriedKmers += hits.size();
            report.positiveKmers += std::count(hits.begin(), hits.end(), uint8_t{1});
        }
        return report;
    }

    void ensureSequenceCapacity(uint64_t bases) const {
        if (bases <= sequenceCapacityBases_) {
            return;
        }
        if (d_sequence_ != nullptr) {
            CUDA_CALL(cudaFree(d_sequence_));
        }
        CUDA_CALL(cudaMalloc(&d_sequence_, bases * sizeof(char)));
        sequenceCapacityBases_ = bases;
    }

    void ensureResultCapacity(uint64_t kmers) const {
        if (kmers <= resultCapacityKmers_) {
            return;
        }
        if (d_resultBuffer_ != nullptr) {
            CUDA_CALL(cudaFree(d_resultBuffer_));
        }
        CUDA_CALL(cudaMalloc(&d_resultBuffer_, kmers * sizeof(uint8_t)));
        resultCapacityKmers_ = kmers;
    }

    void ensurePackedKmerCapacity(uint64_t kmers) const {
        if (kmers <= packedKmerCapacity_) {
            return;
        }
        if (d_packedKmers_ != nullptr) {
            CUDA_CALL(cudaFree(d_packedKmers_));
        }
        CUDA_CALL(cudaMalloc(&d_packedKmers_, kmers * sizeof(uint64_t)));
        packedKmerCapacity_ = kmers;
    }

    void stageSequence(const char* sequence, uint64_t length, cudaStream_t stream) const {
        ensureSequenceCapacity(length);
        CUDA_CALL(cudaMemcpyAsync(
            d_sequence_, sequence, length * sizeof(char), cudaMemcpyHostToDevice, stream
        ));
    }

    void stagePackedKmers(const uint64_t* kmers, uint64_t count, cudaStream_t stream) const {
        ensurePackedKmerCapacity(count);
        CUDA_CALL(cudaMemcpyAsync(
            d_packedKmers_, kmers, count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream
        ));
    }

    void
    launchInsertSequence(const char* d_sequence, uint64_t sequenceLength, cudaStream_t stream) {
        if (sequenceLength < Config::k) {
            return;
        }
        const uint64_t numKmers = sequenceLength - Config::k + 1;
        const uint64_t gridSize = SDIV(numKmers, Config::cudaBlockSize);

        detail::insertKmersKernel<Config><<<gridSize, Config::cudaBlockSize, 0, stream>>>(
            detail::SequenceKmerInput<Config>{d_sequence, sequenceLength}, numShards_, d_shards_
        );
        CUDA_CALL(cudaGetLastError());
    }

    void launchInsertPacked(const uint64_t* d_kmers, uint64_t count, cudaStream_t stream) {
        if (count == 0) {
            return;
        }
        const uint64_t gridSize = SDIV(count, Config::cudaBlockSize);

        detail::insertKmersKernel<Config><<<gridSize, Config::cudaBlockSize, 0, stream>>>(
            detail::PackedKmerInput<Config>{d_kmers, count}, numShards_, d_shards_
        );
        CUDA_CALL(cudaGetLastError());
    }

    void launchContainsSequence(
        const char* d_sequence,
        uint64_t sequenceLength,
        uint8_t* d_output,
        cudaStream_t stream
    ) const {
        const uint64_t numKmers = sequenceLength - Config::k + 1;
        CUDA_CALL(cudaMemsetAsync(d_output, 0, numKmers * sizeof(uint8_t), stream));
        const uint64_t gridSize = SDIV(numKmers, Config::cudaBlockSize);

        detail::containsKmersKernel<Config><<<gridSize, Config::cudaBlockSize, 0, stream>>>(
            detail::SequenceKmerInput<Config>{d_sequence, sequenceLength},
            numShards_,
            d_shards_,
            d_output
        );
        CUDA_CALL(cudaGetLastError());
    }

    void launchContainsPacked(
        const uint64_t* d_kmers,
        uint64_t count,
        uint8_t* d_output,
        cudaStream_t stream
    ) const {
        CUDA_CALL(cudaMemsetAsync(d_output, 0, count * sizeof(uint8_t), stream));
        const uint64_t gridSize = SDIV(count, Config::cudaBlockSize);

        detail::containsKmersKernel<Config><<<gridSize, Config::cudaBlockSize, 0, stream>>>(
            detail::PackedKmerInput<Config>{d_kmers, count}, numShards_, d_shards_, d_output
        );
        CUDA_CALL(cudaGetLastError());
    }
};

namespace detail {

template <typename Config>
struct SequenceKmerInput {
    const char* sequence;
    uint64_t sequenceLength;

    [[nodiscard]] __host__ __device__ uint64_t kmerCount() const {
        return sequenceLength < Config::k ? 0 : (sequenceLength - Config::k + 1);
    }

    [[nodiscard]] __host__ __device__ uint64_t smerCount() const {
        return sequenceLength < Config::s ? 0 : (sequenceLength - Config::s + 1);
    }
};

template <typename Config>
struct PackedKmerInput {
    const uint64_t* kmers;
    uint64_t count;

    [[nodiscard]] __host__ __device__ uint64_t kmerCount() const {
        return count;
    }
};

template <typename Config>
[[nodiscard]] __device__ __forceinline__ uint64_t packedKmerMinimizerHash(uint64_t packedKmer) {
    uint64_t minimizerHash = kInvalidHash;
    _Pragma("unroll")
    for (uint64_t offset = 0; offset < Config::minimizerSpan; ++offset) {
        const uint64_t packedMmer =
            extractPackedSubwindow<Config::m, Config::k>(packedKmer, offset);
        minimizerHash = min(minimizerHash, xxhash::xxhash64(packedMmer));
    }
    return minimizerHash;
}

template <typename Config>
[[nodiscard]] __device__ __forceinline__ uint64_t
packedKmerSmerHash(uint64_t packedKmer, uint64_t start) {
    const uint64_t packedSmer = extractPackedSubwindow<Config::s, Config::k>(packedKmer, start);
    return xxhash::xxhash64(packedSmer);
}

template <typename Config>
__device__ __forceinline__ void loadShardWords4(
    const typename Filter<Config>::Shard* shards,
    uint64_t shardIndex,
    typename Filter<Config>::WordType& word0,
    typename Filter<Config>::WordType& word1,
    typename Filter<Config>::WordType& word2,
    typename Filter<Config>::WordType& word3
) {
#if __CUDA_ARCH__ >= 1000
    detail::load256BitGlobalNC(shards[shardIndex].words, word0, word1, word2, word3);
#else
    const auto& shard = shards[shardIndex];
    word0 = shard.words[0];
    word1 = shard.words[1];
    word2 = shard.words[2];
    word3 = shard.words[3];
#endif
}

template <typename Config>
__device__ __forceinline__ bool prepareSequenceHashTiles(
    const char* sequence,
    uint64_t blockStartKmer,
    uint64_t blockKmers,
    uint64_t numSmers,
    uint8_t* sequenceTile,
    uint64_t* mmerHashes,
    uint64_t* smerHashes
) {
    const uint64_t tileBases = blockKmers + Config::k - 1;

    bool localInvalidBase = false;
    for (uint64_t idx = threadIdx.x; idx < tileBases; idx += Config::cudaBlockSize) {
        const uint8_t encodedBase = encodeBase(sequence[blockStartKmer + idx]);
        sequenceTile[idx] = encodedBase;
        localInvalidBase = localInvalidBase || (encodedBase > 3);
    }
    const bool blockAllValid = __syncthreads_count(localInvalidBase) == 0;

    const uint64_t blockMmers = blockKmers + Config::minimizerSpan - 1;
    for (uint64_t idx = threadIdx.x; idx < blockMmers; idx += Config::cudaBlockSize) {
        if (blockAllValid) {
            const uint64_t packedMmer = packEncodedWindowUnchecked<Config::m>(sequenceTile, idx);
            mmerHashes[idx] = xxhash::xxhash64(packedMmer);
        } else {
            uint64_t packedMmer = 0;
            if (!encodeWindow<Config::m>(sequenceTile, idx, packedMmer)) {
                mmerHashes[idx] = kInvalidHash;
            } else {
                mmerHashes[idx] = xxhash::xxhash64(packedMmer);
            }
        }
    }

    const uint64_t blockSmers =
        min(blockKmers + Config::findereSpan - 1, numSmers - blockStartKmer);
    for (uint64_t idx = threadIdx.x; idx < blockSmers; idx += Config::cudaBlockSize) {
        if (blockAllValid) {
            const uint64_t packedSmer = packEncodedWindowUnchecked<Config::s>(sequenceTile, idx);
            smerHashes[idx] = xxhash::xxhash64(packedSmer);
        } else {
            uint64_t packedSmer = 0;
            if (!encodeWindow<Config::s>(sequenceTile, idx, packedSmer)) {
                smerHashes[idx] = kInvalidHash;
            } else {
                smerHashes[idx] = xxhash::xxhash64(packedSmer);
            }
        }
    }
    __syncthreads();
    return blockAllValid;
}

template <typename Config, typename Input>
__global__ void containsKmersKernel(
    Input input,
    uint64_t numShards,
    const typename Filter<Config>::Shard* shards,
    uint8_t* output
) {
    constexpr bool packedInput = std::is_same_v<Input, PackedKmerInput<Config>>;
    constexpr uint64_t sequenceTileBases = Config::cudaBlockSize + Config::k - 1;
    constexpr uint64_t mmerTileCount = Config::cudaBlockSize + Config::minimizerSpan - 1;
    constexpr uint64_t smerTileCount = Config::cudaBlockSize + Config::findereSpan - 1;

    __shared__ uint8_t sequenceTile[sequenceTileBases];
    __shared__ uint64_t mmerHashes[mmerTileCount];
    __shared__ uint64_t smerHashes[smerTileCount];

    const uint64_t numKmers = input.kmerCount();
    const uint64_t blockStartKmer = static_cast<uint64_t>(blockIdx.x) * Config::cudaBlockSize;
    if (blockStartKmer >= numKmers) {
        return;
    }

    const uint64_t blockKmers =
        min(static_cast<uint64_t>(Config::cudaBlockSize), numKmers - blockStartKmer);
    const auto localKmerIndex = static_cast<uint64_t>(threadIdx.x);
    const bool inRange = localKmerIndex < blockKmers;

    if constexpr (packedInput) {
        if (!inRange) {
            return;
        }

        const uint64_t kmerIndex = blockStartKmer + localKmerIndex;
        uint64_t minimizerHash = kInvalidHash;
        const uint64_t packedKmer = input.kmers[kmerIndex];
        minimizerHash = packedKmerMinimizerHash<Config>(packedKmer);

        typename Config::WordType word0 = 0;
        typename Config::WordType word1 = 0;
        typename Config::WordType word2 = 0;
        typename Config::WordType word3 = 0;
        loadShardWords4<Config>(
            shards, minimizerHash & (numShards - 1), word0, word1, word2, word3
        );

        bool present = true;
        _Pragma("unroll")
        for (uint64_t smerOffset = 0; smerOffset < Config::findereSpan; ++smerOffset) {
            const uint64_t baseHash = packedKmerSmerHash<Config>(packedKmer, smerOffset);
            if (!Filter<Config>::Shard::containsHashInRegisters4(
                    word0, word1, word2, word3, baseHash
                )) {
                present = false;
                break;
            }
        }
        output[kmerIndex] = static_cast<uint8_t>(present);
        return;
    } else {
        prepareSequenceHashTiles<Config>(
            input.sequence,
            blockStartKmer,
            blockKmers,
            input.smerCount(),
            sequenceTile,
            mmerHashes,
            smerHashes
        );

        if (!inRange) {
            return;
        }

        const uint64_t kmerIndex = blockStartKmer + localKmerIndex;
        uint64_t minimizerHash = kInvalidHash;

        bool kmerValid = true;
        _Pragma("unroll")
        for (uint64_t offset = 0; offset < Config::minimizerSpan; ++offset) {
            const uint64_t candidate = mmerHashes[localKmerIndex + offset];
            if (candidate == kInvalidHash) {
                kmerValid = false;
                break;
            }
            minimizerHash = min(candidate, minimizerHash);
        }

        if (!kmerValid) {
            output[kmerIndex] = 0;
            return;
        }

        typename Config::WordType word0 = 0;
        typename Config::WordType word1 = 0;
        typename Config::WordType word2 = 0;
        typename Config::WordType word3 = 0;
        loadShardWords4<Config>(
            shards, minimizerHash & (numShards - 1), word0, word1, word2, word3
        );

        bool present = true;
        _Pragma("unroll")
        for (uint64_t smerOffset = 0; smerOffset < Config::findereSpan; ++smerOffset) {
            const uint64_t baseHash = smerHashes[localKmerIndex + smerOffset];
            if (baseHash == kInvalidHash || !Filter<Config>::Shard::containsHashInRegisters4(
                                                word0, word1, word2, word3, baseHash
                                            )) {
                present = false;
                break;
            }
        }
        output[kmerIndex] = static_cast<uint8_t>(present);
    }
}

template <typename Config, typename Input>
__global__ void
insertKmersKernel(Input input, uint64_t numShards, typename Filter<Config>::Shard* shards) {
    constexpr bool packedInput = std::is_same_v<Input, PackedKmerInput<Config>>;
    constexpr uint64_t sequenceTileBases = Config::cudaBlockSize + Config::k - 1;
    constexpr uint64_t mmerTileCount = Config::cudaBlockSize + Config::minimizerSpan - 1;
    constexpr uint64_t smerTileCount = Config::cudaBlockSize + Config::findereSpan - 1;

    __shared__ uint8_t sequenceTile[sequenceTileBases];
    __shared__ uint64_t mmerHashes[mmerTileCount];
    __shared__ uint64_t smerHashes[smerTileCount];

    const uint64_t numKmers = input.kmerCount();
    const uint64_t blockStartKmer = static_cast<uint64_t>(blockIdx.x) * Config::cudaBlockSize;
    if (blockStartKmer >= numKmers) {
        return;
    }

    const uint64_t blockKmers =
        min(static_cast<uint64_t>(Config::cudaBlockSize), numKmers - blockStartKmer);
    const uint64_t localKmerIndex = static_cast<uint64_t>(threadIdx.x);
    const bool inRange = localKmerIndex < blockKmers;

    typename Config::WordType wordMask0 = 0;
    typename Config::WordType wordMask1 = 0;
    typename Config::WordType wordMask2 = 0;
    typename Config::WordType wordMask3 = 0;

    if constexpr (packedInput) {
        if (!inRange) {
            return;
        }

        const uint64_t kmerIndex = blockStartKmer + localKmerIndex;
        uint64_t minimizerHash = kInvalidHash;
        const uint64_t packedKmer = input.kmers[kmerIndex];
        minimizerHash = packedKmerMinimizerHash<Config>(packedKmer);
        _Pragma("unroll")
        for (uint64_t smerOffset = 0; smerOffset < Config::findereSpan; ++smerOffset) {
            Filter<Config>::Shard::hashToWordMasks4(
                packedKmerSmerHash<Config>(packedKmer, smerOffset),
                wordMask0,
                wordMask1,
                wordMask2,
                wordMask3
            );
        }

        auto& shard = shards[minimizerHash & (numShards - 1)];
        if (wordMask0 != 0) {
            atomicOrWord(&shard.words[0], wordMask0);
        }
        if (wordMask1 != 0) {
            atomicOrWord(&shard.words[1], wordMask1);
        }
        if (wordMask2 != 0) {
            atomicOrWord(&shard.words[2], wordMask2);
        }
        if (wordMask3 != 0) {
            atomicOrWord(&shard.words[3], wordMask3);
        }
    } else {
        prepareSequenceHashTiles<Config>(
            input.sequence,
            blockStartKmer,
            blockKmers,
            input.smerCount(),
            sequenceTile,
            mmerHashes,
            smerHashes
        );

        if (!inRange) {
            return;
        }

        uint64_t minimizerHash = kInvalidHash;

        bool kmerValid = true;
        _Pragma("unroll")
        for (uint64_t offset = 0; offset < Config::minimizerSpan; ++offset) {
            const uint64_t candidate = mmerHashes[localKmerIndex + offset];
            if (candidate == kInvalidHash) {
                kmerValid = false;
                break;
            }
            minimizerHash = min(candidate, minimizerHash);
        }

        if (!kmerValid) {
            return;
        }

        _Pragma("unroll")
        for (uint64_t smerOffset = 0; smerOffset < Config::findereSpan; ++smerOffset) {
            const uint64_t baseHash = smerHashes[localKmerIndex + smerOffset];
            if (baseHash != kInvalidHash) {
                Filter<Config>::Shard::hashToWordMasks4(
                    baseHash, wordMask0, wordMask1, wordMask2, wordMask3
                );
            }
        }

        auto& shard = shards[minimizerHash & (numShards - 1)];
        if (wordMask0 != 0) {
            atomicOrWord(&shard.words[0], wordMask0);
        }
        if (wordMask1 != 0) {
            atomicOrWord(&shard.words[1], wordMask1);
        }
        if (wordMask2 != 0) {
            atomicOrWord(&shard.words[2], wordMask2);
        }
        if (wordMask3 != 0) {
            atomicOrWord(&shard.words[3], wordMask3);
        }
    }
}

}  // namespace detail

}  // namespace bloom
