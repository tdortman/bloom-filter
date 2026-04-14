#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cuda/std/bit>
#include <cuda/std/span>
#include <cuda/stream_ref>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "device_span.cuh"
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
    static constexpr uint64_t queryGroupSize = 1;
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
    static_assert(
        cuda::std::has_single_bit(filterBlockBits),
        "Filter block size must be a power of two"
    );
    static_assert(filterBlockBits % wordBits == 0, "Filter block size must align to the word size");
    static_assert(blockWordCount <= 32, "At most one warp may cooperate on a filter block");
    static_assert(
        cuda::std::has_single_bit(blockWordCount),
        "blockWordCount must be a power of two"
    );
    static_assert(insertGroupSize <= 32, "insertGroupSize must fit in one warp");
    static_assert(queryGroupSize <= 32, "queryGroupSize must fit in one warp");
    static_assert(
        cuda::std::has_single_bit(insertGroupSize),
        "insertGroupSize must be a power of two"
    );
    static_assert(
        cuda::std::has_single_bit(queryGroupSize),
        "queryGroupSize must be a power of two"
    );
    static_assert(
        hashCount >= blockWordCount,
        "Sectorized layout requires hashCount >= blockWordCount"
    );
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

namespace cg = cooperative_groups;

template <typename Config>
struct SequenceKmerInput;

template <typename Config>
struct PackedKmerInput;

template <typename Config>
__global__ void containsSequenceKmersKernel(
    SequenceKmerInput<Config> input,
    device_span<const typename Filter<Config>::Shard> shards,
    device_span<uint8_t> output
);

template <typename Config>
__global__ void __launch_bounds__(Config::cudaBlockSize, 3) containsPackedKmersKernel(
    PackedKmerInput<Config> input,
    device_span<const typename Filter<Config>::Shard> shards,
    device_span<uint8_t> output
);

template <typename Config>
__device__ __forceinline__ bool prepareSequenceHashTiles(
    const char* sequence,
    uint64_t blockStartKmer,
    uint64_t blockKmers,
    uint8_t* sequenceTile
);

template <typename Config>
__global__ void insertSequenceKmersKernel(
    SequenceKmerInput<Config> input,
    device_span<typename Filter<Config>::Shard> shards
);

template <typename Config>
__global__ void __launch_bounds__(Config::cudaBlockSize, 4) insertPackedKmersKernel(
    PackedKmerInput<Config> input,
    device_span<typename Filter<Config>::Shard> shards
);

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

constexpr __host__ __device__ __forceinline__ uint8_t encodeBase(uint8_t base) {
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

}  // namespace detail

template <typename Config>
class Filter {
   public:
    using WordType = typename Config::WordType;

    struct alignas(32) Shard {
        static constexpr uint64_t wordCount = Config::blockWordCount;
        static constexpr uint64_t wordBits = Config::wordBits;
        static constexpr int wordBitsLog2 = cuda::std::bit_width(wordBits) - 1;

        WordType words[wordCount];

        template <uint64_t HashIndex>
        [[nodiscard]] constexpr __host__ __device__ static uint64_t sectorizedBitAddress(
            uint64_t baseHash
        ) {
            static_assert(HashIndex < Config::hashCount, "Hash index out of range");
            const uint64_t mixed = baseHash * detail::multiplicativeSaltLiteral<HashIndex>();
            return static_cast<uint64_t>(mixed >> (64 - wordBitsLog2));
        }

        [[nodiscard]] __device__ __forceinline__ static bool
        sectorizedContainsHash(const WordType* w, uint64_t baseHash) {
            bool present = true;
            detail::forEachHashIndex<Config>(
                [&]<uint64_t HashIndex>(std::integral_constant<uint64_t, HashIndex>) {
                    if (!present)
                        return;
                    constexpr uint64_t s = HashIndex % Config::blockWordCount;
                    const uint64_t bitPos = sectorizedBitAddress<HashIndex>(baseHash);
                    present = (w[s] & (WordType{1} << bitPos)) != 0;
                }
            );
            return present;
        }

        __device__ __forceinline__ static void sectorizedHashToMasks(
            uint64_t baseHash,
            WordType& mask0,
            WordType& mask1,
            WordType& mask2,
            WordType& mask3
        ) {
            detail::forEachHashIndex<Config>(
                [&]<uint64_t HashIndex>(std::integral_constant<uint64_t, HashIndex>) {
                    constexpr uint64_t s = HashIndex % Config::blockWordCount;
                    const uint64_t bitPos = sectorizedBitAddress<HashIndex>(baseHash);
                    const WordType bit = WordType{1} << bitPos;
                    // clang-format off
                    if      constexpr (s == 0) mask0 |= bit;
                    else if constexpr (s == 1) mask1 |= bit;
                    else if constexpr (s == 2) mask2 |= bit;
                    else                       mask3 |= bit;
                    // clang-format on
                }
            );
        }

        [[nodiscard]] __device__ __forceinline__ static WordType
        sectorizedHashToMask(uint64_t baseHash, uint64_t activeWordIndex) {
            WordType mask = 0;
            detail::forEachHashIndex<Config>(
                [&]<uint64_t HashIndex>(std::integral_constant<uint64_t, HashIndex>) {
                    constexpr uint64_t s = HashIndex % Config::blockWordCount;
                    const uint64_t bitPos = sectorizedBitAddress<HashIndex>(baseHash);
                    mask |= WordType(s == activeWordIndex) << bitPos;
                }
            );
            return mask;
        }
    };

    static_assert(
        Config::blockWordCount == 4 && std::is_same_v<WordType, uint64_t>,
        "Filter only supports the fused 256-bit uint64_t shard path"
    );
    static_assert(
        Config::queryGroupSize == 1,
        "Fused path expects Theta=1 independent query mapping"
    );
    static_assert(
        Config::insertGroupSize == Config::blockWordCount,
        "Fused path expects horizontal insert mapping across shard words"
    );

    explicit Filter(uint64_t requestedFilterBits)
        : numShards_(
              cuda::std::bit_ceil(
                  std::max<uint64_t>(1, detail::divUp(requestedFilterBits, Config::filterBlockBits))
              )
          ),
          filterBits_(numShards_ * Config::filterBlockBits),
          d_shards_(numShards_) {
        clear();
    }

    Filter(const Filter&) = delete;
    Filter& operator=(const Filter&) = delete;
    Filter(Filter&&) = default;
    Filter& operator=(Filter&&) = default;
    ~Filter() = default;

    [[nodiscard]] uint64_t
    insertSequence(std::string_view sequence, cuda::stream_ref stream = cudaStream_t{}) {
        if (sequence.size() < Config::k) {
            return 0;
        }

        const uint64_t totalKmers = sequence.size() - Config::k + 1;
        stageSequence({sequence.data(), sequence.size()}, stream);
        launchInsertSequence(
            device_span<const char>{thrust::raw_pointer_cast(d_sequence_.data()), sequence.size()},
            stream
        );
        CUDA_CALL(cudaStreamSynchronize(stream.get()));
        return totalKmers;
    }

    [[nodiscard]] uint64_t insertSequenceDevice(
        device_span<const char> d_sequence,
        cuda::stream_ref stream = cudaStream_t{}
    ) {
        if (d_sequence.size() < Config::k) {
            return 0;
        }

        const uint64_t totalKmers = d_sequence.size() - Config::k + 1;
        launchInsertSequence(d_sequence, stream);
        return totalKmers;
    }

    [[nodiscard]] uint64_t insertPackedKmers(
        cuda::std::span<const uint64_t> kmers,
        cuda::stream_ref stream = cudaStream_t{}
    ) {
        if (kmers.empty()) {
            return 0;
        }

        stagePackedKmers(kmers, stream);
        launchInsertPacked(
            device_span<const uint64_t>{
                thrust::raw_pointer_cast(d_packedKmers_.data()), kmers.size()
            },
            stream
        );
        CUDA_CALL(cudaStreamSynchronize(stream.get()));
        return kmers.size();
    }

    [[nodiscard]] uint64_t insertPackedKmersDevice(
        device_span<const uint64_t> d_kmers,
        cuda::stream_ref stream = cudaStream_t{}
    ) {
        if (d_kmers.empty()) {
            return 0;
        }

        launchInsertPacked(d_kmers, stream);
        return d_kmers.size();
    }

    [[nodiscard]] FastxInsertReport
    insertFastx(std::istream& input, cuda::stream_ref stream = cudaStream_t{}) {
        return insertFastxStream(input, "<stream>", stream);
    }

    [[nodiscard]] FastxInsertReport
    insertFastxFile(std::string_view path, cuda::stream_ref stream = cudaStream_t{}) {
        auto input = detail::openFastxFile(path);
        return insertFastxStream(input, path, stream);
    }

    void containsSequenceDevice(
        device_span<const char> d_sequence,
        device_span<uint8_t> d_output,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        if (d_sequence.size() < Config::k) {
            return;
        }

        launchContainsSequence(d_sequence, d_output, stream);
    }

    [[nodiscard]] std::vector<uint8_t>
    containsSequence(std::string_view sequence, cuda::stream_ref stream = cudaStream_t{}) const {
        if (sequence.size() < Config::k) {
            return {};
        }

        std::vector<uint8_t> output(sequence.size() - Config::k + 1);

        stageSequence({sequence.data(), sequence.size()}, stream);
        ensureResultCapacity(output.size());
        launchContainsSequence(
            device_span<const char>{thrust::raw_pointer_cast(d_sequence_.data()), sequence.size()},
            device_span<uint8_t>{thrust::raw_pointer_cast(d_resultBuffer_.data()), output.size()},
            stream
        );
        CUDA_CALL(cudaMemcpyAsync(
            output.data(),
            thrust::raw_pointer_cast(d_resultBuffer_.data()),
            output.size() * sizeof(uint8_t),
            cudaMemcpyDeviceToHost,
            stream.get()
        ));

        CUDA_CALL(cudaStreamSynchronize(stream.get()));
        return output;
    }

    void containsPackedKmersDevice(
        device_span<const uint64_t> d_kmers,
        device_span<uint8_t> d_output,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        if (d_kmers.empty()) {
            return;
        }

        launchContainsPacked(d_kmers, d_output, stream);
    }

    [[nodiscard]] std::vector<uint8_t> containsPackedKmers(
        cuda::std::span<const uint64_t> kmers,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        if (kmers.empty()) {
            return {};
        }

        std::vector<uint8_t> output(kmers.size());
        stagePackedKmers(kmers, stream);
        ensureResultCapacity(kmers.size());
        launchContainsPacked(
            device_span<const uint64_t>{
                thrust::raw_pointer_cast(d_packedKmers_.data()), kmers.size()
            },
            device_span<uint8_t>{thrust::raw_pointer_cast(d_resultBuffer_.data()), kmers.size()},
            stream
        );
        CUDA_CALL(cudaMemcpyAsync(
            output.data(),
            thrust::raw_pointer_cast(d_resultBuffer_.data()),
            output.size() * sizeof(uint8_t),
            cudaMemcpyDeviceToHost,
            stream.get()
        ));
        CUDA_CALL(cudaStreamSynchronize(stream.get()));
        return output;
    }

    [[nodiscard]] FastxQueryReport
    queryFastx(std::istream& input, cuda::stream_ref stream = cudaStream_t{}) const {
        return queryFastxStream(input, "<stream>", stream);
    }

    [[nodiscard]] FastxQueryReport
    queryFastxFile(std::string_view path, cuda::stream_ref stream = cudaStream_t{}) const {
        auto input = detail::openFastxFile(path);
        return queryFastxStream(input, path, stream);
    }

    void clear(cuda::stream_ref stream = cudaStream_t{}) {
        CUDA_CALL(cudaMemsetAsync(
            thrust::raw_pointer_cast(d_shards_.data()), 0, sizeBytes(), stream.get()
        ));
        CUDA_CALL(cudaStreamSynchronize(stream.get()));
    }

    [[nodiscard]] float loadFactor() const {
        const auto* wordsBegin =
            reinterpret_cast<const WordType*>(thrust::raw_pointer_cast(d_shards_.data()));
        const uint64_t totalWords = numShards_ * Config::blockWordCount;
        const uint64_t setBits = thrust::transform_reduce(
            thrust::device,
            wordsBegin,
            wordsBegin + totalWords,
            [] __device__(WordType w) -> uint64_t { return cuda::std::popcount(w); },
            uint64_t{0},
            cuda::std::plus<uint64_t>()
        );
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
    uint64_t numShards_{};
    uint64_t filterBits_{};

    thrust::device_vector<Shard> d_shards_;
    mutable thrust::device_vector<char> d_sequence_;
    mutable thrust::device_vector<uint64_t> d_packedKmers_;
    mutable thrust::device_vector<uint8_t> d_resultBuffer_;

    [[nodiscard]] uint64_t shardCount() const {
        return numShards_;
    }

    [[nodiscard]] uint64_t sizeBytes() const {
        return shardCount() * sizeof(Shard);
    }

    [[nodiscard]] FastxInsertReport
    insertFastxStream(std::istream& input, std::string_view sourceName, cuda::stream_ref stream) {
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

    [[nodiscard]] FastxQueryReport queryFastxStream(
        std::istream& input,
        std::string_view sourceName,
        cuda::stream_ref stream
    ) const {
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
        if (bases > d_sequence_.size()) {
            d_sequence_.resize(bases);
        }
    }

    void ensureResultCapacity(uint64_t kmers) const {
        if (kmers > d_resultBuffer_.size()) {
            d_resultBuffer_.resize(kmers);
        }
    }

    void ensurePackedKmerCapacity(uint64_t kmers) const {
        if (kmers > d_packedKmers_.size()) {
            d_packedKmers_.resize(kmers);
        }
    }

    void stageSequence(cuda::std::span<const char> sequence, cuda::stream_ref stream) const {
        ensureSequenceCapacity(sequence.size());
        CUDA_CALL(cudaMemcpyAsync(
            thrust::raw_pointer_cast(d_sequence_.data()),
            sequence.data(),
            sequence.size_bytes(),
            cudaMemcpyHostToDevice,
            stream.get()
        ));
    }

    void stagePackedKmers(cuda::std::span<const uint64_t> kmers, cuda::stream_ref stream) const {
        ensurePackedKmerCapacity(kmers.size());
        CUDA_CALL(cudaMemcpyAsync(
            thrust::raw_pointer_cast(d_packedKmers_.data()),
            kmers.data(),
            kmers.size_bytes(),
            cudaMemcpyHostToDevice,
            stream.get()
        ));
    }

    void launchInsertSequence(device_span<const char> d_sequence, cuda::stream_ref stream) {
        if (d_sequence.size() < Config::k) {
            return;
        }
        const uint64_t numKmers = d_sequence.size() - Config::k + 1;
        const uint64_t gridSize = detail::divUp(numKmers, Config::cudaBlockSize);

        detail::insertSequenceKmersKernel<Config>
            <<<gridSize, Config::cudaBlockSize, 0, stream.get()>>>(
                detail::SequenceKmerInput<Config>{d_sequence},
                device_span<Shard>{thrust::raw_pointer_cast(d_shards_.data()), numShards_}
            );
        CUDA_CALL(cudaGetLastError());
    }

    void launchInsertPacked(device_span<const uint64_t> d_kmers, cuda::stream_ref stream) {
        if (d_kmers.empty()) {
            return;
        }
        const uint64_t gridSize = detail::divUp(d_kmers.size(), Config::cudaBlockSize);

        detail::insertPackedKmersKernel<Config>
            <<<gridSize, Config::cudaBlockSize, 0, stream.get()>>>(
                detail::PackedKmerInput<Config>{d_kmers},
                device_span<Shard>{thrust::raw_pointer_cast(d_shards_.data()), numShards_}
            );
        CUDA_CALL(cudaGetLastError());
    }

    void launchContainsSequence(
        device_span<const char> d_sequence,
        device_span<uint8_t> d_output,
        cuda::stream_ref stream
    ) const {
        const uint64_t numKmers = d_sequence.size() - Config::k + 1;
        CUDA_CALL(cudaMemsetAsync(d_output.data(), 0, d_output.size_bytes(), stream.get()));
        const uint64_t gridSize = detail::divUp(numKmers, Config::cudaBlockSize);

        detail::containsSequenceKmersKernel<Config>
            <<<gridSize, Config::cudaBlockSize, 0, stream.get()>>>(
                detail::SequenceKmerInput<Config>{d_sequence},
                device_span<const Shard>{thrust::raw_pointer_cast(d_shards_.data()), numShards_},
                d_output
            );
        CUDA_CALL(cudaGetLastError());
    }

    void launchContainsPacked(
        device_span<const uint64_t> d_kmers,
        device_span<uint8_t> d_output,
        cuda::stream_ref stream
    ) const {
        const uint64_t gridSize = detail::divUp(d_kmers.size(), Config::cudaBlockSize);

        detail::containsPackedKmersKernel<Config>
            <<<gridSize, Config::cudaBlockSize, 0, stream.get()>>>(
                detail::PackedKmerInput<Config>{d_kmers},
                device_span<const Shard>{thrust::raw_pointer_cast(d_shards_.data()), numShards_},
                d_output
            );
        CUDA_CALL(cudaGetLastError());
    }
};

namespace detail {

template <typename Config>
struct SequenceKmerInput {
    device_span<const char> sequence;

    [[nodiscard]] constexpr __host__ __device__ uint64_t kmerCount() const {
        return sequence.size() < Config::k ? 0 : (sequence.size() - Config::k + 1);
    }

    [[nodiscard]] constexpr __host__ __device__ uint64_t smerCount() const {
        return sequence.size() < Config::s ? 0 : (sequence.size() - Config::s + 1);
    }
};

template <typename Config>
struct PackedKmerInput {
    device_span<const uint64_t> kmers;

    [[nodiscard]] constexpr __host__ __device__ uint64_t kmerCount() const {
        return kmers.size();
    }
};

template <typename Config>
[[nodiscard]] __device__ __forceinline__ uint64_t packedKmerMinimizerHash(uint64_t packedKmer) {
    uint64_t minimizerHash = kInvalidHash;
    _Pragma("unroll 1")
    for (uint64_t offset = 0; offset < Config::minimizerSpan; ++offset) {
        const uint64_t packedMmer =
            extractPackedSubwindow<Config::m, Config::k>(packedKmer, offset);
        minimizerHash = min(minimizerHash, detail::hash64(packedMmer));
    }
    return minimizerHash;
}

template <typename Config>
[[nodiscard]] __device__ __forceinline__ uint64_t
packedKmerSmerHash(uint64_t packedKmer, uint64_t start) {
    const uint64_t packedSmer = extractPackedSubwindow<Config::s, Config::k>(packedKmer, start);
    return detail::hash64(packedSmer);
}

template <typename Config>
[[nodiscard]] __device__ __forceinline__ bool
sectorizedContainsPackedKmer(uint64_t packedKmer, const typename Filter<Config>::WordType* w) {
    _Pragma("unroll")
    for (uint64_t smerOffset = 0; smerOffset < Config::findereSpan; ++smerOffset) {
        const uint64_t smerHash = packedKmerSmerHash<Config>(packedKmer, smerOffset);
        if (!Filter<Config>::Shard::sectorizedContainsHash(w, smerHash)) {
            return false;
        }
    }
    return true;
}

template <typename Config>
__device__ __forceinline__ void loadShardWords4(
    const typename Filter<Config>::Shard* shards,
    uint64_t shardIndex,
    typename Filter<Config>::WordType* w
) {
#if __CUDA_ARCH__ >= 1000
    detail::load256BitGlobalNC(shards[shardIndex].words, w[0], w[1], w[2], w[3]);
#else
    const auto& shard = shards[shardIndex];
    w[0] = shard.words[0];
    w[1] = shard.words[1];
    w[2] = shard.words[2];
    w[3] = shard.words[3];
#endif
}

template <typename Config>
__device__ __forceinline__ bool prepareSequenceHashTiles(
    const char* sequence,
    uint64_t blockStartKmer,
    uint64_t blockKmers,
    uint8_t* sequenceTile
) {
    const uint64_t tileBases = blockKmers + Config::k - 1;

    bool localInvalidBase = false;
    for (uint64_t idx = threadIdx.x; idx < tileBases; idx += Config::cudaBlockSize) {
        const uint8_t encodedBase = encodeBase(sequence[blockStartKmer + idx]);
        sequenceTile[idx] = encodedBase;
        localInvalidBase = localInvalidBase || (encodedBase > 3);
    }
    return __syncthreads_count(localInvalidBase) == 0;
}

template <typename Config>
__global__ void containsSequenceKmersKernel(
    SequenceKmerInput<Config> input,
    device_span<const typename Filter<Config>::Shard> shards,
    device_span<uint8_t> output
) {
    constexpr uint64_t sequenceTileBases = Config::cudaBlockSize + Config::k - 1;

    __shared__ uint8_t sequenceTile[sequenceTileBases];

    const uint64_t numKmers = input.kmerCount();
    const uint64_t blockStartKmer = static_cast<uint64_t>(blockIdx.x) * Config::cudaBlockSize;
    if (blockStartKmer >= numKmers) {
        return;
    }

    const uint64_t blockKmers =
        min(static_cast<uint64_t>(Config::cudaBlockSize), numKmers - blockStartKmer);
    const auto localKmerIndex = static_cast<uint64_t>(threadIdx.x);
    const bool inRange = localKmerIndex < blockKmers;

    const bool blockAllValid = prepareSequenceHashTiles<Config>(
        input.sequence.data(), blockStartKmer, blockKmers, sequenceTile
    );

    if (!inRange) {
        return;
    }

    const uint64_t kmerIndex = blockStartKmer + localKmerIndex;

    if (!blockAllValid) {
        bool kmerValid = true;
        _Pragma("unroll")
        for (uint64_t i = 0; i < Config::k; ++i) {
            if (sequenceTile[localKmerIndex + i] > 3) {
                kmerValid = false;
                break;
            }
        }
        if (!kmerValid) {
            output[kmerIndex] = 0;
            return;
        }
    }

    uint64_t h_m = nthash::baseHash<Config::m>(sequenceTile, localKmerIndex);
    uint64_t minimizerHash = h_m;
    _Pragma("unroll 1")
    for (uint64_t offset = 1; offset < Config::minimizerSpan; ++offset) {
        h_m = nthash::rollHash<Config::m>(
            h_m,
            sequenceTile[localKmerIndex + offset - 1],
            sequenceTile[localKmerIndex + offset - 1 + Config::m]
        );
        minimizerHash = min(minimizerHash, h_m);
    }

    typename Config::WordType w[4] = {};
    loadShardWords4<Config>(shards.data(), minimizerHash & (shards.size() - 1), w);

    uint64_t h_s = nthash::baseHash<Config::s>(sequenceTile, localKmerIndex);
    bool present = true;
    if (!Filter<Config>::Shard::sectorizedContainsHash(w, h_s)) {
        present = false;
    } else {
        _Pragma("unroll 1")
        for (uint64_t smerOffset = 1; smerOffset < Config::findereSpan; ++smerOffset) {
            h_s = nthash::rollHash<Config::s>(
                h_s,
                sequenceTile[localKmerIndex + smerOffset - 1],
                sequenceTile[localKmerIndex + smerOffset - 1 + Config::s]
            );
            if (!Filter<Config>::Shard::sectorizedContainsHash(w, h_s)) {
                present = false;
                break;
            }
        }
    }
    output[kmerIndex] = static_cast<uint8_t>(present);
}

template <typename Config>
__global__ void __launch_bounds__(Config::cudaBlockSize, 3) containsPackedKmersKernel(
    PackedKmerInput<Config> input,
    device_span<const typename Filter<Config>::Shard> shards,
    device_span<uint8_t> output
) {
    const uint64_t kmerIndex = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (kmerIndex >= input.kmers.size()) {
        return;
    }

    const uint64_t packedKmer = input.kmers[kmerIndex];
    const uint64_t minimizerHash = packedKmerMinimizerHash<Config>(packedKmer);

    typename Config::WordType w[4] = {};
    loadShardWords4<Config>(shards.data(), minimizerHash & (shards.size() - 1), w);

    const bool present = sectorizedContainsPackedKmer<Config>(packedKmer, w);
    output[kmerIndex] = static_cast<uint8_t>(present);
}

template <typename Config>
__global__ void insertSequenceKmersKernel(
    SequenceKmerInput<Config> input,
    device_span<typename Filter<Config>::Shard> shards
) {
    constexpr uint64_t sequenceTileBases = Config::cudaBlockSize + Config::k - 1;

    __shared__ uint8_t sequenceTile[sequenceTileBases];

    const uint64_t numKmers = input.kmerCount();
    const uint64_t blockStartKmer = static_cast<uint64_t>(blockIdx.x) * Config::cudaBlockSize;
    if (blockStartKmer >= numKmers) {
        return;
    }

    const uint64_t blockKmers =
        min(static_cast<uint64_t>(Config::cudaBlockSize), numKmers - blockStartKmer);
    const auto localKmerIndex = static_cast<uint64_t>(threadIdx.x);
    const bool inRange = localKmerIndex < blockKmers;

    const bool blockAllValid = prepareSequenceHashTiles<Config>(
        input.sequence.data(), blockStartKmer, blockKmers, sequenceTile
    );

    if (!inRange) {
        return;
    }

    if (!blockAllValid) {
        bool kmerValid = true;
        _Pragma("unroll")
        for (uint64_t i = 0; i < Config::k; ++i) {
            if (sequenceTile[localKmerIndex + i] > 3) {
                kmerValid = false;
                break;
            }
        }
        if (!kmerValid) {
            return;
        }
    }

    uint64_t h_m = nthash::baseHash<Config::m>(sequenceTile, localKmerIndex);
    uint64_t minimizerHash = h_m;
    _Pragma("unroll 1")
    for (uint64_t offset = 1; offset < Config::minimizerSpan; ++offset) {
        h_m = nthash::rollHash<Config::m>(
            h_m,
            sequenceTile[localKmerIndex + offset - 1],
            sequenceTile[localKmerIndex + offset - 1 + Config::m]
        );
        minimizerHash = min(minimizerHash, h_m);
    }

    typename Config::WordType wordMask0 = 0;
    typename Config::WordType wordMask1 = 0;
    typename Config::WordType wordMask2 = 0;
    typename Config::WordType wordMask3 = 0;

    uint64_t h_s = nthash::baseHash<Config::s>(sequenceTile, localKmerIndex);
    Filter<Config>::Shard::sectorizedHashToMasks(h_s, wordMask0, wordMask1, wordMask2, wordMask3);
    _Pragma("unroll 1")
    for (uint64_t smerOffset = 1; smerOffset < Config::findereSpan; ++smerOffset) {
        h_s = nthash::rollHash<Config::s>(
            h_s,
            sequenceTile[localKmerIndex + smerOffset - 1],
            sequenceTile[localKmerIndex + smerOffset - 1 + Config::s]
        );
        Filter<Config>::Shard::sectorizedHashToMasks(
            h_s, wordMask0, wordMask1, wordMask2, wordMask3
        );
    }

    auto& shard = shards[minimizerHash & (shards.size() - 1)];
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

template <typename Config>
__global__ void __launch_bounds__(Config::cudaBlockSize, 4) insertPackedKmersKernel(
    PackedKmerInput<Config> input,
    device_span<typename Filter<Config>::Shard> shards
) {
    using WordType = typename Config::WordType;

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<Config::insertGroupSize>(block);

    const uint64_t kmerIndex = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto inRange = static_cast<uint32_t>(kmerIndex < input.kmers.size());

    uint64_t packedKmer = 0;
    uint64_t minimizerHash = 0;
    if (inRange) {
        packedKmer = input.kmers[kmerIndex];
        minimizerHash = packedKmerMinimizerHash<Config>(packedKmer);
    }

    const uint32_t lane = tile.thread_rank();

    _Pragma("unroll 1")
    for (uint32_t src = 0; src < Config::insertGroupSize; ++src) {
        const uint32_t srcActive = tile.shfl(inRange, src);
        const uint64_t srcMinHash = tile.shfl(minimizerHash, src);
        const uint64_t srcPackedKmer = tile.shfl(packedKmer, src);

        WordType laneMask = 0;
        _Pragma("unroll 1")
        for (uint64_t s = 0; s < Config::findereSpan; ++s) {
            const uint64_t smerHash = packedKmerSmerHash<Config>(srcPackedKmer, s);
            laneMask |= Filter<Config>::Shard::sectorizedHashToMask(smerHash, lane);
        }

        if (srcActive && laneMask != 0) {
            atomicOrWord(&shards[srcMinHash & (shards.size() - 1)].words[lane], laneMask);
        }
    }
}

}  // namespace detail

}  // namespace bloom
