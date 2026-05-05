#pragma once

#include <cuda/__cmath/ceil_div.h>
#include <cuda_runtime.h>

#include <cuda/algorithm>
#include <cuda/std/bit>
#include <cuda/std/span>
#include <cuda/stream>

#include <cub/warp/warp_reduce.cuh>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "Alphabet.cuh"
#include "device_span.cuh"
#include "Fastx.hpp"
#include "hashutil.cuh"
#include "helpers.cuh"

namespace bloom {

/**
 * @brief Compile-time configuration for a bloom::Filter.
 *
 * All filter behaviour (k-mer length, minimizer width, s-mer width, hash
 * count, CUDA block size, and alphabet) is encoded in this struct so that
 * separate configurations produce completely independent Filter types with
 * zero run-time overhead.
 *
 * @tparam K_             k-mer length (1-32).
 * @tparam S_             s-mer width used as the Bloom hash seed (1-k).
 * @tparam M_             minimizer width used for shard selection (1-k).
 * @tparam HashCount_     number of independent Bloom hash functions (1-16).
 * @tparam CudaBlockSize_ CUDA threads per block (multiple of 32, default 256).
 * @tparam Alphabet_      byte-to-symbol alphabet encoding.
 */
template <
    uint16_t K_,
    uint16_t S_,
    uint16_t M_,
    uint64_t HashCount_,
    uint64_t CudaBlockSize_ = 256,
    Alphabet Alphabet_ = DnaAlphabet>
struct Config {
    using Alphabet = Alphabet_;

    static constexpr uint16_t k = K_;
    static constexpr uint16_t m = M_;
    static constexpr uint16_t s = S_;
    static constexpr uint64_t hashCount = HashCount_;
    static constexpr uint64_t alphabetSize = Alphabet::symbolCount;
    static constexpr uint64_t symbolBits = cuda::std::bit_width(alphabetSize - 1);
    static constexpr uint64_t symbolMask = (uint64_t{1} << symbolBits) - 1;
    static constexpr uint64_t filterBlockBits = 256;
    static constexpr uint64_t cudaBlockSize = CudaBlockSize_;

    static constexpr uint64_t wordBits = 64;
    static constexpr uint64_t blockWordCount = filterBlockBits / wordBits;
    static constexpr uint64_t minimizerSpan = k - m + 1;
    static constexpr uint64_t findereSpan = k - s + 1;
    static constexpr uint64_t insertGroupSize = blockWordCount;
    static constexpr uint64_t queryGroupSize = 1;
    static constexpr uint64_t maxRunKmers = cudaBlockSize;

    static_assert(k > 0, "k must be positive");
    static_assert(m > 0 && m <= k, "m must satisfy 0 < m <= k");
    static_assert(s > 0 && s <= k, "s must satisfy 0 < s <= k");
    static_assert(k * symbolBits <= 64, "k-mer must fit in one packed uint64_t");
    static_assert(m * symbolBits <= 64, "m-mer must fit in one packed uint64_t");
    static_assert(s * symbolBits <= 64, "s-mer must fit in one packed uint64_t");
    static_assert(hashCount > 0, "At least one Bloom hash is required");
    static_assert(hashCount <= 16, "This implementation provides 16 multiplicative salts");
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

template <typename T>
struct BitwiseOr {
    __host__ __device__ __forceinline__ T operator()(T lhs, T rhs) const {
        return lhs | rhs;
    }
};

template <typename Config>
struct SequenceKmerInput;

template <typename Config>
__global__ void containsSequenceKmersKernel(
    SequenceKmerInput<Config> input,
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

/// @brief Sentinel hash value indicating "no valid minimizer found".
inline constexpr uint64_t kInvalidHash = std::numeric_limits<uint64_t>::max();
/**
 * @brief Compile-time golden-ratio-derived multiplicative salt constants.
 *
 * Used as per-hash-index mixing constants in the sectorised Bloom hash scheme.
 * Index must be in [0, 15].
 *
 * @tparam Index Salt index.
 */
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

/**
 * @brief Returns the multiplicative salt constant for hash function @p Index.
 *
 * @tparam Index Salt index in [0, 15].
 * @return Salt value.
 */
template <uint64_t Index>
[[nodiscard]] __host__ __device__ __forceinline__ constexpr uint64_t multiplicativeSaltLiteral() {
    static_assert(Index < 16, "Salt index out of range");
    return SaltLiteral<Index>::value;
}

/// @brief Implementation helper for forEachHashIndex (fold-expression over an index sequence).
template <typename Config, typename Fn, uint64_t... HashIndices>
__host__ __device__ __forceinline__ void
forEachHashIndexImpl(Fn&& fn, std::index_sequence<HashIndices...>) {
    (fn(std::integral_constant<uint64_t, HashIndices>{}), ...);
}

/**
 * @brief Invokes @p fn for each hash index in [0, Config::hashCount) at compile time.
 *
 * @tparam Config  Filter configuration.
 * @tparam Fn      Callable with signature @c void(std::integral_constant<uint64_t, I>).
 * @param  fn      Callable to invoke for each index.
 */
template <typename Config, typename Fn>
__host__ __device__ __forceinline__ void forEachHashIndex(Fn&& fn) {
    forEachHashIndexImpl<Config>(
        static_cast<Fn&&>(fn), std::make_index_sequence<Config::hashCount>{}
    );
}

/**
 * @brief Returns a bitmask covering @p Length packed alphabet symbols.
 *
 * Returns @c UINT64_MAX when the packed window consumes all 64 bits.
 *
 * @tparam Length Number of symbols.
 */
template <typename Config, uint64_t Length>
[[nodiscard]] __host__ __device__ __forceinline__ constexpr uint64_t packedWindowMask() {
    if constexpr (Length * Config::symbolBits >= 64) {
        return std::numeric_limits<uint64_t>::max();
    } else {
        return (uint64_t{1} << (Config::symbolBits * Length)) - 1;
    }
}

/**
 * @brief Extracts a packed sub-window from a packed k-mer.
 *
 * Extracts @p WindowLength consecutive bases starting at @p start from a
 * packed k-mer of length @p K (MSB = first base).
 *
 * @tparam WindowLength  Length of the sub-window to extract.
 * @tparam K             Length of the full k-mer.
 * @param  packedKmer    Packed k-mer (MSB = first base).
 * @param  start         Zero-based start position.
 * @return Packed sub-window.
 */
template <typename Config, uint64_t WindowLength, uint64_t K>
[[nodiscard]] __host__ __device__ __forceinline__ constexpr uint64_t
extractPackedSubwindow(uint64_t packedKmer, uint64_t start) {
    static_assert(WindowLength <= K, "WindowLength must not exceed K");
    return (packedKmer >> (Config::symbolBits * (K - (start + WindowLength)))) &
           packedWindowMask<Config, WindowLength>();
}

/**
 * @brief Atomically ORs @p value into the device word at @p ptr.
 *
 * @param  ptr       Target device word.
 * @param  value     Value to OR in.
 */
__device__ __forceinline__ void atomicOrWord(uint64_t* ptr, uint64_t value) {
    atomicOr(reinterpret_cast<unsigned long long*>(ptr), static_cast<unsigned long long>(value));
}

}  // namespace detail

/**
 * @brief GPU-accelerated sectorized Bloom filter.
 *
 * Stores an in-device Bloom filter divided into numShards 256-bit shards.
 * Each shard is independently addressed by a minimizer-derived hash, and
 * bits within a shard are updated/tested by a set of s-mer-derived hashes.
 *
 * The filter is **not copyable** (device memory ownership). Move construction
 * and assignment are supported.
 *
 * @tparam Config A @ref bloom::Config specialisation.
 */
template <typename Config>
class Filter {
   public:
    /**
     * @brief One 256-bit filter block stored as an array of Config::blockWordCount words.
     *
     * Each shard is addressed as a unit: a minimizer hash selects the shard,
     * and the s-mer-derived hashes set/test bits within it.
     *
     * The struct is 32-byte aligned to enable vectorised loads via
     * @ref bloom::detail::load256BitGlobalNC.
     */
    struct alignas(32) Shard {
        static constexpr uint64_t wordCount = Config::blockWordCount;
        static constexpr uint64_t wordBits = Config::wordBits;
        static constexpr int wordBitsLog2 = cuda::std::bit_width(wordBits) - 1;
        static constexpr uint64_t wordMask = (1ULL << wordBitsLog2) - 1;
        static constexpr int hashShift = 64 - wordBitsLog2;
        static constexpr uint64_t sliceWidth = 64 / Config::hashCount;
        static constexpr bool useBitSlicing = sliceWidth >= wordBitsLog2;

        uint64_t words[wordCount];

        /**
         * @brief Maps a base hash to a bit position within word sector @p HashIndex.
         *
         * Uses bit-slicing when the hash has enough entropy per slice; otherwise
         * applies a multiplicative salt to redistribute bits.
         *
         * @tparam HashIndex  Hash function index in [0, Config::hashCount).
         * @param  baseHash   The s-mer-derived hash value.
         * @return Bit position (0-based) within the word at sector @p HashIndex % blockWordCount.
         */
        template <uint64_t HashIndex>
        [[nodiscard]] constexpr __host__ __device__ static uint64_t sectorizedBitAddress(
            uint64_t baseHash
        ) {
            static_assert(HashIndex < Config::hashCount, "Hash index out of range");
            // When there are enough bits in a 64-bit hash to give each hash
            // index its own slice, avoid the extra multiply and use
            // bit-slicing instead.
            if constexpr (useBitSlicing) {
                return (baseHash >> (sliceWidth * HashIndex)) & wordMask;
            } else {
                const uint64_t mixed = baseHash * detail::multiplicativeSaltLiteral<HashIndex>();
                return mixed >> hashShift;
            }
        }

        /**
         * @brief Computes four word bitmasks from a single base hash.
         *
         * Iterates over all Config::hashCount hash functions and ORs the
         * corresponding bit into one of the four output word masks (sectors 0-3).
         *
         * @param baseHash  s-mer-derived hash value.
         * @param mask0     Accumulated bits for word 0 (in/out).
         * @param mask1     Accumulated bits for word 1 (in/out).
         * @param mask2     Accumulated bits for word 2 (in/out).
         * @param mask3     Accumulated bits for word 3 (in/out).
         */
        __device__ __forceinline__ static void sectorizedHashToMasks(
            uint64_t baseHash,
            uint64_t& mask0,
            uint64_t& mask1,
            uint64_t& mask2,
            uint64_t& mask3
        ) {
            detail::forEachHashIndex<Config>(
                [&]<uint64_t HashIndex>(std::integral_constant<uint64_t, HashIndex>) {
                    constexpr uint64_t s = HashIndex % Config::blockWordCount;
                    const uint64_t bitPos = sectorizedBitAddress<HashIndex>(baseHash);
                    const uint64_t bit = uint64_t{1} << bitPos;
                    // clang-format off
                    if      constexpr (s == 0) mask0 |= bit;
                    else if constexpr (s == 1) mask1 |= bit;
                    else if constexpr (s == 2) mask2 |= bit;
                    else                       mask3 |= bit;
                    // clang-format on
                }
            );
        }
    };

    static_assert(Config::blockWordCount == 4, "Filter only supports the fused 256-bit shard path");
    static_assert(
        Config::queryGroupSize == 1,
        "Fused path expects Theta=1 independent query mapping"
    );
    static_assert(
        Config::insertGroupSize == Config::blockWordCount,
        "Fused path expects horizontal insert mapping across shard words"
    );

    /**
     * @brief Constructs a Filter with at least @p requestedFilterBits bits of storage.
     *
     * The actual allocated capacity is rounded up to the next power-of-two number of
     * shards.
     *
     * @param requestedFilterBits Desired filter capacity in bits.
     */
    explicit Filter(uint64_t requestedFilterBits)
        : numShards_(
              cuda::std::bit_ceil(
                  std::max<uint64_t>(1, cuda::ceil_div(requestedFilterBits, Config::filterBlockBits))
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

    /**
     * @brief Inserts all valid k-mers from a host-resident sequence.
     *
     * Copies the sequence to device, launches the insert kernel, and synchronises
     * before returning. K-mers containing characters outside {A,C,G,T,a,c,g,t}
     * are skipped.
     *
     * @param sequence  Raw nucleotide sequence.
     * @param stream    CUDA stream to use (default: null stream).
     * @return Number of k-mers attempted (sequences shorter than k yield 0).
     */
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
        stream.sync();
        return totalKmers;
    }

    /**
     * @brief Async insert of k-mers from a device-resident sequence.
     *
     * Does **not** synchronise the stream, the caller is responsible for ordering
     * relative to downstream operations.
     *
     * @param d_sequence  Device-resident nucleotide sequence.
     * @param stream      CUDA stream to use.
     * @return Number of k-mers attempted.
     */
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

    /**
     * @brief Inserts all k-mers from a FASTA/FASTQ input stream.
     *
     * @param input   Input stream containing FASTA or FASTQ records.
     * @param stream  CUDA stream to use.
     * @return Report summarising records indexed, bases processed, and k-mers inserted.
     */
    [[nodiscard]] FastxInsertReport
    insertFastx(std::istream& input, cuda::stream_ref stream = cudaStream_t{}) {
        return insertFastxStream(input, "<stream>", stream);
    }

    /**
     * @brief Inserts all k-mers from a FASTA/FASTQ file.
     *
     * @param path    Path to the FASTA/FASTQ file.
     * @param stream  CUDA stream to use.
     * @return Report summarising records indexed, bases processed, and k-mers inserted.
     */
    [[nodiscard]] FastxInsertReport
    insertFastxFile(std::string_view path, cuda::stream_ref stream = cudaStream_t{}) {
        auto input = detail::openFastxFile(path);
        return insertFastxStream(*input, path, stream);
    }

    /**
     * @brief Async query of k-mers from a device-resident sequence.
     *
     * Does **not** synchronise the stream. Results are written to @p d_output
     * (one byte per k-mer: 1 = present, 0 = absent).
     *
     * @param d_sequence  Device-resident nucleotide sequence.
     * @param d_output    Per-k-mer result buffer (must hold kmerCount() bytes).
     * @param stream      CUDA stream to use.
     */
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

    /**
     * @brief Queries all valid k-mers from a host-resident sequence.
     *
     * Copies the sequence to device, queries, copies results back, and
     * synchronises. The returned vector has one byte per k-mer: 1 = present,
     * 0 = absent.
     *
     * @param sequence  Raw nucleotide sequence.
     * @param stream    CUDA stream to use.
     * @return Per-k-mer membership results (empty if sequence length < k).
     */
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
        BLOOM_CUDA_CALL(cudaMemcpyAsync(
            output.data(),
            thrust::raw_pointer_cast(d_resultBuffer_.data()),
            output.size() * sizeof(uint8_t),
            cudaMemcpyDeviceToHost,
            stream.get()
        ));

        stream.sync();
        return output;
    }

    /**
     * @brief Queries all k-mers from a FASTA/FASTQ input stream.
     *
     * @param input   Input stream containing FASTA or FASTQ records.
     * @param stream  CUDA stream to use.
     * @return Report summarising records queried, bases, and hit counts.
     */
    [[nodiscard]] FastxQueryReport
    queryFastx(std::istream& input, cuda::stream_ref stream = cudaStream_t{}) const {
        return queryFastxStream(input, "<stream>", stream);
    }

    /**
     * @brief Queries all k-mers from a FASTA/FASTQ file.
     *
     * @param path    Path to the FASTA/FASTQ file.
     * @param stream  CUDA stream to use.
     * @return Report summarising records queried, bases, and hit counts.
     */
    [[nodiscard]] FastxQueryReport
    queryFastxFile(std::string_view path, cuda::stream_ref stream = cudaStream_t{}) const {
        auto input = detail::openFastxFile(path);
        return queryFastxStream(*input, path, stream);
    }

    /**
     * @brief Resets all filter bits to zero and synchronises the stream.
     *
     * @param stream CUDA stream to use.
     */
    void clear(cuda::stream_ref stream = cudaStream_t{}) {
        cuda::fill_bytes(
            stream,
            cuda::std::span<uint8_t>{
                reinterpret_cast<uint8_t*>(thrust::raw_pointer_cast(d_shards_.data())), sizeBytes()
            },
            0
        );
        stream.sync();
    }

    /**
     * @brief Computes the fraction of set bits in the filter.
     *
     * @return Load factor in [0, 1].
     */
    [[nodiscard]] float loadFactor() const {
        const auto* wordsBegin =
            reinterpret_cast<const uint64_t*>(thrust::raw_pointer_cast(d_shards_.data()));
        const uint64_t totalWords = numShards_ * Config::blockWordCount;
        const uint64_t setBits = thrust::transform_reduce(
            thrust::device,
            wordsBegin,
            wordsBegin + totalWords,
            [] __device__(uint64_t w) -> uint64_t { return cuda::std::popcount(w); },
            uint64_t{0},
            cuda::std::plus<uint64_t>()
        );
        return static_cast<float>(setBits) / static_cast<float>(filterBits_);
    }

    /// @brief Returns the total allocated capacity of the filter in bits.
    [[nodiscard]] uint64_t filterBits() const {
        return filterBits_;
    }

    /// @brief Returns the number of shards.
    [[nodiscard]] uint64_t numShards() const {
        return numShards_;
    }

   private:
    struct FastxRecordRange {
        uint64_t offset{};
        uint64_t size{};
    };

    uint64_t numShards_{};
    uint64_t filterBits_{};

    thrust::device_vector<Shard> d_shards_;
    mutable thrust::device_vector<char> d_sequence_;
    mutable thrust::device_vector<uint8_t> d_resultBuffer_;

    /// @brief Returns the total size of all shard storage in bytes.
    [[nodiscard]] uint64_t sizeBytes() const {
        return numShards() * sizeof(Shard);
    }

    [[nodiscard]] static uint64_t recordKmerCount(uint64_t bases) {
        return bases < Config::k ? 0 : bases - Config::k + 1;
    }

    [[nodiscard]] static uint64_t validRecordKmerCount(std::string_view sequence) {
        if (sequence.size() < Config::k) {
            return 0;
        }

        uint64_t invalidSymbols = 0;
        for (uint64_t i = 0; i < Config::k; ++i) {
            invalidSymbols +=
                Config::Alphabet::encode(sequence[i]) == Config::Alphabet::invalidSymbol;
        }

        uint64_t validKmers = invalidSymbols == 0 ? 1 : 0;
        for (uint64_t start = 1; start < recordKmerCount(sequence.size()); ++start) {
            invalidSymbols -=
                Config::Alphabet::encode(sequence[start - 1]) == Config::Alphabet::invalidSymbol;
            invalidSymbols += Config::Alphabet::encode(sequence[start + Config::k - 1]) ==
                              Config::Alphabet::invalidSymbol;
            validKmers += invalidSymbols == 0;
        }
        return validKmers;
    }

    static void appendFastxRecord(std::string& sequence, std::string_view recordSequence) {
        if (!sequence.empty()) {
            sequence.push_back(static_cast<char>(Config::Alphabet::separator));
        }
        sequence.append(recordSequence);
    }

    static void appendFastxRecordWithRange(
        std::string& sequence,
        std::vector<FastxRecordRange>& ranges,
        std::string_view recordSequence
    ) {
        if (!sequence.empty()) {
            sequence.push_back(static_cast<char>(Config::Alphabet::separator));
        }
        const uint64_t offset = sequence.size();
        sequence.append(recordSequence);
        ranges.push_back(FastxRecordRange{offset, recordSequence.size()});
    }

    /// @brief Internal implementation shared by insertFastx() and insertFastxFile().
    [[nodiscard]] FastxInsertReport
    insertFastxStream(std::istream& input, std::string_view sourceName, cuda::stream_ref stream) {
        detail::FastxReader reader(input, sourceName);
        detail::FastxRecord record;
        FastxInsertReport report;
        std::string sequence;

        while (reader.nextRecord(record)) {
            ++report.recordsIndexed;
            report.indexedBases += record.sequence.size();
            report.insertedKmers += validRecordKmerCount(record.sequence);
            appendFastxRecord(sequence, record.sequence);
        }

        if (!sequence.empty()) {
            (void)insertSequence(sequence, stream);
        }
        return report;
    }

    /// @brief Internal implementation shared by queryFastx() and queryFastxFile().
    [[nodiscard]] FastxQueryReport queryFastxStream(
        std::istream& input,
        std::string_view sourceName,
        cuda::stream_ref stream
    ) const {
        detail::FastxReader reader(input, sourceName);
        detail::FastxRecord record;
        FastxQueryReport report;
        std::string sequence;
        std::vector<FastxRecordRange> ranges;

        while (reader.nextRecord(record)) {
            ++report.recordsQueried;
            report.queriedBases += record.sequence.size();
            report.queriedKmers += validRecordKmerCount(record.sequence);
            appendFastxRecordWithRange(sequence, ranges, record.sequence);
        }

        if (!sequence.empty()) {
            const auto hits = containsSequence(sequence, stream);
            for (const FastxRecordRange range : ranges) {
                const uint64_t kmers = recordKmerCount(range.size);
                if (kmers == 0) {
                    continue;
                }
                report.positiveKmers += std::count(
                    hits.begin() + static_cast<std::ptrdiff_t>(range.offset),
                    hits.begin() + static_cast<std::ptrdiff_t>(range.offset + kmers),
                    uint8_t{1}
                );
            }
        }
        return report;
    }

    /**
     * @brief Grows the host-to-device sequence staging buffer if necessary.
     * @param bases Minimum required capacity in characters.
     */
    void ensureSequenceCapacity(uint64_t bases) const {
        if (bases > d_sequence_.size()) {
            d_sequence_.resize(bases);
        }
    }

    /**
     * @brief Grows the per-k-mer result staging buffer if necessary.
     * @param kmers Minimum required capacity in bytes.
     */
    void ensureResultCapacity(uint64_t kmers) const {
        if (kmers > d_resultBuffer_.size()) {
            d_resultBuffer_.resize(kmers);
        }
    }

    /**
     * @brief Copies a host-resident sequence to the device staging buffer.
     * @param sequence Source span (host memory).
     * @param stream   CUDA stream.
     */
    void stageSequence(cuda::std::span<const char> sequence, cuda::stream_ref stream) const {
        ensureSequenceCapacity(sequence.size());
        BLOOM_CUDA_CALL(cudaMemcpyAsync(
            thrust::raw_pointer_cast(d_sequence_.data()),
            sequence.data(),
            sequence.size_bytes(),
            cudaMemcpyHostToDevice,
            stream.get()
        ));
    }

    /**
     * @brief Launches the insert kernel for a device-resident sequence.
     * @param d_sequence Device-resident sequence.
     * @param stream     CUDA stream.
     */
    void launchInsertSequence(device_span<const char> d_sequence, cuda::stream_ref stream) {
        if (d_sequence.size() < Config::k) {
            return;
        }
        const uint64_t numKmers = d_sequence.size() - Config::k + 1;
        const uint64_t gridSize = cuda::ceil_div(numKmers, Config::cudaBlockSize);

        detail::insertSequenceKmersKernel<Config>
            <<<gridSize, Config::cudaBlockSize, 0, stream.get()>>>(
                detail::SequenceKmerInput<Config>{d_sequence},
                device_span<Shard>{thrust::raw_pointer_cast(d_shards_.data()), numShards_}
            );
        BLOOM_CUDA_CALL(cudaGetLastError());
    }

    /**
     * @brief Launches the query kernel for a device-resident sequence.
     * @param d_sequence Device-resident sequence.
     * @param d_output   Per-k-mer result buffer (one byte per k-mer).
     * @param stream     CUDA stream.
     */
    void launchContainsSequence(
        device_span<const char> d_sequence,
        device_span<uint8_t> d_output,
        cuda::stream_ref stream
    ) const {
        const uint64_t numKmers = d_sequence.size() - Config::k + 1;
        cuda::fill_bytes(stream, d_output, 0);
        constexpr uint64_t kStride = 4;
        const uint64_t gridSize = cuda::ceil_div(numKmers, Config::cudaBlockSize * kStride);

        detail::containsSequenceKmersKernel<Config>
            <<<gridSize, Config::cudaBlockSize, 0, stream.get()>>>(
                detail::SequenceKmerInput<Config>{d_sequence},
                device_span<const Shard>{thrust::raw_pointer_cast(d_shards_.data()), numShards_},
                d_output
            );
        BLOOM_CUDA_CALL(cudaGetLastError());
    }
};

namespace detail {

/**
 * @brief Kernel input descriptor for a sequence k-mer sweep.
 *
 * Passed by value to both insert and query kernels; holds the device-resident
 * sequence span and provides convenience accessors for k-mer and s-mer counts.
 */
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

/**
 * @brief Computes the minimizer hash for a packed k-mer.
 *
 * Iterates over all m-mers within the k-mer and returns the minimum
 * hash value, which is used to select the target shard.
 *
 * @tparam Config     Filter configuration.
 * @param  packedKmer 2-bit packed k-mer.
 * @return Minimizer hash value.
 */
template <typename Config>
[[nodiscard]] __device__ __forceinline__ uint64_t packedKmerMinimizerHash(uint64_t packedKmer) {
    uint64_t minimizerHash = kInvalidHash;
    _Pragma("unroll")
    for (uint64_t offset = 0; offset < Config::minimizerSpan; ++offset) {
        const uint64_t packedMmer =
            extractPackedSubwindow<Config, Config::m, Config::k>(packedKmer, offset);
        minimizerHash = min(minimizerHash, detail::minimizerHash64(packedMmer));
    }
    return minimizerHash;
}

/**
 * @brief Computes the hash for the s-mer at position @p start within a packed k-mer.
 *
 * @tparam Config     Filter configuration.
 * @param  packedKmer 2-bit packed k-mer.
 * @param  start      Zero-based start position of the s-mer within the k-mer.
 * @return Hash of the s-mer.
 */
template <typename Config>
[[nodiscard]] __device__ __forceinline__ uint64_t
packedKmerSmerHash(uint64_t packedKmer, uint64_t start) {
    const uint64_t packedSmer =
        extractPackedSubwindow<Config, Config::s, Config::k>(packedKmer, start);
    return detail::hash64(packedSmer);
}

/**
 * @brief Loads all four 64-bit words of a shard into a local array.
 *
 * On sm_100+ issues a single 256-bit non-coherent global load, on older
 * architectures falls back to two 128-bit loads.
 *
 * @tparam Config      Filter configuration.
 * @param  shards      Pointer to the device shard array.
 * @param  shardIndex  Index of the shard to load.
 * @param  w           Output array of (at least) four words.
 */
template <typename Config>
__device__ __forceinline__ void
loadShardWords4(const typename Filter<Config>::Shard* shards, uint64_t shardIndex, uint64_t* w) {
#if __CUDA_ARCH__ >= 1000
    detail::load256BitGlobalNC(shards[shardIndex].words, w[0], w[1], w[2], w[3]);
#else
    detail::load128BitGlobalNC(shards[shardIndex].words + 0, w[0], w[1]);
    detail::load128BitGlobalNC(shards[shardIndex].words + 2, w[2], w[3]);
#endif
}

/**
 * @brief Packs @p K symbols from a shared-memory tile into an integer.
 *
 * @tparam K     k-mer length.
 * @param  tile  Encoded symbol tile in shared memory.
 * @param  start Start position within the tile.
 * @return Packed k-mer.
 */
template <typename Config, uint64_t K>
__device__ __forceinline__ uint64_t packKmerFromTile(const uint8_t* tile, uint64_t start) {
    uint64_t packed = 0;
    _Pragma("unroll")
    for (uint64_t i = 0; i < K; ++i) {
        packed = (packed << Config::symbolBits) | (tile[start + i] & Config::symbolMask);
    }
    return packed;
}

/**
 * @brief Slides the packed k-mer window forward by one symbol.
 *
 * Shifts the existing packed representation left by one symbol, inserts the
 * new symbol in the least-significant position, and masks to @p K symbols.
 *
 * @tparam K       k-mer length.
 * @param  packed  Current packed k-mer.
 * @param  newBase Pre-encoded new symbol.
 * @return Updated packed k-mer.
 */
template <typename Config, uint64_t K>
__device__ __forceinline__ uint64_t advancePackedKmer(uint64_t packed, uint8_t newBase) {
    return ((packed << Config::symbolBits) | (newBase & Config::symbolMask)) &
           packedWindowMask<Config, K>();
}

/**
 * @brief Tests whether a packed k-mer is present in a pre-loaded shard.
 *
 * Checks all s-mer hashes across the k-mer against the four shard words.
 * Returns @c false as soon as any required bit is absent.
 *
 * @tparam Config     Filter configuration.
 * @param  packedKmer Packed k-mer to query.
 * @param  w          The four pre-loaded shard words.
 * @return @c true if all required bits are set.
 */
template <typename Config>
__device__ __forceinline__ bool
sectorizedContainsPackedKmer(uint64_t packedKmer, const uint64_t* w) {
    bool present = true;
    _Pragma("unroll")
    for (uint64_t smerOffset = 0; smerOffset < Config::findereSpan; ++smerOffset) {
        const uint64_t smerHash = packedKmerSmerHash<Config>(packedKmer, smerOffset);
        detail::forEachHashIndex<Config>(
            [&]<uint64_t HashIndex>(std::integral_constant<uint64_t, HashIndex>) {
                constexpr uint64_t s = HashIndex % Config::blockWordCount;
                const uint64_t bitPos =
                    Filter<Config>::Shard::template sectorizedBitAddress<HashIndex>(smerHash);
                present &= ((w[s] >> bitPos) & 1) != 0;
            }
        );
    }
    return present;
}

/**
 * @brief Cooperatively loads and encodes a tile of symbols into shared memory.
 *
 * All threads in the block participate. The return value (via
 * @c __syncthreads_count) is @c true only if every base in the tile is a
 * valid alphabet symbol.
 *
 * @tparam Config         Filter configuration.
 * @param  sequence       Device-resident sequence pointer.
 * @param  blockStartKmer Index of the first k-mer assigned to this block.
 * @param  blockKmers     Number of k-mers handled by this block.
 * @param  sequenceTile   Shared-memory output buffer (blockKmers + k - 1 bytes).
 * @return @c true if no invalid symbols are present in the tile.
 */
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
        const uint8_t encodedBase = Config::Alphabet::encode(sequence[blockStartKmer + idx]);
        sequenceTile[idx] = encodedBase;
        localInvalidBase |= (encodedBase == Config::Alphabet::invalidSymbol);
    }
    return __syncthreads_count(localInvalidBase) == 0;
}

/**
 * @brief CUDA kernel: queries k-mers from a sequence against the filter.
 *
 * Each thread processes @c kStride consecutive k-mers to amortise packing and
 * shard loads. Threads sharing the same shard collaborate via
 * @c __match_any_sync to load the shard words once and broadcast them.
 *
 * @tparam Config  Filter configuration.
 * @param  input   Sequence descriptor (device span + k-mer count).
 * @param  shards  Device-resident shard array (read-only).
 * @param  output  Per-k-mer result buffer (1 = present, 0 = absent).
 */
template <typename Config>
__global__ __launch_bounds__(Config::cudaBlockSize, 6) void containsSequenceKmersKernel(
    SequenceKmerInput<Config> input,
    device_span<const typename Filter<Config>::Shard> shards,
    device_span<uint8_t> output
) {
    // Each thread handles this many consecutive k-mers to amortise packing
    constexpr uint32_t kStride = 4;
    constexpr uint64_t sequenceTileBases = Config::cudaBlockSize * kStride + Config::k - 1;

    __shared__ uint8_t sequenceTile[sequenceTileBases];

    const uint64_t numKmers = input.kmerCount();
    const uint64_t blockStartKmer =
        static_cast<uint64_t>(blockIdx.x) * Config::cudaBlockSize * kStride;
    if (blockStartKmer >= numKmers) {
        return;
    }

    const uint64_t blockKmers = min(Config::cudaBlockSize * kStride, numKmers - blockStartKmer);

    const bool blockAllValid = prepareSequenceHashTiles<Config>(
        input.sequence.data(), blockStartKmer, blockKmers, sequenceTile
    );

    const uint64_t threadOffset = static_cast<uint64_t>(threadIdx.x) * kStride;
    if (threadOffset >= blockKmers) {
        return;
    }

    // Bitmask: bit s set = k-mer at offset s is valid.
    uint32_t kmerValidMask = 0;
    _Pragma("unroll")
    for (uint32_t s = 0; s < kStride; ++s) {
        if ((threadOffset + s) < blockKmers) {
            kmerValidMask |= (1u << s);
        }
    }

    if (!blockAllValid) {
        _Pragma("unroll")
        for (uint32_t s = 0; s < kStride; ++s) {
            if (!(kmerValidMask & (1u << s))) {
                continue;
            }
            const uint64_t localIdx = threadOffset + s;
            bool valid = true;
            _Pragma("unroll")
            for (uint64_t i = 0; i < Config::k; ++i) {
                if (sequenceTile[localIdx + i] == Config::Alphabet::invalidSymbol) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                kmerValidMask &= ~(1u << s);
            }
        }
    }

    // Always pack from position 0.  Sliding propagates the packed value forward
    // invalid bases from earlier k-mers are simply shifted out.
    uint64_t packedKmer = packKmerFromTile<Config, Config::k>(sequenceTile, threadOffset);

    for (uint32_t s = 0; s < kStride; ++s) {
        const uint64_t localIdx = threadOffset + s;
        if (localIdx >= blockKmers) {
            break;
        }

        const uint64_t kmerIndex = blockStartKmer + localIdx;

        if (s > 0) {
            packedKmer = advancePackedKmer<Config, Config::k>(
                packedKmer, sequenceTile[localIdx + Config::k - 1]
            );
        }

        if (!(kmerValidMask & (1u << s))) {
            output[kmerIndex] = 0;
            continue;
        }

        const uint64_t minimizerHash = packedKmerMinimizerHash<Config>(packedKmer);

        // Warp-level shard sharing.
        const auto shardIdx = static_cast<uint32_t>(minimizerHash & (shards.size() - 1));
        const uint32_t peers = __match_any_sync(0xFFFFFFFFu, shardIdx);
        const int leader = __ffs(static_cast<int>(peers)) - 1;

        uint64_t w[4];
        if (static_cast<int>(threadIdx.x & 31u) == leader) {
            loadShardWords4<Config>(shards.data(), shardIdx, w);
        }
        w[0] = __shfl_sync(peers, w[0], leader);
        w[1] = __shfl_sync(peers, w[1], leader);
        w[2] = __shfl_sync(peers, w[2], leader);
        w[3] = __shfl_sync(peers, w[3], leader);

        const bool present = sectorizedContainsPackedKmer<Config>(packedKmer, w);
        output[kmerIndex] = present;
    }
}

/**
 * @brief CUDA kernel: inserts k-mers from a sequence into the filter.
 *
 * Each thread processes one k-mer. Consecutive threads targeting the same
 * shard use @c cub::WarpReduce::HeadSegmentedReduce to merge bitmasks before
 * the run head issues a minimal number of @c atomicOr operations.
 *
 * @tparam Config  Filter configuration.
 * @param  input   Sequence descriptor.
 * @param  shards  Device-resident shard array (modified in place).
 */
template <typename Config>
__global__ void insertSequenceKmersKernel(
    SequenceKmerInput<Config> input,
    device_span<typename Filter<Config>::Shard> shards
) {
    constexpr uint64_t sequenceTileBases = Config::cudaBlockSize + Config::k - 1;
    constexpr uint32_t warpSize = 32;
    constexpr uint32_t warpsPerBlock = Config::cudaBlockSize / warpSize;

    using WarpReduceWord = cub::WarpReduce<uint64_t>;

    __shared__ uint8_t sequenceTile[sequenceTileBases];
    __shared__ typename WarpReduceWord::TempStorage reduceStorage[warpsPerBlock][4];

    const uint64_t numKmers = input.kmerCount();
    const uint64_t blockStartKmer = static_cast<uint64_t>(blockIdx.x) * Config::cudaBlockSize;
    if (blockStartKmer >= numKmers) {
        return;
    }

    const uint64_t blockKmers = min(Config::cudaBlockSize, numKmers - blockStartKmer);
    const auto localKmerIndex = static_cast<uint64_t>(threadIdx.x);
    const bool inRange = localKmerIndex < blockKmers;

    const bool blockAllValid = prepareSequenceHashTiles<Config>(
        input.sequence.data(), blockStartKmer, blockKmers, sequenceTile
    );

    // Avoid early returns so all warp lanes can participate in the segmented
    // warp reductions below.
    bool active = inRange;

    if (active && !blockAllValid) {
        _Pragma("unroll")
        for (uint64_t i = 0; i < Config::k; ++i) {
            if (sequenceTile[localKmerIndex + i] == Config::Alphabet::invalidSymbol) {
                active = false;
                break;
            }
        }
    }

    // Inactive threads keep zero masks and a per-lane sentinel shard index so
    // contiguous run detection naturally splits around them.
    uint64_t minimizerHash = 0;
    uint64_t wordMask0 = 0;
    uint64_t wordMask1 = 0;
    uint64_t wordMask2 = 0;
    uint64_t wordMask3 = 0;

    if (active) {
        const uint64_t packedKmer =
            packKmerFromTile<Config, Config::k>(sequenceTile, localKmerIndex);
        minimizerHash = packedKmerMinimizerHash<Config>(packedKmer);

        uint64_t h_s = packedKmerSmerHash<Config>(packedKmer, 0);
        Filter<Config>::Shard::sectorizedHashToMasks(
            h_s, wordMask0, wordMask1, wordMask2, wordMask3
        );
        _Pragma("unroll")
        for (uint64_t smerOffset = 1; smerOffset < Config::findereSpan; ++smerOffset) {
            h_s = packedKmerSmerHash<Config>(packedKmer, smerOffset);
            Filter<Config>::Shard::sectorizedHashToMasks(
                h_s, wordMask0, wordMask1, wordMask2, wordMask3
            );
        }
    }

    // Warp-local segmented reductions: contiguous threads sharing the same
    // shard merge their masks so only the run head issues the atomicOrs.
    const auto shardIdx =
        static_cast<uint32_t>(active ? (minimizerHash & (shards.size() - 1)) : ~threadIdx.x);

    const uint32_t lane = threadIdx.x & (warpSize - 1);
    const uint32_t warpIdx = threadIdx.x / warpSize;
    const uint32_t prevShardIdx = __shfl_up_sync(0xffffffff, shardIdx, 1);
    const bool runHead = (lane == 0) || (shardIdx != prevShardIdx);
    const BitwiseOr<uint64_t> bitwiseOr{};

    wordMask0 = WarpReduceWord(reduceStorage[warpIdx][0])
                    .HeadSegmentedReduce(wordMask0, runHead, bitwiseOr);
    wordMask1 = WarpReduceWord(reduceStorage[warpIdx][1])
                    .HeadSegmentedReduce(wordMask1, runHead, bitwiseOr);
    wordMask2 = WarpReduceWord(reduceStorage[warpIdx][2])
                    .HeadSegmentedReduce(wordMask2, runHead, bitwiseOr);
    wordMask3 = WarpReduceWord(reduceStorage[warpIdx][3])
                    .HeadSegmentedReduce(wordMask3, runHead, bitwiseOr);

    if (runHead && active) {
        auto& shard = shards[shardIdx];
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
