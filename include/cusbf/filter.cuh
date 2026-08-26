#pragma once

#include <cuda/__cmath/ceil_div.h>
#include <cuda_runtime.h>

#include <cuda/std/bit>
#include <cuda/stream>

#include <thrust/copy.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/transform_reduce.h>
#include <cusbf/detail/record_math.cuh>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <cusbf/Alphabet.cuh>
#include <cusbf/config.cuh>
#include <cusbf/detail/chunk_stream_pair.cuh>
#include <cusbf/detail/count_positive_kmers.cuh>
#include <cusbf/detail/dense_packed.cuh>
#include <cusbf/detail/fastx_chunk.cuh>
#include <cusbf/detail/fastx_dense_batch.hpp>
#include <cusbf/detail/fastx_dispatch.hpp>
#include <cusbf/detail/fastx_host_limits.cuh>
#include <cusbf/detail/fastx_pinned_buffer.hpp>
#include <cusbf/detail/fastx_pipeline_core.cuh>
#include <cusbf/detail/fastx_pipeline_state.cuh>
#include <cusbf/detail/filter_impl.cuh>
#include <cusbf/detail/kernels.cuh>
#include <cusbf/detail/sequence_kmer.cuh>
#include <cusbf/device_span.cuh>
#include <cusbf/error.hpp>
#include <cusbf/Fastx.hpp>
#include <cusbf/filter_ref.cuh>
#include <cusbf/hashutil.cuh>
#include <cusbf/helpers.cuh>
#include <cusbf/normalized_record_batch.hpp>

namespace cusbf {

/**
 * @brief GPU-resident Super Bloom filter for batch k-mer insert and query.
 *
 * @p Config fixes k-mer length, minimizer and s-mer widths, hash count, CUDA
 * block size, and alphabet at compile time. Host bulk APIs synchronize before
 * returning, device-span @c *_async methods do not.
 *
 * FASTX and record-batch paths inject alphabet separator bytes between
 * records so cross-record k-mers are never formed.
 *
 * @tparam Config Compile-time filter configuration (@ref cusbf::Config).
 */
template <typename Config>
class filter {
   private:
    struct FastxRecordHeaderRef {
        std::string header;
        uint64_t record_index{};
    };

   public:
    /**
     * @brief One 256-bit filter block stored as an array of Config::blockWordCount words.
     *
     * Each shard is addressed as a unit: a minimizer hash selects the shard,
     * and the s-mer-derived hashes set/test bits within it.
     *
     * The struct is 32-byte aligned to enable vectorised loads via
     * @ref cusbf::detail::load256BitGlobalNC.
     */
    using block_type = filter_block<Config>;
    /// Alias for @ref block_type (one 256-bit shard).
    using Shard = block_type;

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
    explicit filter(uint64_t requestedFilterBits)
        : num_shards_(
              cuda::std::bit_ceil(
                  std::max<uint64_t>(
                      1,
                      cuda::ceil_div(requestedFilterBits, Config::filterBlockBits)
                  )
              )
          ),
          filter_bits_(num_shards_ * Config::filterBlockBits),
          d_shards_(num_shards_) {
        CUSBF_UNWRAP(clear());
    }

    /// Non-copyable (owns device shard storage).
    filter(const filter&) = delete;
    filter& operator=(const filter&) = delete;
    /// Move-constructs, transfers shard vectors and staging buffers.
    filter(filter&&) noexcept = default;
    /// Move-assigns shard storage and staging buffers.
    filter& operator=(filter&&) noexcept = default;
    /// Destroys device allocations and releases staging scratch.
    ~filter() = default;

    /**
     * @brief Non-owning device reference to this filter's shard storage.
     *
     * Trivially copyable, intended to be passed by value into CUDA kernels.
     */
    [[nodiscard]] filter_ref<Config> ref() const noexcept {
        return filter_ref<Config>{
            thrust::raw_pointer_cast(d_shards_.data()),
            num_shards_,
        };
    }

    /**
     * @brief Inserts all valid k-mers from a host-resident sequence.
     *
     * Copies the sequence to device, launches the insert kernel, and synchronises
     * before returning. K-mer windows containing alphabet-invalid symbols are skipped.
     *
     * @param sequence  Host-resident sequence bytes (alphabet-encoded width per symbol).
     * @param stream    CUDA stream to use (default: null stream).
     * @return Number of k-mer windows attempted (sequences shorter than @c k yield 0).
     */
    [[nodiscard]] Result<uint64_t>
    insert_sequence(std::string_view sequence, cuda::stream_ref stream = cudaStream_t{}) {
        if (record_symbol_count(sequence.size()) < Config::k) {
            return 0;
        }

        const uint64_t totalKmers = record_kmer_count(sequence.size());
        const auto d_sequence = CUSBF_TRY(
            fastx_state_.staged_sequence_view({sequence.data(), sequence.size()}, stream)
        );
        CUSBF_TRY(launch_insert_sequence(d_sequence, stream));
        CUSBF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
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
    [[nodiscard]] Result<uint64_t> insert_sequence_async(
        device_span<const char> d_sequence,
        cuda::stream_ref stream = cudaStream_t{}
    ) {
        const uint64_t totalKmers = sequence_kmer_count(d_sequence);
        if (totalKmers == 0) {
            return 0;
        }

        CUSBF_TRY(launch_insert_sequence(d_sequence, stream));
        return totalKmers;
    }

    /**
     * @brief Inserts all k-mers from a dense packed symbol buffer on the device.
     *
     * @p d_words stores @ref dense_packed_word_count(num_symbols) words using
     * @ref Config::symbolBits per encoded symbol. Adjacent k-mers overlap in the same
     * @c uint64_t chunks; this path decodes a per-block symbol tile and slides packed
     * k-mers like @ref insert_sequence_async.
     *
     * Does **not** synchronise the stream.
     *
     * @param d_words       Device-resident dense packed sequence.
     * @param num_symbols   Number of valid encoded symbols in @p d_words.
     * @param stream        CUDA stream to use.
     * @return Number of k-mers attempted.
     */
    [[nodiscard]] Result<uint64_t> insert_dense_packed_async(
        device_span<const uint64_t> d_words,
        uint64_t num_symbols,
        cuda::stream_ref stream = cudaStream_t{}
    ) {
        const uint64_t totalKmers = dense_packed_kmer_count(num_symbols);
        if (totalKmers == 0) {
            return 0;
        }

        CUSBF_TRY(launch_insert_dense_packed(d_words, num_symbols, stream));
        return totalKmers;
    }

    /**
     * @brief Inserts all k-mers from a host-resident dense packed symbol buffer.
     *
     * Copies @p d_words to device staging, launches the insert kernel, and synchronises.
     */
    [[nodiscard]] Result<uint64_t> insert_dense_packed(
        std::span<const uint64_t> words,
        uint64_t num_symbols,
        cuda::stream_ref stream = cudaStream_t{}
    ) {
        const uint64_t totalKmers = dense_packed_kmer_count(num_symbols);
        if (totalKmers == 0) {
            return 0;
        }

        const auto staged = CUSBF_TRY(fastx_state_.staged_dense_packed_view(words, stream));
        CUSBF_TRY(launch_insert_dense_packed(staged, num_symbols, stream));
        CUSBF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
        return totalKmers;
    }

    /**
     * @brief Async query of k-mers from a dense packed symbol buffer on the device.
     *
     * @p d_output receives one byte per k-mer (1 = present, 0 = absent). Does **not**
     * synchronise the stream.
     */
    [[nodiscard]] Result<void> contains_dense_packed_async(
        device_span<const uint64_t> d_words,
        uint64_t num_symbols,
        device_span<uint8_t> d_output,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        if (dense_packed_kmer_count(num_symbols) == 0) {
            return {};
        }

        return launch_contains_dense_packed(d_words, num_symbols, d_output, stream);
    }

    /**
     * @brief Queries all k-mers from a host-resident dense packed symbol buffer into device memory.
     *
     * Copies @p words to device, launches the query kernel into @p d_output, and synchronises.
     * Does not allocate the result buffer or copy hits back to the host.
     */
    [[nodiscard]] Result<void> contains_dense_packed(
        std::span<const uint64_t> words,
        uint64_t num_symbols,
        device_span<uint8_t> d_output,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        const uint64_t numKmers = dense_packed_kmer_count(num_symbols);
        if (numKmers == 0) {
            return {};
        }

        const auto staged = CUSBF_TRY(fastx_state_.staged_dense_packed_view(words, stream));
        CUSBF_TRY(launch_contains_dense_packed(staged, num_symbols, d_output, stream));
        CUSBF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
        return {};
    }

    /**
     * @brief Queries all k-mers from a host-resident dense packed symbol buffer.
     *
     * Copies @p d_words to device, queries, copies results back, and synchronises.
     */
    [[nodiscard]] Result<std::vector<uint8_t>> contains_dense_packed(
        std::span<const uint64_t> words,
        uint64_t num_symbols,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        const uint64_t numKmers = dense_packed_kmer_count(num_symbols);
        if (numKmers == 0) {
            return std::vector<uint8_t>{};
        }

        std::vector<uint8_t> output(numKmers);
        fastx_state_.ensure_result_capacity(output.size());
        const device_span<uint8_t> d_output{
            thrust::raw_pointer_cast(fastx_state_.result_buffer_device().data()), output.size()
        };
        CUSBF_TRY(contains_dense_packed(words, num_symbols, d_output, stream));
        CUSBF_CUDA_TRY(cudaMemcpy(
            output.data(),
            thrust::raw_pointer_cast(fastx_state_.result_buffer_device().data()),
            output.size() * sizeof(uint8_t),
            cudaMemcpyDeviceToHost
        ));
        return output;
    }

    /// @brief Number of @c uint64_t words for @p num_symbols dense packed symbols.
    [[nodiscard]] static constexpr uint64_t dense_packed_word_count(uint64_t num_symbols) {
        return detail::dense_packed_word_count<Config>(num_symbols);
    }

    /// @brief Number of k-mer windows in a dense packed sequence of @p num_symbols symbols.
    [[nodiscard]] constexpr uint64_t dense_packed_kmer_count(uint64_t num_symbols) const {
        return detail::dense_packed_kmer_count<Config>(num_symbols);
    }

    /**
     * @brief Inserts a dense host-resident record batch.
     *
     * @p batch.sequence stores the raw record payloads back-to-back without separators.
     * @p batch.records stores ordered, non-overlapping byte ranges into that dense buffer.
     * The filter injects alphabet separators between records internally, so callers do not
     * need to materialise separator bytes themselves.
     *
     * Synchronises before returning.
     *
     * @param batch   Dense record batch to insert.
     * @param stream  CUDA stream to use.
     * @return Report summarising records indexed, bases processed, and k-mers inserted.
     */
    [[nodiscard]] Result<FastxInsertReport>
    insert_record_batch(RecordBatchView batch, cuda::stream_ref stream = cudaStream_t{}) {
        CUSBF_TRY(
            normalize_record_batch_into<Config>(
                batch,
                fastx_state_.normalized_sequence_scratch(),
                fastx_state_.normalized_records_scratch()
            )
        );
        FastxInsertReport report;
        report.recordsIndexed = fastx_state_.normalized_records_scratch().size();
        for (const NormalizedRecord& record : fastx_state_.normalized_records_scratch()) {
            report.indexedBases += record.size;
            report.insertedKmers += record.valid_kmers;
        }
        if (!fastx_state_.normalized_sequence_scratch().empty()) {
            const auto d_sequence = CUSBF_TRY(fastx_state_.staged_sequence_view(
                {fastx_state_.normalized_sequence_scratch().data(),
                 fastx_state_.normalized_sequence_scratch().size()},
                stream
            ));
            CUSBF_TRY(insert_sequence_async(d_sequence, stream));
            CUSBF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
        }
        fastx_state_.release_all();
        return report;
    }

    /**
     * @brief Async insert of k-mers from a device-resident record batch.
     *
     * The device sequence is assumed to already contain alphabet separator bytes
     * between records, the caller is responsible for embedding them so that
     * cross-record k-mers are not formed. @p records provides ordered,
     * non-overlapping byte ranges into @p d_sequence.
     *
     * The returned count is total attempted k-mers (maximum possible per record
     * size). The kernel still skips k-mer windows that contain invalid symbols.
     *
     * Does **not** synchronise the stream.
     *
     * @param d_sequence  Device-resident sequence (records concatenated with
     *                    separators already in place).
     * @param records     Ordered, non-overlapping record byte ranges.
     * @param stream      CUDA stream to use.
     * @return Total attempted k-mers across all records (0 if none).
     */
    [[nodiscard]] Result<uint64_t> insert_record_batch_async(
        device_span<const char> d_sequence,
        std::span<const RecordRange> records,
        cuda::stream_ref stream = cudaStream_t{}
    ) {
        CUSBF_TRY(validate_device_record_batch(d_sequence, records));
        uint64_t totalKmers = 0;
        for (const RecordRange& record : records) {
            totalKmers += record_kmer_count(record.sequenceBytes);
        }
        if (totalKmers == 0) {
            return 0;
        }
        CUSBF_TRY(launch_insert_sequence(d_sequence, stream));
        return totalKmers;
    }

    /**
     * @brief Inserts all k-mers from a FASTA/FASTQ input stream.
     *
     * Reads records in streaming fashion, accumulating them until the
     * concatenated sequence approaches a GPU staging budget derived from free
     * VRAM (@p fill_fraction of free memory minus a small slack reserve), then
     * inserts each chunk independently. The budget accounts for normalized
     * sequence device staging.
     *
     * @param input        Input stream containing FASTA or FASTQ records.
     * @param fill_fraction Fraction of available GPU memory for per-chunk staging (default 0.7).
     * @param stream       CUDA stream to use.
     * @return Report summarising records indexed, bases processed, and k-mers inserted.
     */
    [[nodiscard]] Result<FastxInsertReport> insert_fastx(
        std::istream& input,
        double fill_fraction = 0.7,
        cuda::stream_ref stream = cudaStream_t{}
    ) {
        return insert_fastx_stream(input, "<stream>", fill_fraction, stream);
    }

    /**
     * @brief Inserts all k-mers from a FASTA/FASTQ file.
     *
     * Uses @p fill_fraction of free GPU memory to size chunks. When the file fits in a
     * single chunk, reads via a lightweight stream (no mmap, no ping-pong). Larger files
     * are mmap'd if they fit in host RAM, otherwise streamed, multi-chunk paths overlap
     * host assembly with GPU work.
     *
     * @param path           Path to a FASTA or FASTQ file (optionally gzip-compressed).
     * @param fill_fraction  Fraction of free GPU memory for per-chunk staging (default 0.7).
     * @param stream         Optional CUDA stream, default uses an internal pipelined path.
     * @return Report summarising records indexed, bases processed, and k-mers inserted.
     */
    [[nodiscard]] Result<FastxInsertReport> insert_fastx_file(
        const std::filesystem::path& path,
        double fill_fraction = 0.7,
        cuda::stream_ref stream = cudaStream_t{}
    ) {
        return detail::dispatch_fastx_file<Config>(
            path,
            detail::fastx_chunk_mode::insert,
            fill_fraction,
            [&](auto& reader, auto dispatch_path) {
                return insert_fastx_reader(
                    reader, std::string_view{path.native()}, fill_fraction, stream, dispatch_path
                );
            }
        );
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
    [[nodiscard]] Result<void> contains_sequence_async(
        device_span<const char> d_sequence,
        device_span<uint8_t> d_output,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        if (sequence_kmer_count(d_sequence) == 0) {
            return {};
        }

        return launch_contains_sequence(d_sequence, d_output, stream);
    }

    /**
     * @brief Queries all valid k-mers from a host-resident sequence into device memory.
     *
     * Copies the sequence to device, launches the query kernel into @p d_output, and
     * synchronises. Does not allocate the result buffer or copy hits back to the host.
     *
     * @param sequence   Raw nucleotide sequence.
     * @param d_output   Per-k-mer result buffer (must hold kmerCount() bytes).
     * @param stream     CUDA stream to use.
     */
    [[nodiscard]] Result<void> contains_sequence(
        std::string_view sequence,
        device_span<uint8_t> d_output,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        if (record_symbol_count(sequence.size()) < Config::k) {
            return {};
        }

        const auto d_sequence = CUSBF_TRY(
            fastx_state_.staged_sequence_view({sequence.data(), sequence.size()}, stream)
        );
        CUSBF_TRY(launch_contains_sequence(d_sequence, d_output, stream));
        CUSBF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
        return {};
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
    [[nodiscard]] Result<std::vector<uint8_t>>
    contains_sequence(std::string_view sequence, cuda::stream_ref stream = cudaStream_t{}) const {
        if (record_symbol_count(sequence.size()) < Config::k) {
            return std::vector<uint8_t>{};
        }

        std::vector<uint8_t> output(record_kmer_count(sequence.size()));
        fastx_state_.ensure_result_capacity(output.size());
        const device_span<uint8_t> d_output{
            thrust::raw_pointer_cast(fastx_state_.result_buffer_device().data()), output.size()
        };
        CUSBF_TRY(contains_sequence(sequence, d_output, stream));
        CUSBF_CUDA_TRY(cudaMemcpy(
            output.data(),
            thrust::raw_pointer_cast(fastx_state_.result_buffer_device().data()),
            output.size() * sizeof(uint8_t),
            cudaMemcpyDeviceToHost
        ));
        return output;
    }

    /**
     * @brief Queries a dense host-resident record batch and returns aggregate counts.
     *
     * @p batch.sequence stores raw record payloads back-to-back without separators.
     * @p batch.records stores ordered, non-overlapping byte ranges into that dense buffer.
     * The filter injects alphabet separators between records internally, so cross-record
     * k-mers are never formed.
     *
     * Synchronises before returning.
     *
     * @param batch   Dense record batch to query.
     * @param stream  CUDA stream to use.
     * @return Aggregate query summary for the whole batch.
     */
    [[nodiscard]] Result<FastxQueryReport>
    query_record_batch(RecordBatchView batch, cuda::stream_ref stream = cudaStream_t{}) const {
        return query_record_batch_aggregate(batch, stream);
    }

    /**
     * @brief Queries a dense host-resident record batch and streams per-record results.
     *
     * The callback receives one @ref RecordQueryView per input record in source
     * order. The hit span remains valid only for the duration of the callback.
     *
     * Synchronises before returning.
     *
     * @param batch    Dense record batch to query.
     * @param consume  Per-record callback.
     * @param stream   CUDA stream to use.
     * @return Aggregate query summary for the whole batch.
     */
    template <RecordQueryConsumer Consumer>
    [[nodiscard]] Result<FastxQueryReport> query_record_batch(
        RecordBatchView batch,
        Consumer&& consume,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        CUSBF_TRY(
            normalize_record_batch_into<Config>(
                batch,
                fastx_state_.normalized_sequence_scratch(),
                fastx_state_.normalized_records_scratch()
            )
        );
        return query_normalized_record_batch_with_hits(
            batch.sequence, std::forward<Consumer>(consume), stream
        );
    }

    /**
     * @brief Async query of k-mers from a device-resident record batch.
     *
     * The device sequence is assumed to already contain alphabet separator bytes
     * between records: the caller is responsible for embedding them. @p records
     * provides ordered, non-overlapping byte ranges into @p d_sequence.
     *
     * @p d_output receives one byte per k-mer across all records (1 = present,
     * 0 = absent). The buffer must hold at least the total k-mer count.
     *
     * Does **not** synchronise the stream.
     *
     * @param d_sequence  Device-resident sequence (records concatenated with
     *                    separators already in place).
     * @param records     Ordered, non-overlapping record byte ranges.
     * @param d_output    Per-k-mer result buffer.
     * @param stream      CUDA stream to use.
     */
    [[nodiscard]] Result<void> contains_record_batch_async(
        device_span<const char> d_sequence,
        std::span<const RecordRange> records,
        device_span<uint8_t> d_output,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        CUSBF_TRY(validate_device_record_batch(d_sequence, records));
        const uint64_t totalKmers = sequence_kmer_count(d_sequence);
        if (totalKmers == 0) {
            return {};
        }
        if (d_output.size() < totalKmers) {
            return Err(Error::invalid_argument("record batch query output span is too small"));
        }
        return launch_contains_sequence(d_sequence, d_output, stream);
    }

    /**
     * @brief Queries all k-mers from a FASTA/FASTQ input stream via chunked streaming.
     *
     * Returns aggregate counts only (no per-record hit vectors). For per-record callbacks
     * use @ref query_fastx_records, for owning per-record hits use @ref query_fastx_detailed.
     *
     * @param input         Input stream containing FASTA or FASTQ records.
     * @param fill_fraction Fraction of free GPU memory for per-chunk staging (default 0.7).
     * @param stream        CUDA stream to use.
     * @return Aggregate query summary for the whole stream.
     * @see insert_fastx
     */
    [[nodiscard]] Result<FastxQueryReport> query_fastx(
        std::istream& input,
        double fill_fraction = 0.7,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        return query_fastx_stream(input, "<stream>", fill_fraction, stream);
    }

    /**
     * @brief Queries all k-mers from a FASTA/FASTQ file.
     *
     * Returns aggregate counts only. See @ref query_fastx_file_records and
     * @ref query_fastx_file_detailed for per-record results.
     *
     * @param path          Path to a FASTA or FASTQ file (optionally gzip-compressed).
     * @param fill_fraction Fraction of free GPU memory for per-chunk staging (default 0.7).
     * @param stream        CUDA stream to use.
     * @return Aggregate query summary for the whole file.
     * @see insert_fastx_file for dispatch and chunking behavior.
     */
    [[nodiscard]] Result<FastxQueryReport> query_fastx_file(
        const std::filesystem::path& path,
        double fill_fraction = 0.7,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        return detail::dispatch_fastx_file<Config>(
            path,
            detail::fastx_chunk_mode::query,
            fill_fraction,
            [&](auto& reader, auto dispatch_path) {
                return query_fastx_reader(
                    reader, std::string_view{path.native()}, fill_fraction, stream, dispatch_path
                );
            }
        );
    }

    /**
     * @brief Queries a FASTA/FASTQ stream and emits one record result per parsed record.
     *
     * The callback receives record headers, record sequences, aggregate counts, and the
     * per-window hit span for each record as soon as its chunk has been processed. The hit
     * span remains valid only for the duration of the callback.
     *
     * @param input         Input stream containing FASTA or FASTQ records.
     * @param consume       Per-record callback.
     * @param fill_fraction  Fraction of available GPU memory for per-chunk staging (default 0.7).
     *                       Query mode reserves space for both sequence and per-k-mer hit buffers.
     * @param stream        CUDA stream to use.
     * @return Aggregate query summary for the whole stream.
     */
    template <FastxRecordConsumer Consumer>
    [[nodiscard]] Result<FastxQueryReport> query_fastx_records(
        std::istream& input,
        Consumer&& consume,
        double fill_fraction = 0.7,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        detail::FastxReader reader(input, "<stream>");
        return query_fastx_records_stream(
            reader,
            "<stream>",
            consume,
            fill_fraction,
            stream,
            detail::fastx_dispatch_path::chunked_stream
        );
    }

    /**
     * @brief Queries a FASTA/FASTQ file and emits one record result per parsed record.
     *
     * @param path           Path to a FASTA or FASTQ file (optionally gzip-compressed).
     * @param consume        Per-record callback.
     * @param fill_fraction  Fraction of free GPU memory for per-chunk staging (default 0.7).
     * @param stream         CUDA stream to use.
     * @return Aggregate query summary for the whole file.
     * @see query_fastx_records
     */
    template <FastxRecordConsumer Consumer>
    [[nodiscard]] Result<FastxQueryReport> query_fastx_file_records(
        const std::filesystem::path& path,
        Consumer&& consume,
        double fill_fraction = 0.7,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        return detail::dispatch_fastx_file<Config>(
            path,
            detail::fastx_chunk_mode::query,
            fill_fraction,
            [&](auto& reader, auto dispatch_path) {
                return query_fastx_records_stream(
                    reader,
                    std::string_view{path.native()},
                    consume,
                    fill_fraction,
                    stream,
                    dispatch_path
                );
            }
        );
    }

    /**
     * @brief Queries all k-mers from a FASTA/FASTQ input stream via chunked streaming and
     * preserves per-record hit vectors.
     *
     * The returned report keeps aggregate counts plus one detailed record result in source
     * order. Each detailed hit vector contains one byte per k-mer window: 1 = present,
     * 0 = absent. Invalid-symbol windows remain in the vector as 0 and are excluded from
     * queriedKmers.
     *
     * @param input         Input stream containing FASTA or FASTQ records.
     * @param fill_fraction Fraction of free GPU memory for per-chunk staging (default 0.7).
     * @param stream        CUDA stream to use.
     * @return Aggregate and per-record query results.
     * @see query_fastx
     */
    [[nodiscard]] Result<FastxDetailedQueryReport> query_fastx_detailed(
        std::istream& input,
        double fill_fraction = 0.7,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        detail::FastxReader reader(input, "<stream>");
        return query_fastx_detailed_stream(
            reader, "<stream>", fill_fraction, stream, detail::fastx_dispatch_path::chunked_stream
        );
    }

    /**
     * @brief Queries all k-mers from a FASTA/FASTQ file via chunked streaming and
     * preserves per-record hit vectors.
     *
     * @param path           Path to a FASTA or FASTQ file (optionally gzip-compressed).
     * @param fill_fraction  Fraction of free GPU memory for per-chunk staging (default 0.7).
     * @param stream         CUDA stream to use.
     * @return Aggregate and per-record query results.
     * @see query_fastx_detailed
     */
    [[nodiscard]] Result<FastxDetailedQueryReport> query_fastx_file_detailed(
        const std::filesystem::path& path,
        double fill_fraction = 0.7,
        cuda::stream_ref stream = cudaStream_t{}
    ) const {
        return detail::dispatch_fastx_file<Config>(
            path,
            detail::fastx_chunk_mode::query,
            fill_fraction,
            [&](auto& reader, auto dispatch_path) {
                return query_fastx_detailed_stream(
                    reader, std::string_view{path.native()}, fill_fraction, stream, dispatch_path
                );
            }
        );
    }

    /**
     * @brief Resets all filter bits to zero and synchronises the stream.
     *
     * @param stream CUDA stream to use.
     */
    [[nodiscard]] Result<void> clear(cuda::stream_ref stream = cudaStream_t{}) {
        CUSBF_CUDA_TRY(cudaMemsetAsync(
            thrust::raw_pointer_cast(d_shards_.data()),
            0,
            d_shards_.size() * sizeof(block_type),
            stream.get()
        ));

        CUSBF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
        return {};
    }

    /**
     * @brief Computes the fraction of set bits in the filter.
     *
     * @return Load factor in [0, 1].
     */
    [[nodiscard]] float load_factor() const {
        const auto* wordsBegin =
            reinterpret_cast<const uint64_t*>(thrust::raw_pointer_cast(d_shards_.data()));
        const uint64_t totalWords = num_shards_ * Config::blockWordCount;
        const uint64_t setBits = thrust::transform_reduce(
            thrust::device,
            wordsBegin,
            wordsBegin + totalWords,
            [] __device__(uint64_t w) -> uint64_t { return cuda::std::popcount(w); },
            uint64_t{0},
            cuda::std::plus<uint64_t>()
        );
        return static_cast<float>(setBits) / static_cast<float>(filter_bits_);
    }

    /// @brief Returns the total allocated capacity of the filter in bits.
    [[nodiscard]] uint64_t filter_bits() const {
        return filter_bits_;
    }

    /// @brief Returns the number of shards.
    [[nodiscard]] uint64_t num_shards() const {
        return num_shards_;
    }

   private:
    uint64_t num_shards_{};
    uint64_t filter_bits_{};

    thrust::device_vector<block_type> d_shards_;
    mutable detail::FastxPipelineState fastx_state_;

    /**
     * @brief Validates record byte ranges against a device-resident sequence.
     *
     * Checks: ranges are ordered and non-overlapping, each range fits within
     * @p d_sequence, and each range is symbol-aligned.
     */
    [[nodiscard]] static Result<void> validate_device_record_batch(
        device_span<const char> d_sequence,
        std::span<const RecordRange> records
    ) {
        uint64_t next_offset = 0;
        for (const RecordRange& record : records) {
            if (record.sequenceOffset < next_offset) {
                return Err(
                    Error::invalid_argument(
                        "record batch ranges must be ordered and non-overlapping"
                    )
                );
            }
            if (record.sequenceOffset > d_sequence.size() ||
                record.sequenceBytes > d_sequence.size() - record.sequenceOffset) {
                return Err(Error::invalid_argument("record batch range exceeds device sequence"));
            }
            if (record.sequenceOffset % Config::symbolWidth != 0 ||
                record.sequenceBytes % Config::symbolWidth != 0) {
                return Err(
                    Error::invalid_argument(
                        "record batch ranges must align to the configured alphabet symbol width"
                    )
                );
            }
            next_offset = record.sequenceOffset + record.sequenceBytes;
        }
        return {};
    }

    static Result<void> normalize_record_batch_into_pinned(
        RecordBatchView batch,
        detail::FastxPinnedSequenceBuffer& sequence_out,
        std::vector<NormalizedRecord>& records_out
    ) {
        const uint64_t estimated_bytes = detail::estimate_normalized_batch_bytes<Config>(batch);
        CUSBF_TRY(sequence_out.reserve(static_cast<size_t>(estimated_bytes)));
        size_t sequence_out_bytes = 0;
        detail::normalize_record_batch_into_buffer<Config>(
            batch, sequence_out.data(), sequence_out_bytes, records_out
        );
        CUSBF_TRY(sequence_out.set_size(sequence_out_bytes));
        return {};
    }

    /// @brief Returns the total size of all shard storage in bytes.
    [[nodiscard]] uint64_t size_bytes() const {
        return num_shards() * sizeof(block_type);
    }

    [[nodiscard]] static constexpr uint64_t record_symbol_count(uint64_t bases) {
        return detail::record_symbol_count<Config>(bases);
    }

    [[nodiscard]] static constexpr uint64_t record_kmer_count(uint64_t bases) {
        return detail::record_kmer_count<Config>(bases);
    }

    static void accumulate_insert_report(FastxInsertReport& total, const FastxInsertReport& chunk) {
        total.recordsIndexed += chunk.recordsIndexed;
        total.indexedBases += chunk.indexedBases;
        total.insertedKmers += chunk.insertedKmers;
    }

    static void accumulate_query_report(FastxQueryReport& total, const FastxQueryReport& chunk) {
        total.recordsQueried += chunk.recordsQueried;
        total.queriedBases += chunk.queriedBases;
        total.queriedKmers += chunk.queriedKmers;
        total.positive_kmers += chunk.positive_kmers;
    }

    /**
     * @brief Queries a dense record batch and returns aggregate counts only.
     *
     * Normalizes @p batch internally, then runs a single GPU query without copying
     * per-k-mer hits back for callbacks.
     *
     * @param batch  Dense record batch to query.
     * @param stream CUDA stream to use.
     * @return Aggregate query summary for the whole batch.
     */
    [[nodiscard]] Result<FastxQueryReport>
    query_record_batch_aggregate(RecordBatchView batch, cuda::stream_ref stream) const {
        CUSBF_TRY(
            normalize_record_batch_into<Config>(
                batch,
                fastx_state_.normalized_sequence_scratch(),
                fastx_state_.normalized_records_scratch()
            )
        );
        return query_normalized_record_batch_aggregate(stream);
    }

    /**
     * @brief Queries @ref fastx_state_.normalized_sequence_scratch() and returns aggregate counts
     * only.
     *
     * Caller must populate normalized sequence and record metadata first (for example via
     * @ref normalize_record_batch_into).
     *
     * @param stream CUDA stream to use.
     * @return Aggregate query summary.
     */
    [[nodiscard]] Result<FastxQueryReport> query_normalized_record_batch_aggregate(
        cuda::stream_ref stream
    ) const {
        const detail::QueryLayout layout =
            detail::QueryLayout::build<Config>(fastx_state_.normalized_records_scratch());

        FastxQueryReport report;
        report.recordsQueried = layout.records().size();
        for (const detail::QueryLayoutRecord& record : layout.records()) {
            report.queriedBases += record.size;
            report.queriedKmers += record.valid_kmers;
        }
        if (fastx_state_.normalized_sequence_scratch().empty()) {
            return report;
        }

        const auto d_sequence = CUSBF_TRY(fastx_state_.staged_sequence_view(
            {fastx_state_.normalized_sequence_scratch().data(),
             fastx_state_.normalized_sequence_scratch().size()},
            stream
        ));
        const uint64_t num_kmers = layout.total_hit_count();
        fastx_state_.ensure_result_capacity(num_kmers);
        CUSBF_TRY(launch_contains_sequence(
            d_sequence,
            device_span<uint8_t>{
                thrust::raw_pointer_cast(fastx_state_.result_buffer_device().data()), num_kmers
            },
            stream
        ));
        report.positive_kmers = detail::count_positive_kmers_total<Config>(
            device_span<const uint8_t>{
                thrust::raw_pointer_cast(fastx_state_.result_buffer_device().data()), num_kmers
            },
            stream
        );
        CUSBF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
        fastx_state_.release_all();
        return report;
    }

    /**
     * @brief Queries normalized scratch buffers and invokes @p consume per record with hits.
     *
     * Uses @ref fastx_state_.normalized_sequence_scratch() and @ref
     * fastx_state_.normalized_records_scratch(). Each
     * @ref RecordQueryView::hits span points into host scratch valid only during @p consume.
     * @p input_sequence supplies original record bytes for @ref RecordQueryView::sequence
     * (typically the dense @ref RecordBatchView::sequence passed to @ref query_record_batch).
     *
     * @param input_sequence Dense source sequence for per-record sequence slices.
     * @param consume        Per-record callback.
     * @param stream         CUDA stream to use.
     * @return Aggregate query summary for the batch.
     */
    template <RecordQueryConsumer Consumer>
    [[nodiscard]] Result<FastxQueryReport> query_normalized_record_batch_with_hits(
        std::string_view input_sequence,
        Consumer&& consume,
        cuda::stream_ref stream
    ) const {
        const detail::QueryLayout layout =
            detail::QueryLayout::build<Config>(fastx_state_.normalized_records_scratch());

        FastxQueryReport report;
        report.recordsQueried = layout.records().size();
        for (const detail::QueryLayoutRecord& record : layout.records()) {
            report.queriedBases += record.size;
            report.queriedKmers += record.valid_kmers;
        }
        if (fastx_state_.normalized_sequence_scratch().empty()) {
            return report;
        }

        const auto d_sequence = CUSBF_TRY(fastx_state_.staged_sequence_view(
            {fastx_state_.normalized_sequence_scratch().data(),
             fastx_state_.normalized_sequence_scratch().size()},
            stream
        ));
        const uint64_t num_kmers = layout.total_hit_count();
        fastx_state_.ensure_result_capacity(num_kmers);
        CUSBF_TRY(launch_contains_sequence(
            d_sequence,
            device_span<uint8_t>{
                thrust::raw_pointer_cast(fastx_state_.result_buffer_device().data()), num_kmers
            },
            stream
        ));

        const uint64_t record_count = layout.records().size();
        if (record_count > fastx_state_.query_layout_records_device().size()) {
            fastx_state_.query_layout_records_device().resize(record_count);
        }
        if (record_count > fastx_state_.record_positive_kmers_device().size()) {
            fastx_state_.record_positive_kmers_device().resize(record_count);
        }
        CUSBF_CUDA_TRY(cudaMemcpyAsync(
            thrust::raw_pointer_cast(fastx_state_.query_layout_records_device().data()),
            layout.records().data(),
            record_count * sizeof(detail::QueryLayoutRecord),
            cudaMemcpyHostToDevice,
            stream.get()
        ));
        CUSBF_TRY(
            detail::count_positive_kmers_per_record<Config>(
                device_span<const uint8_t>{
                    thrust::raw_pointer_cast(fastx_state_.result_buffer_device().data()), num_kmers
                },
                device_span<const detail::QueryLayoutRecord>{
                    thrust::raw_pointer_cast(fastx_state_.query_layout_records_device().data()),
                    record_count
                },
                device_span<uint64_t>{
                    thrust::raw_pointer_cast(fastx_state_.record_positive_kmers_device().data()),
                    record_count
                },
                stream
            )
        );

        if (record_count > fastx_state_.record_positive_kmers_scratch().size()) {
            fastx_state_.record_positive_kmers_scratch().resize(record_count);
        }
        CUSBF_CUDA_TRY(cudaMemcpyAsync(
            fastx_state_.record_positive_kmers_scratch().data(),
            thrust::raw_pointer_cast(fastx_state_.record_positive_kmers_device().data()),
            record_count * sizeof(uint64_t),
            cudaMemcpyDeviceToHost,
            stream.get()
        ));

        fastx_state_.result_hits_scratch().resize(num_kmers);
        CUSBF_CUDA_TRY(cudaMemcpyAsync(
            fastx_state_.result_hits_scratch().data(),
            thrust::raw_pointer_cast(fastx_state_.result_buffer_device().data()),
            num_kmers * sizeof(uint8_t),
            cudaMemcpyDeviceToHost,
            stream.get()
        ));
        CUSBF_CUDA_TRY(cudaStreamSynchronize(stream.get()));

        const auto all_hits = std::span<const uint8_t>{
            fastx_state_.result_hits_scratch().data(), fastx_state_.result_hits_scratch().size()
        };
        for (size_t layout_index = 0; layout_index < layout.records().size(); ++layout_index) {
            const detail::QueryLayoutRecord& record = layout.records()[layout_index];
            const auto sequence = input_sequence.substr(
                static_cast<size_t>(record.input_offset), static_cast<size_t>(record.size)
            );
            if (record.hit_count == 0) {
                consume(
                    RecordQueryView{
                        record.record_index,
                        sequence,
                        record.size,
                        record.valid_kmers,
                        0,
                        std::span<const uint8_t>{},
                    }
                );
                continue;
            }

            const auto hit_span = layout.hits_for_record(all_hits, layout_index);
            const uint64_t positive_kmers =
                fastx_state_.record_positive_kmers_scratch()[layout_index];
            report.positive_kmers += positive_kmers;
            consume(
                RecordQueryView{
                    record.record_index,
                    sequence,
                    record.size,
                    record.valid_kmers,
                    positive_kmers,
                    hit_span,
                }
            );
        }
        fastx_state_.release_all();
        return report;
    }

    /// @brief Internal implementation shared by insert_fastx() and insert_fastx_file().
    [[nodiscard]] Result<FastxInsertReport> insert_fastx_stream(
        std::istream& input,
        std::string_view source_name,
        double fill_fraction,
        cuda::stream_ref stream
    ) {
        detail::FastxReader reader(input, source_name);
        return insert_fastx_reader(
            reader, source_name, fill_fraction, stream, detail::fastx_dispatch_path::chunked_stream
        );
    }

    template <typename FastxReaderType>
    [[nodiscard]] Result<FastxInsertReport> insert_fastx_reader(
        FastxReaderType& reader,
        std::string_view source_name,
        double fill_fraction,
        cuda::stream_ref stream,
        detail::fastx_dispatch_path dispatch_path
    ) {
        struct InsertAdapter {
            using report_type = FastxInsertReport;
            [[nodiscard]] detail::fastx_chunk_mode chunk_mode() const {
                return detail::fastx_chunk_mode::insert;
            }

            [[nodiscard]] bool supports_pipelined() const {
                return true;
            }

            filter& self;
            FastxInsertReport report{};

            void
            on_record_collected(detail::FastxRecord&, uint64_t, DenseRecordBatchBuilder&) const {}

            [[nodiscard]] Result<void>
            flush_sync(DenseRecordBatchBuilder& chunk, cuda::stream_ref stream) {
                if (chunk.empty()) {
                    return {};
                }
                accumulate_insert_report(
                    report, CUSBF_TRY(self.insert_record_batch(chunk.view(), stream))
                );
                chunk.clear_and_shrink();
                return {};
            }

            [[nodiscard]] Result<void> flush_pipelined(
                DenseRecordBatchBuilder& chunk,
                detail::ChunkStreamPair& chunk_streams,
                size_t& ping,
                bool& has_inflight
            ) {
                return self.flush_chunk_insert_pipelined(
                    chunk, report, chunk_streams, ping, has_inflight
                );
            }

            [[nodiscard]] Result<void>
            finish_pipelined(detail::ChunkStreamPair&, size_t&, bool&) const {
                return {};
            }

            [[nodiscard]] Result<FastxInsertReport> finish() {
                return report;
            }
        };

        return detail::run_fastx_pipeline<Config>(
            reader,
            source_name,
            fill_fraction,
            stream,
            dispatch_path,
            fastx_state_,
            InsertAdapter{*this}
        );
    }

    /// @brief Internal implementation shared by query_fastx() and query_fastx_file().
    [[nodiscard]] Result<FastxQueryReport> query_fastx_stream(
        std::istream& input,
        std::string_view source_name,
        double fill_fraction,
        cuda::stream_ref stream
    ) const {
        detail::FastxReader reader(input, source_name);
        return query_fastx_reader(
            reader, source_name, fill_fraction, stream, detail::fastx_dispatch_path::chunked_stream
        );
    }

    template <typename FastxReaderType>
    [[nodiscard]] Result<FastxQueryReport> query_fastx_reader(
        FastxReaderType& reader,
        std::string_view source_name,
        double fill_fraction,
        cuda::stream_ref stream,
        detail::fastx_dispatch_path dispatch_path
    ) const {
        struct QueryAdapter {
            using report_type = FastxQueryReport;
            [[nodiscard]] detail::fastx_chunk_mode chunk_mode() const {
                return detail::fastx_chunk_mode::query;
            }

            [[nodiscard]] bool supports_pipelined() const {
                return true;
            }

            const filter& self;
            FastxQueryReport report{};

            void
            on_record_collected(detail::FastxRecord&, uint64_t, DenseRecordBatchBuilder&) const {}

            [[nodiscard]] Result<void>
            flush_sync(DenseRecordBatchBuilder& chunk, cuda::stream_ref stream) {
                if (chunk.empty()) {
                    return {};
                }
                accumulate_query_report(
                    report, CUSBF_TRY(self.query_record_batch_aggregate(chunk.view(), stream))
                );
                chunk.clear_and_shrink();
                return {};
            }

            [[nodiscard]] Result<void> flush_pipelined(
                DenseRecordBatchBuilder& chunk,
                detail::ChunkStreamPair& chunk_streams,
                size_t& ping,
                bool& has_inflight
            ) {
                return self.flush_chunk_query_aggregate_pipelined(
                    chunk, report, chunk_streams, ping, has_inflight
                );
            }

            [[nodiscard]] Result<void>
            finish_pipelined(detail::ChunkStreamPair&, size_t&, bool&) const {
                return {};
            }

            [[nodiscard]] Result<FastxQueryReport> finish() {
                return report;
            }
        };

        return detail::run_fastx_pipeline<Config>(
            reader,
            source_name,
            fill_fraction,
            stream,
            dispatch_path,
            fastx_state_,
            QueryAdapter{*this}
        );
    }

    void accumulate_normalized_insert_report(FastxInsertReport& report, size_t ping_slot) const {
        const std::vector<NormalizedRecord>& records =
            fastx_state_.normalized_records_ping(ping_slot);
        report.recordsIndexed += records.size();
        for (const NormalizedRecord& record : records) {
            report.indexedBases += record.size;
            report.insertedKmers += record.valid_kmers;
        }
    }

    void accumulate_normalized_query_report(FastxQueryReport& report, size_t ping_slot) const {
        const std::vector<NormalizedRecord>& records =
            fastx_state_.normalized_records_ping(ping_slot);
        report.recordsQueried += records.size();
        for (const NormalizedRecord& record : records) {
            report.queriedBases += record.size;
            report.queriedKmers += record.valid_kmers;
        }
    }

    Result<void> flush_chunk_insert_pipelined(
        DenseRecordBatchBuilder& chunk,
        FastxInsertReport& report,
        detail::ChunkStreamPair& chunk_streams,
        size_t& ping,
        bool& has_inflight
    ) {
        if (chunk.empty()) {
            return {};
        }

        if (has_inflight) {
            CUSBF_CUDA_TRY(cudaStreamSynchronize(chunk_streams[(ping - 1U) & 1U].get()));
        }

        const size_t slot = ping & 1U;
        const cuda::stream_ref active_stream = chunk_streams[slot];

        CUSBF_TRY(normalize_record_batch_into_pinned(
            chunk.view(),
            fastx_state_.normalized_sequence_ping(slot),
            fastx_state_.normalized_records_ping(slot)
        ));
        chunk.clear();
        accumulate_normalized_insert_report(report, slot);

        if (fastx_state_.normalized_sequence_ping(slot).size() == 0) {
            return {};
        }

        const auto d_sequence = CUSBF_TRY(fastx_state_.stage_sequence_ping(
            ping, fastx_state_.normalized_sequence_ping(slot).view(), active_stream
        ));
        CUSBF_TRY(launch_insert_sequence(d_sequence, active_stream));
        has_inflight = true;
        ping += 1;
        return {};
    }

    Result<void> flush_chunk_query_aggregate_pipelined(
        DenseRecordBatchBuilder& chunk,
        FastxQueryReport& report,
        detail::ChunkStreamPair& chunk_streams,
        size_t& ping,
        bool& has_inflight
    ) const {
        if (chunk.empty()) {
            return {};
        }

        if (has_inflight) {
            CUSBF_CUDA_TRY(cudaStreamSynchronize(chunk_streams[(ping - 1U) & 1U].get()));
        }

        const size_t slot = ping & 1U;
        const cuda::stream_ref active_stream = chunk_streams[slot];

        CUSBF_TRY(normalize_record_batch_into_pinned(
            chunk.view(),
            fastx_state_.normalized_sequence_ping(slot),
            fastx_state_.normalized_records_ping(slot)
        ));
        chunk.clear();
        accumulate_normalized_query_report(report, slot);

        if (fastx_state_.normalized_sequence_ping(slot).size() == 0) {
            return {};
        }

        const auto d_sequence = CUSBF_TRY(fastx_state_.stage_sequence_ping(
            ping, fastx_state_.normalized_sequence_ping(slot).view(), active_stream
        ));
        const uint64_t num_kmers = sequence_kmer_count(d_sequence);
        fastx_state_.ensure_result_capacity(num_kmers);
        CUSBF_TRY(launch_contains_sequence(
            d_sequence,
            device_span<uint8_t>{
                thrust::raw_pointer_cast(fastx_state_.result_buffer_device().data()), num_kmers
            },
            active_stream
        ));
        const uint64_t positive_kmers = detail::count_positive_kmers_total<Config>(
            device_span<const uint8_t>{
                thrust::raw_pointer_cast(fastx_state_.result_buffer_device().data()), num_kmers
            },
            active_stream
        );
        report.positive_kmers += positive_kmers;
        has_inflight = true;
        ping += 1;
        return {};
    }

    template <typename FastxReaderType, FastxRecordConsumer Consumer>
    [[nodiscard]] Result<FastxQueryReport> query_fastx_records_stream(
        FastxReaderType& reader,
        std::string_view source_name,
        Consumer&& consume,
        double fill_fraction,
        cuda::stream_ref stream,
        detail::fastx_dispatch_path dispatch_path
    ) const {
        struct DetailedQueryAdapter {
            using report_type = FastxQueryReport;

            struct PendingChunk {
                std::string sequence{};
                std::vector<RecordRange> ranges{};
                std::vector<FastxRecordHeaderRef> record_headers{};
                detail::QueryLayout layout{};
                std::vector<uint64_t> positive_kmers{};
                std::vector<uint8_t> hits{};
                uint64_t record_index_base{};
                bool ready{};

                void clear() {
                    sequence.clear();
                    ranges.clear();
                    record_headers.clear();
                    layout = detail::QueryLayout{};
                    positive_kmers.clear();
                    hits.clear();
                    record_index_base = 0;
                    ready = false;
                }
            };

            [[nodiscard]] detail::fastx_chunk_mode chunk_mode() const {
                return detail::fastx_chunk_mode::query;
            }

            [[nodiscard]] bool supports_pipelined() const {
                return true;
            }

            const filter& self;
            Consumer& consume;
            FastxQueryReport report{};
            std::vector<FastxRecordHeaderRef> record_headers{};
            uint64_t record_index_base{0};
            PendingChunk pending_chunks[2]{};

            DetailedQueryAdapter(const filter& self_in, Consumer& consume_in)
                : self(self_in), consume(consume_in), pending_chunks{} {}

            void on_record_collected(
                detail::FastxRecord& record,
                uint64_t local_index,
                DenseRecordBatchBuilder&
            ) {
                record_headers.push_back(
                    FastxRecordHeaderRef{std::move(record.header), local_index}
                );
            }

            [[nodiscard]] Result<void> retire_pending_chunk(size_t slot) {
                PendingChunk& pending = pending_chunks[slot];
                if (!pending.ready) {
                    return {};
                }

                const auto all_hits =
                    std::span<const uint8_t>{pending.hits.data(), pending.hits.size()};
                for (size_t layout_index = 0; layout_index < pending.layout.records().size();
                     ++layout_index) {
                    const detail::QueryLayoutRecord& record =
                        pending.layout.records()[layout_index];
                    const RecordRange& range =
                        pending.ranges[static_cast<size_t>(record.record_index)];
                    const FastxRecordHeaderRef& record_header =
                        pending.record_headers[static_cast<size_t>(record.record_index)];
                    const auto sequence = std::string_view{pending.sequence}.substr(
                        static_cast<size_t>(range.sequenceOffset),
                        static_cast<size_t>(range.sequenceBytes)
                    );
                    const auto hit_span =
                        record.hit_count == 0
                            ? std::span<const uint8_t>{}
                            : pending.layout.hits_for_record(all_hits, layout_index);
                    const uint64_t positive_kmers = pending.positive_kmers[layout_index];
                    report.positive_kmers += positive_kmers;
                    consume(
                        FastxRecordView{
                            pending.record_index_base + record_header.record_index,
                            record_header.header,
                            sequence,
                            record.size,
                            record.valid_kmers,
                            positive_kmers,
                            hit_span,
                        }
                    );
                }

                pending.clear();
                return {};
            }

            [[nodiscard]] Result<void>
            flush_sync(DenseRecordBatchBuilder& chunk, cuda::stream_ref stream) {
                if (chunk.empty()) {
                    return {};
                }
                const FastxQueryReport chunk_report = CUSBF_TRY(self.query_record_batch(
                    chunk.view(),
                    [&](const RecordQueryView& record_view) {
                        const FastxRecordHeaderRef& record_header =
                            record_headers[static_cast<size_t>(record_view.record_index)];
                        const RecordRange& range =
                            chunk.ranges()[static_cast<size_t>(record_view.record_index)];
                        consume(
                            FastxRecordView{
                                record_index_base + record_header.record_index,
                                record_header.header,
                                chunk.sequence_view().substr(
                                    static_cast<size_t>(range.sequenceOffset),
                                    static_cast<size_t>(range.sequenceBytes)
                                ),
                                record_view.queriedBases,
                                record_view.queriedKmers,
                                record_view.positive_kmers,
                                record_view.hits,
                            }
                        );
                    },
                    stream
                ));
                accumulate_query_report(report, chunk_report);
                record_index_base += chunk.recordCount();
                chunk.clear();
                record_headers.clear();
                return {};
            }

            [[nodiscard]] Result<void> flush_pipelined(
                DenseRecordBatchBuilder& chunk,
                detail::ChunkStreamPair& chunk_streams,
                size_t& ping,
                bool& has_inflight
            ) {
                if (chunk.empty()) {
                    return {};
                }

                if (has_inflight) {
                    const size_t completed_slot = (ping - 1U) & 1U;
                    CUSBF_CUDA_TRY(cudaStreamSynchronize(chunk_streams[completed_slot].get()));
                    CUSBF_TRY(retire_pending_chunk(completed_slot));
                    has_inflight = false;
                }

                const size_t slot = ping & 1U;
                const cuda::stream_ref active_stream = chunk_streams[slot];
                PendingChunk& pending = pending_chunks[slot];
                pending.clear();
                pending.sequence.assign(chunk.sequence_view());
                pending.ranges.assign(chunk.ranges().begin(), chunk.ranges().end());
                pending.record_headers = std::move(record_headers);
                pending.record_index_base = record_index_base;

                CUSBF_TRY(self.normalize_record_batch_into_pinned(
                    chunk.view(),
                    self.fastx_state_.normalized_sequence_ping(slot),
                    self.fastx_state_.normalized_records_ping(slot)
                ));
                self.accumulate_normalized_query_report(report, slot);
                pending.layout = detail::QueryLayout::build<Config>(
                    self.fastx_state_.normalized_records_ping(slot)
                );

                const uint64_t chunk_records = chunk.recordCount();
                chunk.clear();
                record_index_base += chunk_records;

                if (self.fastx_state_.normalized_sequence_ping(slot).size() == 0) {
                    pending.ready = true;
                    CUSBF_TRY(retire_pending_chunk(slot));
                    return {};
                }

                const auto d_sequence = CUSBF_TRY(self.fastx_state_.stage_sequence_ping(
                    ping, self.fastx_state_.normalized_sequence_ping(slot).view(), active_stream
                ));
                const uint64_t num_kmers = pending.layout.total_hit_count();
                self.fastx_state_.ensure_result_capacity(num_kmers);
                CUSBF_TRY(self.launch_contains_sequence(
                    d_sequence,
                    device_span<uint8_t>{
                        thrust::raw_pointer_cast(self.fastx_state_.result_buffer_device().data()),
                        num_kmers
                    },
                    active_stream
                ));

                const uint64_t record_count = pending.layout.records().size();
                if (record_count > self.fastx_state_.query_layout_records_device().size()) {
                    self.fastx_state_.query_layout_records_device().resize(record_count);
                }
                if (record_count > self.fastx_state_.record_positive_kmers_device().size()) {
                    self.fastx_state_.record_positive_kmers_device().resize(record_count);
                }
                CUSBF_CUDA_TRY(cudaMemcpyAsync(
                    thrust::raw_pointer_cast(
                        self.fastx_state_.query_layout_records_device().data()
                    ),
                    pending.layout.records().data(),
                    record_count * sizeof(detail::QueryLayoutRecord),
                    cudaMemcpyHostToDevice,
                    active_stream.get()
                ));
                CUSBF_TRY(
                    detail::count_positive_kmers_per_record<Config>(
                        device_span<const uint8_t>{
                            thrust::raw_pointer_cast(
                                self.fastx_state_.result_buffer_device().data()
                            ),
                            num_kmers
                        },
                        device_span<const detail::QueryLayoutRecord>{
                            thrust::raw_pointer_cast(
                                self.fastx_state_.query_layout_records_device().data()
                            ),
                            record_count
                        },
                        device_span<uint64_t>{
                            thrust::raw_pointer_cast(
                                self.fastx_state_.record_positive_kmers_device().data()
                            ),
                            record_count
                        },
                        active_stream
                    )
                );

                pending.positive_kmers.resize(record_count);
                CUSBF_CUDA_TRY(cudaMemcpyAsync(
                    pending.positive_kmers.data(),
                    thrust::raw_pointer_cast(
                        self.fastx_state_.record_positive_kmers_device().data()
                    ),
                    record_count * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost,
                    active_stream.get()
                ));

                pending.hits.resize(num_kmers);
                CUSBF_CUDA_TRY(cudaMemcpyAsync(
                    pending.hits.data(),
                    thrust::raw_pointer_cast(self.fastx_state_.result_buffer_device().data()),
                    num_kmers * sizeof(uint8_t),
                    cudaMemcpyDeviceToHost,
                    active_stream.get()
                ));

                pending.ready = true;
                has_inflight = true;
                ping += 1;
                return {};
            }

            [[nodiscard]] Result<void>
            finish_pipelined(detail::ChunkStreamPair&, size_t& ping, bool& has_inflight) {
                if (!has_inflight) {
                    return {};
                }

                const size_t completed_slot = (ping - 1U) & 1U;
                CUSBF_TRY(retire_pending_chunk(completed_slot));
                has_inflight = false;
                return {};
            }

            [[nodiscard]] Result<FastxQueryReport> finish() {
                return report;
            }
        };

        return detail::run_fastx_pipeline<Config>(
            reader,
            source_name,
            fill_fraction,
            stream,
            dispatch_path,
            fastx_state_,
            DetailedQueryAdapter{*this, consume}
        );
    }

    /// @brief Internal implementation shared by query_fastx_detailed() and
    /// query_fastx_file_detailed().
    template <typename FastxReaderType>
    [[nodiscard]] Result<FastxDetailedQueryReport> query_fastx_detailed_stream(
        FastxReaderType& reader,
        std::string_view source_name,
        double fill_fraction,
        cuda::stream_ref stream,
        detail::fastx_dispatch_path dispatch_path
    ) const {
        FastxDetailedQueryReport report;
        report.summary = CUSBF_TRY(query_fastx_records_stream(
            reader,
            source_name,
            [&report](const FastxRecordView& record) {
                report.records.push_back(
                    FastxDetailedQueryRecord{
                        record.record_index,
                        std::string(record.header),
                        std::string(record.sequence),
                        record.queriedBases,
                        record.queriedKmers,
                        record.positive_kmers,
                        std::vector<uint8_t>(record.hits.begin(), record.hits.end()),
                    }
                );
            },
            fill_fraction,
            stream,
            dispatch_path
        ));
        return report;
    }

    /// @brief Number of k-mer windows in a device-resident encoded sequence.
    [[nodiscard]] static uint64_t sequence_kmer_count(device_span<const char> d_sequence) {
        return detail::SequenceKmerInput<Config>{d_sequence}.kmerCount();
    }

    Result<void> launch_insert_dense_packed(
        device_span<const uint64_t> d_words,
        uint64_t num_symbols,
        cuda::stream_ref stream
    ) {
        const detail::DensePackedKmerInput<Config> input{d_words, num_symbols};
        const uint64_t numKmers = input.kmerCount();
        if (numKmers == 0) {
            return {};
        }
        if (d_words.size() < dense_packed_word_count(num_symbols)) {
            return Err(Error::invalid_argument("dense packed span is too small for num_symbols"));
        }

        const uint64_t gridSize = cuda::ceil_div(numKmers, Config::cudaBlockSize);
        detail::insert_dense_packed_kmers_kernel<Config>
            <<<gridSize, Config::cudaBlockSize, 0, stream.get()>>>(
                input.words.data(),
                numKmers,
                thrust::raw_pointer_cast(d_shards_.data()),
                num_shards_
            );
        CUSBF_CUDA_TRY(cudaGetLastError());
        return {};
    }

    Result<void> launch_contains_dense_packed(
        device_span<const uint64_t> d_words,
        uint64_t num_symbols,
        device_span<uint8_t> d_output,
        cuda::stream_ref stream
    ) const {
        const detail::DensePackedKmerInput<Config> input{d_words, num_symbols};
        const uint64_t numKmers = input.kmerCount();
        if (numKmers == 0) {
            return {};
        }
        if (d_words.size() < dense_packed_word_count(num_symbols)) {
            return Err(Error::invalid_argument("dense packed span is too small for num_symbols"));
        }
        if (d_output.size() < numKmers) {
            return Err(Error::invalid_argument("dense packed query output span is too small"));
        }

        const uint64_t gridSize =
            cuda::ceil_div(numKmers, Config::cudaBlockSize * detail::kContainsSequenceStride);
        detail::contains_dense_packed_kmers_kernel<Config>
            <<<gridSize, Config::cudaBlockSize, 0, stream.get()>>>(
                input.words.data(),
                numKmers,
                thrust::raw_pointer_cast(d_shards_.data()),
                num_shards_,
                d_output.data()
            );
        CUSBF_CUDA_TRY(cudaGetLastError());
        return {};
    }

    /**
     * @brief Launches the insert kernel for a device-resident sequence.
     * @param d_sequence Device-resident sequence.
     * @param stream     CUDA stream.
     */
    Result<void>
    launch_insert_sequence(device_span<const char> d_sequence, cuda::stream_ref stream) {
        const auto input = detail::SequenceKmerInput<Config>{d_sequence};
        const uint64_t numKmers = input.kmerCount();
        if (numKmers == 0) {
            return {};
        }
        const uint64_t gridSize = cuda::ceil_div(numKmers, Config::cudaBlockSize);

        detail::insert_sequence_kmers_kernel<Config>
            <<<gridSize, Config::cudaBlockSize, 0, stream.get()>>>(
                input.sequence.data(),
                numKmers,
                thrust::raw_pointer_cast(d_shards_.data()),
                num_shards_
            );
        CUSBF_CUDA_TRY(cudaGetLastError());
        return {};
    }

    /**
     * @brief Launches the query kernel for a device-resident sequence.
     * @param d_sequence Device-resident sequence.
     * @param d_output   Per-k-mer result buffer (one byte per k-mer).
     * @param stream     CUDA stream.
     */
    Result<void> launch_contains_sequence(
        device_span<const char> d_sequence,
        device_span<uint8_t> d_output,
        cuda::stream_ref stream
    ) const {
        const auto input = detail::SequenceKmerInput<Config>{d_sequence};
        const uint64_t numKmers = input.kmerCount();
        const uint64_t gridSize =
            cuda::ceil_div(numKmers, Config::cudaBlockSize * detail::kContainsSequenceStride);

        detail::contains_sequence_kmers_kernel<Config>
            <<<gridSize, Config::cudaBlockSize, 0, stream.get()>>>(
                input.sequence.data(),
                numKmers,
                thrust::raw_pointer_cast(d_shards_.data()),
                num_shards_,
                d_output.data()
            );
        CUSBF_CUDA_TRY(cudaGetLastError());
        return {};
    }
};

}  // namespace cusbf
