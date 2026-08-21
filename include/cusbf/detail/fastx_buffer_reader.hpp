#pragma once

#include <bit>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#if defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
#endif
#if defined(__ARM_FEATURE_SVE2)
    #include <arm_sve.h>
#elif defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#endif

#include <cusbf/error.hpp>
#include <cusbf/Fastx.hpp>

namespace cusbf::detail {

[[nodiscard]] inline size_t fastx_line_end_scalar(std::string_view data, size_t position) {
    while (position < data.size() && data[position] != '\n' && data[position] != '\r') {
        ++position;
    }
    return position;
}

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && !defined(__CUDACC__)
[[gnu::target("avx2")]] [[nodiscard]] inline size_t
fastx_line_end_avx2(std::string_view data, size_t position) {
    const __m256i newline = _mm256_set1_epi8('\n');
    const __m256i carriage_return = _mm256_set1_epi8('\r');
    while (data.size() - position >= 32) {
        const __m256i bytes =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data.data() + position));
        const auto mask = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_or_si256(
            _mm256_cmpeq_epi8(bytes, newline), _mm256_cmpeq_epi8(bytes, carriage_return)
        )));
        if (mask != 0) {
            return position + static_cast<size_t>(__builtin_ctz(mask));
        }
        position += 32;
    }
    return fastx_line_end_scalar(data, position);
}
#endif

#if defined(__ARM_FEATURE_SVE)
[[nodiscard]] inline size_t fastx_line_end_sve(std::string_view data, size_t position) {
    while (position < data.size()) {
        const svbool_t active = svwhilelt_b8(position, data.size());
        const svuint8_t bytes =
            svld1_u8(active, reinterpret_cast<const uint8_t*>(data.data() + position));
        const svbool_t matches =
            svorr_b_z(active, svcmpeq_n_u8(active, bytes, '\n'), svcmpeq_n_u8(active, bytes, '\r'));
        if (svptest_any(active, matches)) {
            return position + svcntp_b8(active, svbrkb_z(active, matches));
        }
        position += svcntb();
    }
    return position;
}
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
[[nodiscard]] inline size_t fastx_line_end_neon(std::string_view data, size_t position) {
    const uint8x16_t newline = vdupq_n_u8('\n');
    const uint8x16_t carriage_return = vdupq_n_u8('\r');
    while (data.size() - position >= 16) {
        const uint8x16_t bytes = vld1q_u8(reinterpret_cast<const uint8_t*>(data.data() + position));
        const uint8x16_t matches =
            vorrq_u8(vceqq_u8(bytes, newline), vceqq_u8(bytes, carriage_return));
        const uint64x2_t words = vreinterpretq_u64_u8(matches);
        const uint64_t low = vgetq_lane_u64(words, 0);
        if (low != 0) {
            return position + (std::countr_zero(low) >> 3);
        }
        const uint64_t high = vgetq_lane_u64(words, 1);
        if (high != 0) {
            return position + 8 + (std::countr_zero(high) >> 3);
        }
        position += 16;
    }
    return fastx_line_end_scalar(data, position);
}
#endif

using fastx_line_end_fn = decltype(&fastx_line_end_scalar);

[[nodiscard]] inline fastx_line_end_fn resolve_fastx_line_end() {
#if defined(__ARM_FEATURE_SVE)
    return fastx_line_end_sve;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return fastx_line_end_neon;
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && !defined(__CUDACC__)
    return __builtin_cpu_supports("avx2") ? fastx_line_end_avx2 : fastx_line_end_scalar;
#else
    return fastx_line_end_scalar;
#endif
}

static const fastx_line_end_fn line_end = resolve_fastx_line_end();

[[nodiscard]] inline size_t fastx_line_end(std::string_view data, size_t position) {
    return line_end(data, position);
}

/// @brief FASTA/FASTQ parser over a contiguous in-memory buffer.
class FastxBufferReader {
   public:
    /**
     * @brief Constructs a reader over a contiguous in-memory FASTA/FASTQ buffer.
     *
     * @param data         Entire file or chunk payload.
     * @param source_name  Label used in parse error messages.
     */
    explicit FastxBufferReader(std::string_view data, std::string_view source_name = "<buffer>")
        : data_(data), source_name_(source_name) {}

    /**
     * @brief Reads the next record into @p record.
     *
     * @param record Output record, cleared before fill.
     * @return @c false at end-of-buffer, @c true when a record was read, or an error.
     */
    [[nodiscard]] Result<bool> nextRecord(FastxRecord& record) {
        record.header.clear();
        record.sequence.clear();

        const auto header = readHeaderLine();
        if (!header) {
            return Err(header.error());
        }
        if (header->empty()) {
            return false;
        }

        const char header_tag = header->front();
        if (format_ == FastxFormat::unknown) {
            if (header_tag == '>') {
                format_ = FastxFormat::fasta;
            } else if (header_tag == '@') {
                format_ = FastxFormat::fastq;
            } else {
                return Err(
                    parseError("expected FASTA or FASTQ header", fastx_column_at(*header, 0))
                );
            }
        }

        const char expected_header = format_ == FastxFormat::fasta ? '>' : '@';
        if (header_tag != expected_header) {
            return Err(parseError(
                "mixed FASTA and FASTQ records are not supported", fastx_column_at(*header, 0)
            ));
        }

        record.header.assign(header->substr(1));
        if (format_ == FastxFormat::fasta) {
            CUSBF_TRY(readFastaSequence(record.sequence));
        } else {
            CUSBF_TRY(readFastqSequence(record.sequence));
        }
        return true;
    }

    /// @brief Entire mmap or owned buffer backing this reader.
    [[nodiscard]] std::string_view buffer() const noexcept {
        return data_;
    }

    /**
     * @brief Parses one record with optional zero-copy sequence views for single-line FASTA.
     *
     * When the sequence fits one mmap line, returns a @ref RecordRange into @p buffer instead
     * of appending to @p sequence. Otherwise appends sequence bytes to @p sequence and returns
     * an owned range offset.
     *
     * @param record   Output header (sequence may stay empty on zero-copy path).
     * @param sequence Growing buffer for multi-line or owned FASTA sequence data.
     * @param buffer   Set to the full mmap view on first zero-copy record.
     * @return Record byte range, @c std::nullopt at end-of-buffer, or an error.
     */
    [[nodiscard]] Result<std::optional<RecordRange>>
    appendNextRecord(FastxRecord& record, std::string& sequence, std::string_view& buffer) {
        record.header.clear();
        record.sequence.clear();

        const auto header = readHeaderLine();
        if (!header) {
            return Err(header.error());
        }
        if (header->empty()) {
            return std::nullopt;
        }

        const char header_tag = header->front();
        if (format_ == FastxFormat::unknown) {
            if (header_tag == '>') {
                format_ = FastxFormat::fasta;
            } else if (header_tag == '@') {
                format_ = FastxFormat::fastq;
            } else {
                return Err(
                    parseError("expected FASTA or FASTQ header", fastx_column_at(*header, 0))
                );
            }
        }

        const char expected_header = format_ == FastxFormat::fasta ? '>' : '@';
        if (header_tag != expected_header) {
            return Err(parseError(
                "mixed FASTA and FASTQ records are not supported", fastx_column_at(*header, 0)
            ));
        }

        record.header.assign(header->substr(1));
        if (format_ == FastxFormat::fasta) {
            const auto sequence_offset = static_cast<uint64_t>(position_);
            const std::string_view line = readLine();
            if (line.empty()) {
                return Err(parseError("FASTA record missing sequence", fastx_column_at(line, 0)));
            }
            if (!line.empty() && line.front() == '>') {
                return Err(parseError("FASTA record missing sequence", fastx_column_at(line, 0)));
            }

            if (position_ < data_.size() && data_[position_] != '>') {
                const auto owned_offset = static_cast<uint64_t>(sequence.size());
                sequence.append(line.data(), line.size());
                CUSBF_TRY(readFastaSequence(sequence));
                return RecordRange{
                    owned_offset,
                    static_cast<uint64_t>(sequence.size()) - owned_offset,
                };
            }

            if (buffer.empty()) {
                buffer = data_;
            }
            return RecordRange{sequence_offset, static_cast<uint64_t>(line.size())};
        }

        const auto sequence_offset = static_cast<uint64_t>(sequence.size());
        CUSBF_TRY(readFastqSequence(sequence, sequence.size()));
        return RecordRange{
            sequence_offset,
            static_cast<uint64_t>(sequence.size()) - sequence_offset,
        };
    }

   private:
    std::string_view data_;
    std::string_view source_name_;
    size_t position_{0};
    FastxFormat format_{FastxFormat::unknown};
    uint64_t line_number_{};

    [[nodiscard]] Error parseError(std::string_view message, uint32_t column) const {
        return Error::fastx_parse(
            SourceLocation::fastx(source_name_, static_cast<uint32_t>(line_number_), column),
            message
        );
    }

    [[nodiscard]] std::string_view readLine() {
        if (position_ >= data_.size()) {
            return {};
        }

        const size_t end = fastx_line_end(data_, position_);

        const std::string_view line = data_.substr(position_, end - position_);
        position_ = end;
        if (position_ < data_.size() && data_[position_] == '\r') {
            ++position_;
        }
        if (position_ < data_.size() && data_[position_] == '\n') {
            ++position_;
        }
        ++line_number_;
        return line;
    }

    [[nodiscard]] Result<std::string_view> readHeaderLine() {
        while (position_ < data_.size()) {
            const std::string_view line = readLine();
            if (!line.empty()) {
                return line;
            }
        }
        return std::string_view{};
    }

    [[nodiscard]] Result<void> readFastaSequence(std::string& sequence) {
        while (position_ < data_.size()) {
            if (data_[position_] == '>') {
                return {};
            }
            const std::string_view line = readLine();
            sequence.append(line.data(), line.size());
        }
        return {};
    }

    [[nodiscard]] Result<void>
    readFastqSequence(std::string& sequence, uint64_t sequence_offset = 0) {
        std::string_view last_line;
        while (position_ < data_.size()) {
            const std::string_view line = readLine();
            last_line = line;
            if (!line.empty() && line.front() == '+') {
                CUSBF_TRY(readFastqQualities(sequence.size() - sequence_offset));
                return {};
            }
            sequence.append(line.data(), line.size());
        }
        return Err(parseError(
            "unterminated FASTQ record: missing '+' separator",
            fastx_column_at(last_line, last_line.size() > 0 ? last_line.size() - 1 : 0)
        ));
    }

    [[nodiscard]] Result<void> readFastqQualities(uint64_t expected_length) {
        uint64_t quality_length = 0;
        std::string_view last_line;
        while (quality_length < expected_length && position_ < data_.size()) {
            const std::string_view line = readLine();
            last_line = line;
            quality_length += line.size();
            if (quality_length > expected_length) {
                return Err(parseError(
                    "FASTQ quality length exceeds sequence length",
                    fastx_quality_excess_column(quality_length, expected_length, line)
                ));
            }
        }
        if (quality_length != expected_length) {
            return Err(parseError(
                "FASTQ quality length does not match sequence length",
                fastx_quality_short_column(last_line)
            ));
        }
        return {};
    }
};

}  // namespace cusbf::detail
