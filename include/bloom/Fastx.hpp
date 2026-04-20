#pragma once

#include <cstdint>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace bloom {

/// @brief Summary statistics returned by Filter insert operations on FASTX input.
struct FastxInsertReport {
    uint64_t recordsIndexed{};
    uint64_t indexedBases{};
    uint64_t insertedKmers{};
};

/// @brief Summary statistics returned by Filter query operations on FASTX input.
struct FastxQueryReport {
    uint64_t recordsQueried{};
    uint64_t queriedBases{};
    uint64_t queriedKmers{};
    uint64_t positiveKmers{};
};

namespace detail {

/// @brief Detected file format for a FASTA/FASTQ stream.
enum class FastxFormat : uint8_t {
    unknown,
    fasta,
    fastq,
};

/// @brief A single sequence record extracted from a FASTA/FASTQ stream.
struct FastxRecord {
    std::string sequence;
};

/// @brief Removes a trailing '\r' from @p line if present (Windows line endings).
inline void trimTrailingCarriageReturn(std::string& line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
}

/**
 * @brief Streaming FASTA/FASTQ parser.
 *
 * Reads one record at a time via @ref nextRecord. Supports both FASTA and
 * FASTQ formats, auto-detected from the first header character. Mixed
 * formats within a single stream are rejected with an exception.
 */
class FastxReader {
   public:
    explicit FastxReader(std::istream& input, std::string_view sourceName = "<stream>")
        : input_(input), sourceName_(sourceName) {
    }

    [[nodiscard]] bool nextRecord(FastxRecord& record) {
        record.sequence.clear();

        const std::string_view header = readHeaderLine();
        if (header.empty()) {
            return false;
        }

        const char headerTag = header.front();

        if (format_ == FastxFormat::unknown) {
            if (headerTag == '>') {
                format_ = FastxFormat::fasta;
            } else if (headerTag == '@') {
                format_ = FastxFormat::fastq;
            } else {
                throwParseError("expected FASTA or FASTQ header");
            }
        }

        const char expectedHeader = format_ == FastxFormat::fasta ? '>' : '@';
        if (headerTag != expectedHeader) {
            throwParseError("mixed FASTA and FASTQ records are not supported");
        }

        if (format_ == FastxFormat::fasta) {
            readFastaSequence(record.sequence);
        } else {
            readFastqSequence(record.sequence);
        }
        return true;
    }

   private:
    std::istream& input_;
    std::string_view sourceName_;
    std::string pendingHeader_;
    std::string lineBuffer_;
    FastxFormat format_{FastxFormat::unknown};
    uint64_t lineNumber_{};

    [[noreturn]] void throwParseError(std::string_view message) const {
        throw std::runtime_error(
            std::string(sourceName_) + ":" + std::to_string(lineNumber_) + ": " +
            std::string(message)
        );
    }

    [[nodiscard]] std::string_view readHeaderLine() {
        if (!pendingHeader_.empty()) {
            lineBuffer_ = std::move(pendingHeader_);
            pendingHeader_.clear();
            return lineBuffer_;
        }

        while (std::getline(input_, lineBuffer_)) {
            ++lineNumber_;
            trimTrailingCarriageReturn(lineBuffer_);
            if (!lineBuffer_.empty()) {
                return lineBuffer_;
            }
        }

        if (input_.bad()) {
            throw std::runtime_error(
                std::string("Failed to read FASTA/FASTQ input from ") + std::string(sourceName_)
            );
        }
        return {};
    }

    void readFastaSequence(std::string& sequence) {
        while (std::getline(input_, lineBuffer_)) {
            ++lineNumber_;
            trimTrailingCarriageReturn(lineBuffer_);
            if (!lineBuffer_.empty() && lineBuffer_.front() == '>') {
                pendingHeader_ = std::move(lineBuffer_);
                return;
            }
            sequence += lineBuffer_;
        }

        if (input_.bad()) {
            throw std::runtime_error(
                std::string("Failed to read FASTA/FASTQ input from ") + std::string(sourceName_)
            );
        }
    }

    void readFastqSequence(std::string& sequence) {
        while (std::getline(input_, lineBuffer_)) {
            ++lineNumber_;
            trimTrailingCarriageReturn(lineBuffer_);
            if (!lineBuffer_.empty() && lineBuffer_.front() == '+') {
                readFastqQualities(sequence.size());
                return;
            }
            sequence += lineBuffer_;
        }

        if (input_.bad()) {
            throw std::runtime_error(
                std::string("Failed to read FASTA/FASTQ input from ") + std::string(sourceName_)
            );
        }
        throwParseError("unterminated FASTQ record: missing '+' separator");
    }

    void readFastqQualities(uint64_t expectedLength) {
        uint64_t qualityLength = 0;
        while (qualityLength < expectedLength && std::getline(input_, lineBuffer_)) {
            ++lineNumber_;
            trimTrailingCarriageReturn(lineBuffer_);
            qualityLength += lineBuffer_.size();
            if (qualityLength > expectedLength) {
                throwParseError("FASTQ quality length exceeds sequence length");
            }
        }

        if (qualityLength == expectedLength) {
            return;
        }
        if (input_.bad()) {
            throw std::runtime_error(
                std::string("Failed to read FASTA/FASTQ input from ") + std::string(sourceName_)
            );
        }
        throwParseError("FASTQ quality length does not match sequence length");
    }
};

/**
 * @brief Opens a FASTA/FASTQ file for reading.
 *
 * @param path  File path.
 * @return Open input file stream.
 * @throws std::runtime_error if the file cannot be opened.
 */
inline std::ifstream openFastxFile(std::string_view path) {
    std::ifstream input{std::string(path)};
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open FASTA/FASTQ file: " + std::string(path));
    }
    return input;
}

}  // namespace detail

}  // namespace bloom
