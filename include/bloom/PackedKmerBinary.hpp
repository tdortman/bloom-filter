#pragma once

#include <cstdint>
#include <fstream>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace bloom {

/**
 * @brief Binary file header for a packed k-mer file.
 *
 * Stores the k-mer length and the total number of k-mers that follow the
 * header in the file.
 */
struct PackedKmerBinaryHeader {
    uint64_t k{};
    uint64_t count{};
};

/**
 * @brief In-memory representation of a packed k-mer binary file.
 *
 * Each k-mer is a 2-bit packed @c uint64_t value (MSB = first base).
 */
struct PackedKmerBinaryFile {
    uint64_t k{};
    std::vector<uint64_t> kmers;
};

namespace detail {

/**
 * @brief Validates that @p k is a legal k-mer length (1-32).
 *
 * @param k          k-mer length to validate.
 * @param sourceName Source name for error messages.
 * @throws std::runtime_error if @p k is out of range.
 */
inline void validatePackedKmerBinaryK(uint64_t k, std::string_view sourceName) {
    if (k == 0 || k > 32) {
        throw std::runtime_error(
            std::string(sourceName) +
            ": invalid k-mer length in packed binary: " + std::to_string(k)
        );
    }
}

/**
 * @brief Reads exactly @p size bytes from @p input into @p buffer.
 *
 * @param input       Source stream.
 * @param buffer      Destination buffer.
 * @param size        Number of bytes to read.
 * @param sourceName  Source name used in error messages.
 * @param what        Description of the data being read (for error messages).
 * @throws std::runtime_error on read failure.
 */
inline void readExact(
    std::istream& input,
    void* buffer,
    std::streamsize size,
    std::string_view sourceName,
    std::string_view what
) {
    input.read(static_cast<char*>(buffer), size);
    if (!input) {
        throw std::runtime_error(
            std::string(sourceName) + ": failed to read " + std::string(what) +
            " from packed k-mer binary"
        );
    }
}

}  // namespace detail

/**
 * @brief Writes a packed k-mer binary header to @p output.
 *
 * @param output  Destination stream.
 * @param k       k-mer length.
 * @param count   Number of k-mers.
 * @throws std::runtime_error on write failure.
 */
inline void writePackedKmerBinaryHeader(std::ostream& output, uint64_t k, uint64_t count) {
    detail::validatePackedKmerBinaryK(k, "<stream>");
    const PackedKmerBinaryHeader header{k, count};
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!output) {
        throw std::runtime_error("failed to write packed k-mer binary header");
    }
}

/**
 * @brief Reads and validates a packed k-mer binary header from @p input.
 *
 * @param input       Source stream.
 * @param sourceName  Source name for error messages (default \"<stream>\").
 * @return The parsed header.
 * @throws std::runtime_error on read failure or invalid k-mer length.
 */
inline PackedKmerBinaryHeader
readPackedKmerBinaryHeader(std::istream& input, std::string_view sourceName = "<stream>") {
    PackedKmerBinaryHeader header;
    detail::readExact(input, &header, sizeof(header), sourceName, "packed binary header");
    detail::validatePackedKmerBinaryK(header.k, sourceName);
    return header;
}

/**
 * @brief Reads a complete packed k-mer binary file from @p input.
 *
 * @param input       Source stream.
 * @param sourceName  Source name for error messages (default \"<stream>\").
 * @return Parsed k-mer data.
 * @throws std::runtime_error on read failure or invalid content.
 */
inline PackedKmerBinaryFile
readPackedKmerBinary(std::istream& input, std::string_view sourceName = "<stream>") {
    const PackedKmerBinaryHeader header = readPackedKmerBinaryHeader(input, sourceName);
    if (header.count > (std::numeric_limits<size_t>::max() / sizeof(uint64_t))) {
        throw std::runtime_error(
            std::string(sourceName) + ": packed k-mer binary is too large to load"
        );
    }

    PackedKmerBinaryFile file;
    file.k = header.k;
    file.kmers.resize(static_cast<size_t>(header.count));
    if (header.count != 0) {
        detail::readExact(
            input,
            file.kmers.data(),
            static_cast<std::streamsize>(header.count * sizeof(uint64_t)),
            sourceName,
            "packed k-mers"
        );
    }
    return file;
}

/**
 * @brief Opens and reads a packed k-mer binary file at @p path.
 *
 * @param path  Path to the binary file.
 * @return Parsed k-mer data.
 * @throws std::runtime_error if the file cannot be opened or is invalid.
 */
inline PackedKmerBinaryFile readPackedKmerBinaryFile(std::string_view path) {
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open packed k-mer binary file: " + std::string(path));
    }
    return readPackedKmerBinary(input, path);
}

}  // namespace bloom
