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

struct PackedKmerBinaryHeader {
    uint64_t k{};
    uint64_t count{};
};

struct PackedKmerBinaryFile {
    uint64_t k{};
    std::vector<uint64_t> kmers;
};

namespace detail {

inline void validatePackedKmerBinaryK(uint64_t k, std::string_view sourceName) {
    if (k == 0 || k > 32) {
        throw std::runtime_error(
            std::string(sourceName) +
            ": invalid k-mer length in packed binary: " + std::to_string(k)
        );
    }
}

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

inline void writePackedKmerBinaryHeader(std::ostream& output, uint64_t k, uint64_t count) {
    detail::validatePackedKmerBinaryK(k, "<stream>");
    const PackedKmerBinaryHeader header{k, count};
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!output) {
        throw std::runtime_error("failed to write packed k-mer binary header");
    }
}

inline PackedKmerBinaryHeader
readPackedKmerBinaryHeader(std::istream& input, std::string_view sourceName = "<stream>") {
    PackedKmerBinaryHeader header;
    detail::readExact(input, &header, sizeof(header), sourceName, "packed binary header");
    detail::validatePackedKmerBinaryK(header.k, sourceName);
    return header;
}

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

inline PackedKmerBinaryFile readPackedKmerBinaryFile(std::string_view path) {
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open packed k-mer binary file: " + std::string(path));
    }
    return readPackedKmerBinary(input, path);
}

}  // namespace bloom
