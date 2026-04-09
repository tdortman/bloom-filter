#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <unistd.h>

#include <bloom/BloomFilter.cuh>

using TestConfig = bloom::Config<5, 3, 4, 3>;

class BloomFilterTest : public ::testing::Test {
   protected:
    void SetUp() override {
        int deviceCount = 0;
        const auto status = cudaGetDeviceCount(&deviceCount);
        if (status != cudaSuccess || deviceCount == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
    }
};

inline bool allOnes(const std::vector<uint8_t>& values) {
    return std::all_of(values.begin(), values.end(), [](uint8_t value) { return value == 1; });
}

struct TempFile {
    std::string path;

    explicit TempFile(std::string pathValue) : path(std::move(pathValue)) {
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    TempFile(TempFile&& other) noexcept : path(std::move(other.path)) {
        other.path.clear();
    }

    ~TempFile() {
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }
};

inline TempFile writeTempFile(std::string_view contents) {
    std::string pathTemplate = "/tmp/bloom-XXXXXX";
    std::vector<char> pathBuffer(pathTemplate.begin(), pathTemplate.end());
    pathBuffer.push_back('\0');

    const int fd = mkstemp(pathBuffer.data());
    if (fd == -1) {
        throw std::runtime_error("Failed to create temporary file");
    }
    close(fd);

    std::ofstream output(pathBuffer.data(), std::ios::binary);
    if (!output.is_open()) {
        std::remove(pathBuffer.data());
        throw std::runtime_error("Failed to open temporary file for writing");
    }
    output << contents;
    output.close();

    return TempFile{pathBuffer.data()};
}

inline TempFile makeTempBinaryFile() {
    std::string pathTemplate = "/tmp/bloom-packed-XXXXXX";
    std::vector<char> pathBuffer(pathTemplate.begin(), pathTemplate.end());
    pathBuffer.push_back('\0');

    const int fd = mkstemp(pathBuffer.data());
    if (fd == -1) {
        throw std::runtime_error("Failed to create temporary packed k-mer file");
    }
    close(fd);

    return TempFile{pathBuffer.data()};
}
