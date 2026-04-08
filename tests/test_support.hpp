#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

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
