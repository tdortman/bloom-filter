#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include <bloom/BloomFilter.cuh>

namespace {

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

TEST_F(BloomFilterTest, InsertAndQuerySameSequenceHasNoFalseNegatives) {
    bloom::Filter<TestConfig> filter(1 << 12, 9);

    const std::string sequence = "ACGTACGTACGTACGT";
    const size_t inserted = filter.insertSequence(sequence);
    const auto hits = filter.containsSequence(sequence);

    ASSERT_EQ(inserted, sequence.size() - TestConfig::k + 1);
    ASSERT_EQ(hits.size(), inserted);
    EXPECT_TRUE(std::all_of(hits.begin(), hits.end(), [](uint8_t value) { return value == 1; }));
}

TEST_F(BloomFilterTest, InvalidBasesResetForwardWindows) {
    bloom::Filter<TestConfig> filter(1 << 12, 8);

    const std::string sequence = "ACGTNACGTACGTA";
    const auto inserted = filter.insertSequence(sequence);
    const auto hits = filter.containsSequence(sequence);

    EXPECT_EQ(inserted, hits.size());
    const std::vector<uint8_t> expected = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    EXPECT_EQ(hits, expected);
}

TEST_F(BloomFilterTest, ChunkedInsertMatchesLargeChunkQueryResults) {
    const std::string sequence = "ACGTACGTACGTACGTACGTACGTACGTACGT";
    const std::string query = "TACGTACGTACGTACGTACGTACGTACGTACG";

    bloom::Filter<TestConfig> smallChunkFilter(1 << 13, 7);
    bloom::Filter<TestConfig> largeChunkFilter(1 << 13, sequence.size());

    const auto smallInserted = smallChunkFilter.insertSequence(sequence);
    const auto largeInserted = largeChunkFilter.insertSequence(sequence);

    const auto smallHits = smallChunkFilter.containsSequence(query);
    const auto largeHits = largeChunkFilter.containsSequence(query);

    EXPECT_EQ(smallInserted, largeInserted);
    EXPECT_EQ(smallHits, largeHits);
}

TEST_F(BloomFilterTest, RepeatedInsertionIsIdempotent) {
    bloom::Filter<TestConfig> filter(1 << 12, 9);

    const std::string sequence = "ACGTACGTACGTACGT";
    const auto firstInserted = filter.insertSequence(sequence);
    const float firstLoadFactor = filter.loadFactor();

    const auto secondInserted = filter.insertSequence(sequence);
    const float secondLoadFactor = filter.loadFactor();

    EXPECT_EQ(firstInserted, secondInserted);
    EXPECT_FLOAT_EQ(firstLoadFactor, secondLoadFactor);
}

}  // namespace
