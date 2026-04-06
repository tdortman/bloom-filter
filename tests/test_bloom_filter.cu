#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
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

bool allOnes(const std::vector<uint8_t>& values) {
    return std::all_of(values.begin(), values.end(), [](uint8_t value) { return value == 1; });
}

TEST_F(BloomFilterTest, InsertAndQuerySameSequenceHasNoFalseNegatives) {
    bloom::Filter<TestConfig> filter(1 << 12, 9);

    const std::string sequence = "ACGTACGTACGTACGT";
    const uint64_t inserted = filter.insertSequence(sequence);
    const auto hits = filter.containsSequence(sequence);

    ASSERT_EQ(inserted, sequence.size() - TestConfig::k + 1);
    ASSERT_EQ(hits.size(), inserted);
    EXPECT_TRUE(allOnes(hits));
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

TEST_F(BloomFilterTest, ShortSequenceInsertAndQueryReturnEmpty) {
    bloom::Filter<TestConfig> filter(1 << 12, 8);

    const std::string shortSequence = "ACGT";
    const uint64_t inserted = filter.insertSequence(shortSequence);
    const auto hits = filter.containsSequence(shortSequence);

    EXPECT_EQ(inserted, 0);
    EXPECT_TRUE(hits.empty());
}

TEST_F(BloomFilterTest, ShortSequenceDeviceOutputBufferRemainsUnchanged) {
    bloom::Filter<TestConfig> filter(1 << 12, 8);

    const std::string shortSequence = "ACGT";
    uint8_t* d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_output, sizeof(uint8_t)), cudaSuccess);

    uint8_t sentinel = 0xAB;
    ASSERT_EQ(
        cudaMemcpy(d_output, &sentinel, sizeof(uint8_t), cudaMemcpyHostToDevice), cudaSuccess
    );

    filter.containsSequence(shortSequence.data(), shortSequence.size(), d_output);

    uint8_t after = 0;
    ASSERT_EQ(cudaMemcpy(&after, d_output, sizeof(uint8_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(after, sentinel);

    ASSERT_EQ(cudaFree(d_output), cudaSuccess);
}

TEST_F(BloomFilterTest, ClearResetsMembership) {
    bloom::Filter<TestConfig> filter(1 << 12, 9);

    const std::string sequence = "ACGTACGTACGTACGT";
    (void)filter.insertSequence(sequence);
    filter.clear();

    const auto hits = filter.containsSequence(sequence);
    EXPECT_TRUE(std::all_of(hits.begin(), hits.end(), [](uint8_t value) { return value == 0; }));
}

TEST_F(BloomFilterTest, DeviceOutputMatchesHostContainsResults) {
    bloom::Filter<TestConfig> filter(1 << 13, 8);

    const std::string insertedSequence = "ACGTACGTACGTACGTACGTACGT";
    const std::string querySequence = "TACGTACGTACGTACGTACGTACG";
    (void)filter.insertSequence(insertedSequence);

    const auto hostHits = filter.containsSequence(querySequence);
    ASSERT_FALSE(hostHits.empty());

    uint8_t* d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_output, hostHits.size() * sizeof(uint8_t)), cudaSuccess);

    filter.containsSequence(querySequence.data(), querySequence.size(), d_output);

    std::vector<uint8_t> deviceHits(hostHits.size());
    ASSERT_EQ(
        cudaMemcpy(
            deviceHits.data(), d_output, hostHits.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost
        ),
        cudaSuccess
    );

    EXPECT_EQ(deviceHits, hostHits);
    ASSERT_EQ(cudaFree(d_output), cudaSuccess);
}

TEST_F(BloomFilterTest, MultipleInsertionsRemainQueryable) {
    bloom::Filter<TestConfig> filter(1 << 14, 9);

    const std::string sequenceA = "ACGTACGTACGTACGT";
    const std::string sequenceB = "TGCATGCATGCATGCA";

    (void)filter.insertSequence(sequenceA);
    (void)filter.insertSequence(sequenceB);

    const auto hitsA = filter.containsSequence(sequenceA);
    const auto hitsB = filter.containsSequence(sequenceB);

    EXPECT_TRUE(allOnes(hitsA));
    EXPECT_TRUE(allOnes(hitsB));
}

TEST_F(BloomFilterTest, LowercaseInsertionMatchesUppercaseQuery) {
    bloom::Filter<TestConfig> filter(1 << 12, 8);

    const std::string lowerSequence = "acgtacgtacgtacgt";
    const std::string upperSequence = "ACGTACGTACGTACGT";

    (void)filter.insertSequence(lowerSequence);

    const auto upperHits = filter.containsSequence(upperSequence);
    const auto lowerHits = filter.containsSequence(lowerSequence);

    EXPECT_TRUE(allOnes(upperHits));
    EXPECT_TRUE(allOnes(lowerHits));
}

TEST_F(BloomFilterTest, ChunkingWithInvalidBasesMatchesAcrossChunkSizes) {
    const std::string sequence = "ACGTNACGTACGTNACGTACGTNACGTACGT";

    bloom::Filter<TestConfig> smallChunkFilter(1 << 13, 5);
    bloom::Filter<TestConfig> largeChunkFilter(1 << 13, sequence.size());

    (void)smallChunkFilter.insertSequence(sequence);
    (void)largeChunkFilter.insertSequence(sequence);

    const auto smallHits = smallChunkFilter.containsSequence(sequence);
    const auto largeHits = largeChunkFilter.containsSequence(sequence);

    EXPECT_EQ(smallHits, largeHits);
}
