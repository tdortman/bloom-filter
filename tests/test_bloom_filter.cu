#include <thrust/device_vector.h>
#include <cuda/std/span>
#include <string>

#include <bloom/device_span.cuh>

#include "test_support.hpp"

TEST_F(BloomFilterTest, InsertAndQuerySameSequenceHasNoFalseNegatives) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGTACGT";
    const uint64_t inserted = filter.insertSequence(sequence);
    const auto hits = filter.containsSequence(sequence);

    ASSERT_EQ(inserted, sequence.size() - TestConfig::k + 1);
    ASSERT_EQ(hits.size(), inserted);
    EXPECT_TRUE(allOnes(hits));
}

TEST_F(BloomFilterTest, InvalidBasesResetForwardWindows) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTNACGTACGTA";
    const auto inserted = filter.insertSequence(sequence);
    const auto hits = filter.containsSequence(sequence);

    EXPECT_EQ(inserted, hits.size());
    const std::vector<uint8_t> expected = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    EXPECT_EQ(hits, expected);
}

TEST_F(BloomFilterTest, RepeatedInsertionIsIdempotent) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGTACGT";
    const auto firstInserted = filter.insertSequence(sequence);
    const float firstLoadFactor = filter.loadFactor();

    const auto secondInserted = filter.insertSequence(sequence);
    const float secondLoadFactor = filter.loadFactor();

    EXPECT_EQ(firstInserted, secondInserted);
    EXPECT_FLOAT_EQ(firstLoadFactor, secondLoadFactor);
}

TEST_F(BloomFilterTest, ShortSequenceInsertAndQueryReturnEmpty) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const std::string shortSequence = "ACGT";
    const uint64_t inserted = filter.insertSequence(shortSequence);
    const auto hits = filter.containsSequence(shortSequence);

    EXPECT_EQ(inserted, 0);
    EXPECT_TRUE(hits.empty());
}

TEST_F(BloomFilterTest, ShortSequenceDeviceOutputBufferRemainsUnchanged) {
    bloom::Filter<TestConfig> filter(1 << 12);

    thrust::device_vector<char> d_sequence({'A', 'C', 'G', 'T'});
    thrust::device_vector<uint8_t> d_output(1, uint8_t{0xAB});

    filter.containsSequenceDevice(
        bloom::device_span<const char>{
            thrust::raw_pointer_cast(d_sequence.data()), d_sequence.size()
        },
        bloom::device_span<uint8_t>{thrust::raw_pointer_cast(d_output.data()), d_output.size()}
    );

    uint8_t after = 0;
    ASSERT_EQ(
        cudaMemcpy(
            &after,
            thrust::raw_pointer_cast(d_output.data()),
            sizeof(uint8_t),
            cudaMemcpyDeviceToHost
        ),
        cudaSuccess
    );
    EXPECT_EQ(after, uint8_t{0xAB});
}

TEST_F(BloomFilterTest, ClearResetsMembership) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGTACGT";
    (void)filter.insertSequence(sequence);
    filter.clear();

    const auto hits = filter.containsSequence(sequence);
    EXPECT_TRUE(std::all_of(hits.begin(), hits.end(), [](uint8_t value) { return value == 0; }));
}

TEST_F(BloomFilterTest, DeviceOutputMatchesHostContainsResults) {
    bloom::Filter<TestConfig> filter(1 << 13);

    const std::string insertedSequence = "ACGTACGTACGTACGTACGTACGT";
    const std::string querySequence = "TACGTACGTACGTACGTACGTACG";
    (void)filter.insertSequence(insertedSequence);

    const auto hostHits = filter.containsSequence(querySequence);
    ASSERT_FALSE(hostHits.empty());

    thrust::device_vector<char> d_query(querySequence.begin(), querySequence.end());
    thrust::device_vector<uint8_t> d_output(hostHits.size());

    filter.containsSequenceDevice(
        bloom::device_span<const char>{thrust::raw_pointer_cast(d_query.data()), d_query.size()},
        bloom::device_span<uint8_t>{thrust::raw_pointer_cast(d_output.data()), d_output.size()}
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint8_t> deviceHits(hostHits.size());
    thrust::copy(d_output.begin(), d_output.end(), deviceHits.begin());

    EXPECT_EQ(deviceHits, hostHits);
}

TEST_F(BloomFilterTest, MultipleInsertionsRemainQueryable) {
    bloom::Filter<TestConfig> filter(1 << 14);

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
    bloom::Filter<TestConfig> filter(1 << 12);

    const std::string lowerSequence = "acgtacgtacgtacgt";
    const std::string upperSequence = "ACGTACGTACGTACGT";

    (void)filter.insertSequence(lowerSequence);

    const auto upperHits = filter.containsSequence(upperSequence);
    const auto lowerHits = filter.containsSequence(lowerSequence);

    EXPECT_TRUE(allOnes(upperHits));
    EXPECT_TRUE(allOnes(lowerHits));
}

TEST_F(BloomFilterTest, ProteinAlphabetInsertAndQuerySameSequenceHasNoFalseNegatives) {
    bloom::Filter<ProteinTestConfig> filter(1 << 12);

    const std::string sequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const uint64_t inserted = filter.insertSequence(sequence);
    const auto hits = filter.containsSequence(sequence);

    ASSERT_EQ(inserted, sequence.size() - ProteinTestConfig::k + 1);
    ASSERT_EQ(hits.size(), inserted);
    EXPECT_TRUE(allOnes(hits));
}

TEST_F(BloomFilterTest, ProteinAlphabetLowercaseMatchesUppercaseQuery) {
    bloom::Filter<ProteinTestConfig> filter(1 << 12);

    const std::string lowerSequence = "abcdefghijklmnopqrstuvwxyz";
    const std::string upperSequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    (void)filter.insertSequence(lowerSequence);

    EXPECT_TRUE(allOnes(filter.containsSequence(upperSequence)));
    EXPECT_TRUE(allOnes(filter.containsSequence(lowerSequence)));
}

TEST_F(BloomFilterTest, ProteinAlphabetInvalidSymbolsResetForwardWindows) {
    bloom::Filter<ProteinTestConfig> filter(1 << 12);

    const std::string sequence = "ACDE*ACDEFGHI";
    const auto inserted = filter.insertSequence(sequence);
    const auto hits = filter.containsSequence(sequence);

    EXPECT_EQ(inserted, hits.size());
    const std::vector<uint8_t> expected = {0, 0, 0, 0, 0, 1, 1, 1, 1};
    EXPECT_EQ(hits, expected);
}
