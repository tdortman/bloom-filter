#include <unistd.h>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <cuda/std/span>

#include <bloom/device_span.cuh>
#include <bloom/PackedKmerBinary.hpp>

#include "test_support.hpp"

namespace {

template <uint64_t K>
bool packWindow(std::string_view sequence, uint64_t start, uint64_t& packed) {
    packed = 0;
    for (uint64_t offset = 0; offset < K; ++offset) {
        const uint8_t encoded =
            bloom::detail::encodeBase(static_cast<uint8_t>(sequence[start + offset]));
        if (encoded > 3) {
            return false;
        }
        packed = (packed << 2) | static_cast<uint64_t>(encoded);
    }
    return true;
}

template <uint64_t K>
uint64_t packKmer(std::string_view kmer) {
    uint64_t packed = 0;
    if (kmer.size() != K || !packWindow<K>(kmer, 0, packed)) {
        throw std::runtime_error("Failed to pack test k-mer");
    }
    return packed;
}

template <uint64_t K>
std::vector<uint64_t> packValidKmers(std::string_view sequence) {
    std::vector<uint64_t> kmers;
    if (sequence.size() < K) {
        return kmers;
    }

    kmers.reserve(sequence.size() - K + 1);
    for (uint64_t index = 0; index + K <= sequence.size(); ++index) {
        uint64_t packed = 0;
        if (packWindow<K>(sequence, index, packed)) {
            kmers.push_back(packed);
        }
    }
    return kmers;
}

}  // namespace

TEST_F(BloomFilterTest, PackedInsertSupportsPackedQuery) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGT";
    const auto kmers = packValidKmers<TestConfig::k>(sequence);

    ASSERT_EQ(filter.insertPackedKmers(kmers), kmers.size());
    const auto hits = filter.containsPackedKmers(kmers);

    ASSERT_EQ(hits.size(), kmers.size());
    EXPECT_TRUE(allOnes(hits));
}

TEST_F(BloomFilterTest, SequenceQueryMatchesHostAndDeviceResults) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGT";

    (void)filter.insertSequence(sequence);

    const auto hostHits = filter.containsSequence(sequence);
    ASSERT_EQ(hostHits.size(), sequence.size() - TestConfig::k + 1);
    EXPECT_TRUE(allOnes(hostHits));

    thrust::device_vector<char> d_sequence(sequence.begin(), sequence.end());
    thrust::device_vector<uint8_t> d_output(hostHits.size());

    filter.containsSequenceDevice(
        bloom::device_span<const char>{
            thrust::raw_pointer_cast(d_sequence.data()), d_sequence.size()
        },
        bloom::device_span<uint8_t>{thrust::raw_pointer_cast(d_output.data()), d_output.size()}
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint8_t> deviceHits(hostHits.size());
    thrust::copy(d_output.begin(), d_output.end(), deviceHits.begin());

    EXPECT_EQ(deviceHits, hostHits);
}

TEST_F(BloomFilterTest, PackedInsertIsIdempotent) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const auto kmers = packValidKmers<TestConfig::k>("ACGTACGTACGT");

    ASSERT_EQ(filter.insertPackedKmers(kmers), kmers.size());
    const float firstLoadFactor = filter.loadFactor();
    ASSERT_EQ(filter.insertPackedKmers(kmers), kmers.size());
    const float secondLoadFactor = filter.loadFactor();

    EXPECT_FLOAT_EQ(firstLoadFactor, secondLoadFactor);
}

TEST_F(BloomFilterTest, PackedInsertTailCountNotDivisibleByFourSupportsQueries) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACG";
    const auto kmers = packValidKmers<TestConfig::k>(sequence);

    ASSERT_FALSE(kmers.empty());
    ASSERT_NE(kmers.size() % TestConfig::blockWordCount, 0ULL);
    ASSERT_EQ(filter.insertPackedKmers(kmers), kmers.size());

    const auto packedHits = filter.containsPackedKmers(kmers);
    EXPECT_TRUE(allOnes(packedHits));
}

TEST_F(BloomFilterTest, PackedTailQueryMatchesHostAndDeviceResults) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACG";
    const auto kmers = packValidKmers<TestConfig::k>(sequence);
    ASSERT_NE(kmers.size() % TestConfig::blockWordCount, 0ULL);

    ASSERT_EQ(filter.insertPackedKmers(kmers), kmers.size());

    std::vector<uint64_t> query = kmers;
    query.push_back(kmers.front());
    query.push_back(kmers.back());
    ASSERT_NE(query.size() % TestConfig::blockWordCount, 0ULL);

    const auto hostHits = filter.containsPackedKmers(query);
    ASSERT_EQ(hostHits.size(), query.size());
    EXPECT_TRUE(allOnes(hostHits));

    thrust::device_vector<uint64_t> d_query(query.begin(), query.end());
    thrust::device_vector<uint8_t> d_output(hostHits.size());

    filter.containsPackedKmersDevice(
        bloom::device_span<const uint64_t>{
            thrust::raw_pointer_cast(d_query.data()), d_query.size()
        },
        bloom::device_span<uint8_t>{thrust::raw_pointer_cast(d_output.data()), d_output.size()}
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint8_t> deviceHits(hostHits.size());
    thrust::copy(d_output.begin(), d_output.end(), deviceHits.begin());

    EXPECT_EQ(deviceHits, hostHits);
    EXPECT_TRUE(allOnes(deviceHits));
}

TEST_F(BloomFilterTest, PackedBinaryStoresKAndLoadsKmers) {
    const auto file = makeTempBinaryFile();
    const auto kmers = packValidKmers<TestConfig::k>("ACGTACGTACGT");

    std::ofstream output(file.path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    bloom::writePackedKmerBinaryHeader(output, TestConfig::k, kmers.size());
    output.write(reinterpret_cast<const char*>(kmers.data()), kmers.size() * sizeof(uint64_t));
    output.close();

    const auto loaded = bloom::readPackedKmerBinaryFile(file.path);
    EXPECT_EQ(loaded.k, TestConfig::k);
    EXPECT_EQ(loaded.kmers, kmers);
}

TEST_F(BloomFilterTest, PackedBinaryThrowsOnTruncatedPayload) {
    const auto file = makeTempBinaryFile();
    const auto kmers = packValidKmers<TestConfig::k>("ACGTACGTACGT");

    std::ofstream output(file.path, std::ios::binary);
    ASSERT_TRUE(output.is_open());
    bloom::writePackedKmerBinaryHeader(output, TestConfig::k, kmers.size() + 1);
    output.write(reinterpret_cast<const char*>(kmers.data()), kmers.size() * sizeof(uint64_t));
    output.close();

    EXPECT_THROW((void)bloom::readPackedKmerBinaryFile(file.path), std::runtime_error);
}
