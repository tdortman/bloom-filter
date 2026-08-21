#include <gtest/gtest.h>

#include <string>

#include <cusbf/detail/fastx_buffer_reader.hpp>

TEST(FastxBufferReaderTest, FindsLineEndAcrossVectorBoundaries) {
    std::string data(96, 'A');
    data[31] = '\n';
    data[64] = '\r';

    EXPECT_EQ(cusbf::detail::fastx_line_end(data, 0), 31u);
    EXPECT_EQ(cusbf::detail::fastx_line_end(data, 32), 64u);
    EXPECT_EQ(cusbf::detail::fastx_line_end(data, 65), data.size());
}

TEST(FastxBufferReaderTest, ParsesWrappedFastqWithCrLf) {
    const std::string data =
        "@read\r\n"
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\r\n"
        "ACGT\r\n"
        "+\r\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\r\n"
        "IIII\r\n";
    cusbf::detail::FastxBufferReader reader(data);
    cusbf::detail::FastxRecord record;

    const auto result = reader.nextRecord(record);

    ASSERT_TRUE(result);
    ASSERT_TRUE(*result);
    EXPECT_EQ(record.header, "read");
    EXPECT_EQ(record.sequence, "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT");
    const auto end = reader.nextRecord(record);
    ASSERT_TRUE(end);
    EXPECT_FALSE(*end);
}

TEST(FastxBufferReaderTest, AppendsConsecutiveFastqRecords) {
    const std::string data =
        "@first\nACGT\n+\nIIII\n"
        "@second\nTGCA\n+\nJJJJ\n";
    cusbf::detail::FastxBufferReader reader(data);
    cusbf::detail::FastxRecord record;
    std::string sequences;
    std::string_view external_sequence;

    const auto first = reader.appendNextRecord(record, sequences, external_sequence);
    ASSERT_TRUE(first);
    ASSERT_TRUE(*first);
    EXPECT_EQ((**first).sequenceOffset, 0u);
    EXPECT_EQ((**first).sequenceBytes, 4u);

    const auto second = reader.appendNextRecord(record, sequences, external_sequence);
    ASSERT_TRUE(second);
    ASSERT_TRUE(*second);
    EXPECT_EQ((**second).sequenceOffset, 4u);
    EXPECT_EQ((**second).sequenceBytes, 4u);
    EXPECT_EQ(sequences, "ACGTTGCA");
}
