#include <unistd.h>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "test_support.hpp"

TEST_F(BloomFilterTest, InsertFastxFileParsesWrappedFastaRecords) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGT";
    const auto file = writeTempFile(
        ">wrapped\n"
        "ACGT\n"
        "ACGT\n"
        "ACGT\n"
    );

    const auto report = filter.insertFastxFile(file.path);
    const auto hits = filter.containsSequence(sequence);

    EXPECT_EQ(report.recordsIndexed, 1);
    EXPECT_EQ(report.indexedBases, sequence.size());
    EXPECT_EQ(report.insertedKmers, sequence.size() - TestConfig::k + 1);
    EXPECT_TRUE(allOnes(hits));
}

TEST_F(BloomFilterTest, QueryFastxFileParsesWrappedFastqWithCrLf) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const std::string sequence = "ACGTACGTACGT";
    (void)filter.insertSequence(sequence);

    const auto file = writeTempFile(
        "@wrapped\r\n"
        "ACGTAC\r\n"
        "GTACGT\r\n"
        "+\r\n"
        "IIIIII\r\n"
        "IIIIII\r\n"
    );

    const auto report = filter.queryFastxFile(file.path);

    EXPECT_EQ(report.recordsQueried, 1);
    EXPECT_EQ(report.queriedBases, sequence.size());
    EXPECT_EQ(report.queriedKmers, sequence.size() - TestConfig::k + 1);
    EXPECT_EQ(report.positiveKmers, report.queriedKmers);
}

TEST_F(BloomFilterTest, QueryFastxFileDoesNotCreateCrossRecordKmers) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const std::string sequenceA = "ACGTACGT";
    const std::string sequenceB = "TGCATGCA";
    (void)filter.insertSequence(sequenceA);
    (void)filter.insertSequence(sequenceB);

    const auto file = writeTempFile(
        ">first\n"
        "ACGT\n"
        "ACGT\n"
        ">second\n"
        "TGCA\n"
        "TGCA\n"
    );

    const auto report = filter.queryFastxFile(file.path);

    EXPECT_EQ(report.recordsQueried, 2);
    EXPECT_EQ(report.queriedBases, sequenceA.size() + sequenceB.size());
    EXPECT_EQ(
        report.queriedKmers,
        (sequenceA.size() - TestConfig::k + 1) + (sequenceB.size() - TestConfig::k + 1)
    );
    EXPECT_EQ(report.positiveKmers, report.queriedKmers);
}

TEST_F(BloomFilterTest, FastxReportsOnlyValidKmers) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const auto file = writeTempFile(
        ">with-invalid\n"
        "ACGTNACGTACGTA\n"
    );

    const auto insertReport = filter.insertFastxFile(file.path);
    const auto queryReport = filter.queryFastxFile(file.path);

    EXPECT_EQ(insertReport.recordsIndexed, 1);
    EXPECT_EQ(insertReport.indexedBases, 14);
    EXPECT_EQ(insertReport.insertedKmers, 5);
    EXPECT_EQ(queryReport.recordsQueried, 1);
    EXPECT_EQ(queryReport.queriedBases, 14);
    EXPECT_EQ(queryReport.queriedKmers, 5);
    EXPECT_EQ(queryReport.positiveKmers, 5);
}

TEST_F(BloomFilterTest, MalformedFastqThrowsOnQualityLengthMismatch) {
    bloom::Filter<TestConfig> filter(1 << 12);

    const auto file = writeTempFile(
        "@broken\n"
        "ACGTACGT\n"
        "+\n"
        "IIIIIII\n"
    );

    EXPECT_THROW((void)filter.queryFastxFile(file.path), std::runtime_error);
}
