#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <vector>

#include "test_support.hpp"

namespace {

struct TempFile {
    std::string path;

    explicit TempFile(std::string pathValue) : path(std::move(pathValue)) {}

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

TempFile writeTempFile(std::string_view contents) {
    std::string pathTemplate = "/tmp/bloom-fastx-XXXXXX";
    std::vector<char> pathBuffer(pathTemplate.begin(), pathTemplate.end());
    pathBuffer.push_back('\0');

    const int fd = mkstemp(pathBuffer.data());
    if (fd == -1) {
        throw std::runtime_error("Failed to create temporary FASTX file");
    }
    close(fd);

    std::ofstream output(pathBuffer.data(), std::ios::binary);
    if (!output.is_open()) {
        std::remove(pathBuffer.data());
        throw std::runtime_error("Failed to open temporary FASTX file for writing");
    }
    output << contents;
    output.close();

    return TempFile{pathBuffer.data()};
}

}  // namespace

TEST_F(BloomFilterTest, InsertFastxFileParsesWrappedFastaRecords) {
    bloom::Filter<TestConfig> filter(1 << 12, 8);

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
    bloom::Filter<TestConfig> filter(1 << 12, 8);

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
    bloom::Filter<TestConfig> filter(1 << 12, 8);

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

TEST_F(BloomFilterTest, MalformedFastqThrowsOnQualityLengthMismatch) {
    bloom::Filter<TestConfig> filter(1 << 12, 8);

    const auto file = writeTempFile(
        "@broken\n"
        "ACGTACGT\n"
        "+\n"
        "IIIIIII\n"
    );

    EXPECT_THROW((void)filter.queryFastxFile(file.path), std::runtime_error);
}
