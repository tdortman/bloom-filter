#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cusbf/detail/fastx_sequence_scan.hpp>

namespace {

// A rolling-window consumer: counts valid bases and emits one canonical 2-bit k-mer per full window
// (reverse complement via the scalar loop for clarity).
struct window_consumer {
    explicit window_consumer(uint32_t k) : k_(k), mask_((1ULL << (2U * k_)) - 1ULL) {}

    void consume(char ch) {
        if (cusbf::detail::fastx_is_sequence_whitespace(ch)) {
            return;
        }
        auto const encoded = cusbf::DnaAlphabet::encode(&ch);
        if (encoded == cusbf::DnaAlphabet::invalidSymbol) {
            if (window_len_ < k_ && window_len_ > 0) {
                ++invalid_windows;
            }
            window_len_ = 0;
            window_ = 0;
            return;
        }
        ++bases;
        window_ = ((window_ << 2U) | encoded) & mask_;
        if (window_len_ < k_) {
            ++window_len_;
        }
        if (window_len_ == k_) {
            kmers.push_back(window_);
        }
    }

    uint32_t k_;
    uint64_t mask_;
    uint64_t window_ = 0;
    uint32_t window_len_ = 0;
    uint64_t bases = 0;
    uint64_t invalid_windows = 0;
    std::vector<uint64_t> kmers;
};

std::vector<uint64_t> scan_single_threaded(std::string_view data, uint32_t k) {
    auto const extents = cusbf::detail::fastx_fasta_extents(data);
    window_consumer consumer(k);
    for (auto const& extent : extents) {
        for (const char* q = extent.begin; q < extent.end; ++q) {
            consumer.consume(*q);
        }
    }
    return consumer.kmers;
}

std::vector<uint64_t> scan_parallel(std::string_view data, uint32_t k, uint32_t threads) {
    auto const extents = cusbf::detail::fastx_fasta_extents(data);
    auto const spans =
        cusbf::detail::fastx_split_sequence_spans(extents, k > 0 ? k - 1 : 0, threads);
    std::vector<std::vector<uint64_t>> partials(spans.size());
    std::vector<std::thread> workers;
    workers.reserve(spans.size());
    for (size_t t = 0; t < spans.size(); ++t) {
        workers.emplace_back([&, t] {
            window_consumer consumer(k);
            for (char ch : spans[t].prefix) {
                consumer.consume(ch);
            }
            for (auto const& segment : spans[t].segments) {
                for (char ch : segment) {
                    consumer.consume(ch);
                }
            }
            partials[t] = std::move(consumer.kmers);
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    std::vector<uint64_t> result;
    for (auto& partial : partials) {
        result.insert(result.end(), partial.begin(), partial.end());
    }
    return result;
}

}  // namespace

TEST(FastxSequenceScanTest, ExtentsExcludeHeaders) {
    std::string const data = ">one\nACGT\n>two\nTT\n>three\nGGGG\n";
    auto const extents = cusbf::detail::fastx_fasta_extents(data);
    ASSERT_EQ(extents.size(), 3u);
    EXPECT_EQ(std::string_view(extents[0].begin, extents[0].end), std::string_view("ACGT\n"));
    EXPECT_EQ(std::string_view(extents[1].begin, extents[1].end), std::string_view("TT\n"));
    EXPECT_EQ(std::string_view(extents[2].begin, extents[2].end), std::string_view("GGGG\n"));
}

TEST(FastxSequenceScanTest, HeaderScanningPreservesLineBoundaries) {
    std::string const text =
        "ignored > text\n>first > title\r\nAC>GT\r>empty\r>last\nTT\n>trailing";
    auto extents = cusbf::detail::fastx_fasta_extents(text);
    ASSERT_EQ(extents.size(), 2U);
    EXPECT_EQ((std::string_view{extents[0].begin, extents[0].end}), "\nAC>GT\r");
    EXPECT_EQ((std::string_view{extents[1].begin, extents[1].end}), "TT\n");
    EXPECT_TRUE(cusbf::detail::fastx_fasta_extents("").empty());
    EXPECT_TRUE(cusbf::detail::fastx_fasta_extents("AC>GT\n").empty());
}

TEST(FastxSequenceScanTest, EmptyAndHeaderlessInputs) {
    EXPECT_TRUE(cusbf::detail::fastx_fasta_extents("").empty());
    EXPECT_TRUE(cusbf::detail::fastx_fasta_extents("ACGT\n").empty());
    EXPECT_TRUE(cusbf::detail::fastx_fasta_extents(">only").empty());
}

TEST(FastxSequenceScanTest, ParallelMatchesSingleThreadAcrossRecords) {
    // Records, multi-line sequences, CRLF, invalid bases, and cross-record windows.
    std::string const data =
        ">a\nACGTACGTACGTACGTACGTACGT\r\n"
        "ACGTNACGTACGTACGTACGTACGT\r\n"
        ">b\nTTTTGGGGAAAACCCC\r\n"
        ">c\nAC\r\n";
    for (uint32_t k = 1; k <= 12; ++k) {
        for (uint32_t threads : {1u, 2u, 3u, 5u, 8u}) {
            EXPECT_EQ(scan_parallel(data, k, threads), scan_single_threaded(data, k))
                << "k=" << k << " threads=" << threads;
        }
    }
}

TEST(FastxSequenceScanTest, PrefixIsBoundedBySeedBases) {
    auto const data = std::string(">s\n") + std::string(256, 'A') + "\n";
    auto const extents = cusbf::detail::fastx_fasta_extents(data);
    auto const spans = cusbf::detail::fastx_split_sequence_spans(extents, 8, 4);
    ASSERT_GE(spans.size(), 2u);
    for (auto const& span : spans) {
        uint32_t bases = 0;
        for (char ch : span.prefix) {
            if (!cusbf::detail::fastx_is_sequence_whitespace(ch)) {
                ++bases;
            }
        }
        EXPECT_LE(bases, 8u);
    }
}

TEST(FastxSequenceScanTest, SpansCoverTheWholeStream) {
    auto const data =
        std::string(">s\n") + std::string(500, 'G') + "\n>t\n" + std::string(300, 'C') + "\n";
    auto const extents = cusbf::detail::fastx_fasta_extents(data);
    auto const spans = cusbf::detail::fastx_split_sequence_spans(extents, 5, 7);
    uint64_t covered = 0;
    for (auto const& span : spans) {
        covered += span.total_bytes;
        uint64_t segment_bytes = 0;
        for (auto const& segment : span.segments) {
            segment_bytes += segment.size();
        }
        EXPECT_EQ(segment_bytes, span.total_bytes);
    }
    EXPECT_EQ(covered, 500u + 1u + 300u + 1u);
}
