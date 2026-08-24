#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cusbf/Alphabet.cuh>
#include <cusbf/detail/fastx_buffer_reader.hpp>

namespace cusbf::detail {

/// @brief A contiguous run of sequence bytes (record header lines excluded).
struct fastx_sequence_extent {
    const char* begin;
    const char* end;
};

/// @brief True for bytes that FASTA sequence consumers conventionally skip between bases.
[[nodiscard]] inline constexpr bool fastx_is_sequence_whitespace(char ch) noexcept {
    return ch == '\n' || ch == '\r' || ch == ' ' || ch == '\t';
}

/**
 * @brief Collects the sequence extents of every FASTA record in @p data.
 *
 * Each extent spans from the byte after a `>` header line to the byte before the next `>`
 * header line (or the end of the buffer), header bytes excluded. Line ends are located with the
 * SIMD-dispatched @ref fastx_line_end. Bytes outside extents (headers, leading blank lines) are
 * not part of any extent.
 */
[[nodiscard]] inline std::vector<fastx_sequence_extent> fastx_fasta_extents(std::string_view data) {
    std::vector<fastx_sequence_extent> extents;
    const char* const begin = data.data();
    const char* const end = begin + data.size();
    const char* p = begin;
    while (p < end) {
        auto const line_end =
            p + fastx_line_end(std::string_view{p, static_cast<size_t>(end - p)}, 0);
        const char* const line_end_ptr = line_end < end ? line_end : end;
        if (line_end_ptr > p && *p == '>') {
            const char* const seq_start = line_end < end ? line_end + 1 : end;
            const char* q = seq_start;
            while (q < end) {
                auto const next_end =
                    q + fastx_line_end(std::string_view{q, static_cast<size_t>(end - q)}, 0);
                const char* const next_end_ptr = next_end < end ? next_end : end;
                if (next_end_ptr > q && *q == '>') {
                    break;
                }
                q = next_end < end ? next_end + 1 : end;
            }
            if (q > seq_start) {
                extents.push_back({seq_start, q});
            }
            p = q;
        } else {
            p = line_end_ptr < end ? line_end_ptr + 1 : end;
        }
    }
    return extents;
}

/**
 * @brief One parallel scan span of the combined FASTA sequence stream.
 *
 * `segments` covers the span's bytes in order; it may cross record boundaries, with header bytes
 * excluded. `prefix` holds the raw bytes immediately before the span's first position (headers
 * excluded) that a rolling-window consumer must replay to rebuild its window state: up to
 * @p seed_bases valid bases, stopping early at an invalid base or the start of the buffer.
 */
struct fastx_sequence_span {
    std::string prefix;
    std::vector<std::string_view> segments;
    uint64_t total_bytes = 0;
};

namespace {

inline bool scan_step_back(
    std::vector<fastx_sequence_extent> const& extents, size_t& extent, const char*& position
) {
    if (position > extents[extent].begin) {
        --position;
        return true;
    }
    if (extent > 0) {
        --extent;
        position = extents[extent].end - 1;
        return true;
    }
    return false;
}

inline bool scan_step_forward(
    std::vector<fastx_sequence_extent> const& extents, size_t& extent, const char*& position
) {
    if (position + 1 < extents[extent].end) {
        ++position;
        return true;
    }
    if (extent + 1 < extents.size()) {
        ++extent;
        position = extents[extent].begin;
        return true;
    }
    return false;
}

}  // namespace

/**
 * @brief Splits the combined FASTA sequence stream into @p count contiguous spans.
 *
 * Each span carries the raw byte ranges of its slice of the sequence stream plus a materialised
 * prefix of the preceding @p seed_bases valid bases (whitespace included, headers excluded, cut
 * short at an invalid base or the buffer start). A rolling-window consumer that replays the
 * prefix, then consumes the segments, reproduces the single-threaded byte stream exactly, so the
 * per-span results can be concatenated in order. Whitespace between bases is preserved in both
 * the prefix and the segments; consumers skip it with @ref fastx_is_sequence_whitespace.
 */
[[nodiscard]] inline std::vector<fastx_sequence_span> fastx_split_sequence_spans(
    std::vector<fastx_sequence_extent> const& extents, uint32_t seed_bases, uint32_t count
) {
    std::vector<fastx_sequence_span> spans;
    if (extents.empty() || count == 0U) {
        return spans;
    }
    size_t total = 0;
    for (auto const& extent : extents) {
        total += static_cast<size_t>(extent.end - extent.begin);
    }
    if (total == 0) {
        return spans;
    }
    if (static_cast<size_t>(count) > total) {
        count = static_cast<uint32_t>(total);
    }
    auto const per = total / count + (total % count != 0U);
    spans.reserve(count);

    for (uint32_t t = 0; t < count; ++t) {
        auto const span_begin = static_cast<size_t>(t) * per;
        if (span_begin >= total) {
            break;
        }
        auto const span_end = span_begin + per < total ? span_begin + per : total;

        fastx_sequence_span span;
        span.total_bytes = span_end - span_begin;

        // Locate the extent and offset of the span start.
        size_t extent = 0;
        size_t offset = span_begin;
        while (extent < extents.size() &&
               offset >= static_cast<size_t>(extents[extent].end - extents[extent].begin)) {
            offset -= static_cast<size_t>(extents[extent].end - extents[extent].begin);
            ++extent;
        }

        // Seed: walk back up to seed_bases valid bases (or until an invalid base / EOF).
        size_t seed_extent = extent;
        const char* seed_start = extents[extent].begin + offset;
        {
            size_t e = extent;
            const char* q = seed_start;
            uint32_t collected = 0;
            while (collected < seed_bases && scan_step_back(extents, e, q)) {
                auto const ch = *q;
                if (fastx_is_sequence_whitespace(ch)) {
                    continue;
                }
                if (DnaAlphabet::encode(&ch) == DnaAlphabet::invalidSymbol) {
                    seed_extent = e;
                    seed_start = q;
                    scan_step_forward(extents, seed_extent, seed_start);
                    break;
                }
                ++collected;
                seed_extent = e;
                seed_start = q;
            }
        }

        // Materialise the prefix from the seed start to the span start.
        {
            size_t e = seed_extent;
            const char* q = seed_start;
            const char* target = extents[extent].begin + offset;
            while (e < extent || q < target) {
                if (q >= extents[e].end) {
                    ++e;
                    q = extents[e].begin;
                    continue;
                }
                span.prefix.push_back(*q);
                ++q;
            }
        }

        // Collect the span's segments.
        {
            size_t remaining = span_end - span_begin;
            const char* q = extents[extent].begin + offset;
            size_t e = extent;
            while (remaining > 0 && e < extents.size()) {
                auto const limit = static_cast<size_t>(extents[e].end - q);
                auto const step = limit < remaining ? limit : remaining;
                span.segments.emplace_back(q, step);
                remaining -= step;
                if (remaining > 0) {
                    ++e;
                    if (e < extents.size()) {
                        q = extents[e].begin;
                    }
                }
            }
        }

        spans.push_back(std::move(span));
    }
    return spans;
}

}  // namespace cusbf::detail
