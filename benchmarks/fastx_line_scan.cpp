#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>

#include <cusbf/detail/fastx_buffer_reader.hpp>
#include <cusbf/detail/fastx_dense_batch.hpp>
#include <random>

namespace {

constexpr size_t kBufferBytes = 64u << 20;
constexpr size_t kLineBytes = 151;
const std::string long_line_input = std::string(kBufferBytes, 'A');

const std::string& input() {
    static const std::string data = [] {
        std::string value(kBufferBytes, 'A');
        for (size_t i = kLineBytes - 1; i < value.size(); i += kLineBytes) {
            value[i] = '\n';
        }
        return value;
    }();
    return data;
}

const std::string& fastq_input() {
    static const std::string data = [] {
        const std::string record =
            "@read\n" + std::string(150, 'A') + "\n+\n" + std::string(150, 'I') + "\n";
        std::string value;
        value.reserve(kBufferBytes);
        while (value.size() + record.size() <= kBufferBytes) {
            value += record;
        }
        return value;
    }();
    return data;
}

std::string runtime_fastq_input() {
    std::string data = fastq_input();
    std::mt19937 rng(1);
    constexpr size_t record_bytes = 310;
    for (size_t record = 0; record + record_bytes <= data.size(); record += record_bytes) {
        for (size_t offset = record + 6; offset < record + 156; ++offset) {
            data[offset] = "ACGT"[rng() & 3];
        }
        for (size_t offset = record + 159; offset < record + 309; ++offset) {
            data[offset] = static_cast<char>('!' + (rng() % 41));
        }
    }
    return data;
}

size_t libc_line_end(std::string_view data, size_t position) {
    const char* begin = data.data() + position;
    const size_t remaining = data.size() - position;
    const auto* newline = static_cast<const char*>(std::memchr(begin, '\n', remaining));
    const size_t newline_offset =
        newline == nullptr ? remaining : static_cast<size_t>(newline - begin);
    const auto* carriage_return =
        static_cast<const char*>(std::memchr(begin, '\r', newline_offset));
    return position + (carriage_return == nullptr ? newline_offset
                                                  : static_cast<size_t>(carriage_return - begin));
}

template <auto FindLineEnd>
void scan(benchmark::State& state) {
    const std::string_view data = input();
    for (auto _ : state) {
        size_t position = 0;
        while (position < data.size()) {
            position = FindLineEnd(data, position);
            position += position < data.size();
        }
        benchmark::DoNotOptimize(position);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * data.size()));
}

template <auto FindLineEnd>
void scan_long_line(benchmark::State& state) {
    const std::string_view data = long_line_input;
    for (auto _ : state) {
        benchmark::DoNotOptimize(FindLineEnd(data, 0));
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * data.size()));
}

void parser(benchmark::State& state) {
    const std::string_view data = fastq_input();
    for (auto _ : state) {
        cusbf::detail::FastxBufferReader reader(data);
        cusbf::detail::FastxRecord record;
        size_t records = 0;
        while (true) {
            auto result = reader.nextRecord(record);
            if (!result || !*result) {
                break;
            }
            ++records;
        }
        benchmark::DoNotOptimize(records);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * data.size()));
}
BENCHMARK(parser);

void dense_batch(benchmark::State& state) {
    const std::string data = runtime_fastq_input();
    const std::string_view input_data = data;
    cusbf::detail::FastxRecord record;
    cusbf::DenseRecordBatchBuilder batch(data.size());
    for (auto _ : state) {
        batch.clear();
        cusbf::detail::FastxBufferReader reader(input_data);
        while (true) {
            const auto range = reader.appendNextRecord(
                record, batch.sequence_buffer(), batch.external_sequence_slot()
            );
            if (!range) {
                state.SkipWithError(range.error().message());
                return;
            }
            if (!*range) {
                break;
            }
            batch.push_range(**range);
        }
        uint64_t checksum = 0;
        for (const auto& range : batch.ranges()) {
            const std::string_view sequence = batch.sequence_view().substr(
                static_cast<size_t>(range.sequenceOffset), static_cast<size_t>(range.sequenceBytes)
            );
            for (const unsigned char byte : sequence) {
                checksum += byte;
            }
        }
        benchmark::DoNotOptimize(checksum);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * input_data.size()));
}
BENCHMARK(dense_batch);

void libc(benchmark::State& state) {
    scan<libc_line_end>(state);
}
BENCHMARK(libc);

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && !defined(__CUDACC__)
void avx2_long_line(benchmark::State& state) {
    scan_long_line<cusbf::detail::fastx_line_end_avx2>(state);
}
BENCHMARK(avx2_long_line);
#endif

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && !defined(__CUDACC__)
[[gnu::target("avx2")]]
void avx2_memory_read(benchmark::State& state) {
    const std::string_view data = long_line_input;
    for (auto _ : state) {
        __m256i accumulator = _mm256_setzero_si256();
        for (size_t position = 0; position < data.size(); position += 32) {
            accumulator = _mm256_xor_si256(
                accumulator,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data.data() + position))
            );
        }
        benchmark::DoNotOptimize(_mm256_movemask_epi8(accumulator));
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * data.size()));
}
BENCHMARK(avx2_memory_read);
#endif
void scalar(benchmark::State& state) {
    scan<cusbf::detail::fastx_line_end_scalar>(state);
}
BENCHMARK(scalar);

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && !defined(__CUDACC__)
void avx2(benchmark::State& state) {
    scan<cusbf::detail::fastx_line_end_avx2>(state);
}
BENCHMARK(avx2);
#endif

#if defined(__ARM_FEATURE_SVE2)
void sve2(benchmark::State& state) {
    scan<cusbf::detail::fastx_line_end_sve2>(state);
}
BENCHMARK(sve2);
#elif defined(__aarch64__) || defined(_M_ARM64)
void neon(benchmark::State& state) {
    scan<cusbf::detail::fastx_line_end_neon>(state);
}
BENCHMARK(neon);
#endif

}  // namespace

BENCHMARK_MAIN();
