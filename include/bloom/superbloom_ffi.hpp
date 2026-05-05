#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

/// Create a SuperBloom filter with the given configuration.
/// Returns an opaque handle, or NULL on failure.
void* superbloom_create(
    uint16_t k,
    uint16_t m,
    uint16_t s,
    size_t n_hashes,
    uint8_t bit_vector_size_exponent,
    uint8_t block_size_exponent
);

/// Insert a raw DNA sequence into the filter (mutable mode).
/// Returns the number of k-mers added, or -1 on error.
int64_t superbloom_insert_sequence(void* handle, const uint8_t* seq, size_t len);

/// Insert every FASTA/FASTQ record in a file into the filter (mutable mode).
/// Returns the total number of k-mers added, or -1 on error.
int64_t superbloom_insert_fastx_path(void* handle, const char* path);

/// Query a raw DNA sequence (automatically freezes if needed).
/// Returns the number of k-mers reported present, or -1 on error.
int64_t superbloom_query_sequence(const void* handle, const uint8_t* seq, size_t len);

/// Query every FASTA/FASTQ record in a file (automatically freezes if needed).
/// Returns the total number of positive k-mers, or -1 on error.
int64_t superbloom_query_fastx_path(const void* handle, const char* path);

/// Explicitly freeze the filter for query-only access.
/// Returns 0 on success, -1 on error.
int32_t superbloom_freeze(const void* handle);

/// Set the thread count for the Rayon thread pool.
/// Must be called before any insert/query operations.
/// Returns 0 on success, -1 on error.
int32_t superbloom_set_threads(void* handle, size_t n);

/// Return the total number of filter bits (2^bit_vector_size_exponent).
uint64_t superbloom_filter_bits(const void* handle);

/// Destroy a filter created with superbloom_create.
void superbloom_destroy(void* handle);
}
