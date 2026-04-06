#include <CLI/CLI.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>

#include <bloom/BloomFilter.cuh>

namespace {

std::string generateRandomDNA(uint64_t length, uint32_t seed) {
    static constexpr char bases[] = {'A', 'C', 'G', 'T'};

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 3);

    std::string sequence(length, 'A');
    for (char& base : sequence) {
        base = bases[dist(rng)];
    }
    return sequence;
}

}  // namespace

int main(int argc, char** argv) {
    using Config = bloom::Config<31, 21, 27, 6, 256, 512>;

    CLI::App app{"GPU SuperBloom demo"};

    std::string sequence;
    std::string query;
    uint64_t filterBits = 1ULL << 24;
    uint64_t chunkBases = bloom::Filter<Config>::defaultChunkBases;
    uint64_t sequenceLength = 1ULL << 16;
    uint32_t seed = 42;

    app.add_option("sequence", sequence, "Sequence to insert into the filter (optional if --length is used)");
    app.add_option("query", query, "Sequence to query (defaults to the inserted sequence)");
    app.add_option("--length", sequenceLength, "Generate a random DNA sequence of this length")
        ->default_val(sequenceLength);
    app.add_option("--seed", seed, "Random seed for generated DNA")->default_val(seed);
    app.add_option("--filter-bits", filterBits, "Total bloom-filter bits before power-of-two rounding")
        ->default_val(filterBits);
    app.add_option("--chunk-bases", chunkBases, "Host-to-GPU chunk size in bases")
        ->default_val(chunkBases);

    CLI11_PARSE(app, argc, argv);

    if (sequence.empty()) {
        sequence = generateRandomDNA(sequenceLength, seed);
    }

    if (query.empty()) {
        query = sequence;
    }

    bloom::Filter<Config> filter(filterBits, chunkBases);
    const uint64_t inserted = filter.insertSequence(sequence);
    const auto hits = filter.containsSequence(query);
    const uint64_t positives = std::count(hits.begin(), hits.end(), uint8_t{1});

    std::cout << "Inserted k-mers: " << inserted << "\n";
    std::cout << "Query k-mers: " << hits.size() << "\n";
    std::cout << "Positive k-mers: " << positives << "\n";
    std::cout << "Load factor: " << filter.loadFactor() << "\n";
    std::cout << "Sequence length: " << sequence.size() << "\n";
    return 0;
}
