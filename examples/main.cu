#include <CLI/CLI.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>

#include <bloom/BloomFilter.cuh>
#include <bloom/PackedKmerBinary.hpp>

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

std::string wrapLines(std::string_view text, uint64_t lineLength) {
    const uint64_t width = std::max<uint64_t>(1, lineLength);
    std::string wrapped;
    wrapped.reserve(text.size() + text.size() / width + 1);

    if (text.empty()) {
        wrapped.push_back('\n');
        return wrapped;
    }

    for (uint64_t offset = 0; offset < text.size(); offset += width) {
        const auto chunk = std::min<uint64_t>(width, text.size() - offset);
        wrapped.append(text.substr(offset, chunk));
        wrapped.push_back('\n');
    }
    return wrapped;
}

std::string makeFastaRecord(std::string_view name, std::string_view sequence, uint64_t lineLength) {
    std::string record = ">" + std::string(name) + "\n";
    record += wrapLines(sequence, lineLength);
    return record;
}

std::string makeFastqRecord(std::string_view name, std::string_view sequence, uint64_t lineLength) {
    std::string quality(sequence.size(), 'I');
    std::string record = "@" + std::string(name) + "\n";
    record += wrapLines(sequence, lineLength);
    record += "+\n";
    record += wrapLines(quality, lineLength);
    return record;
}

int main(int argc, char** argv) {
    using Config = bloom::Config<31, 27, 21, 6, 256>;

    CLI::App app{"GPU SuperBloom demo"};

    std::string sequence;
    std::string query;
    std::string sequenceFastxPath;
    std::string queryFastxPath;
    std::string sequencePackedPath;
    std::string queryPackedPath;
    uint64_t filterBits = 1ULL << 24;
    uint64_t sequenceLength = 1ULL << 16;
    uint32_t seed = 42;

    auto* sequenceOption = app.add_option(
        "sequence",
        sequence,
        "Raw DNA sequence to insert into the filter (optional if --length or --sequence-fastx is "
        "used)"
    );
    auto* queryOption = app.add_option(
        "query", query, "Raw DNA sequence to query (defaults to the inserted input)"
    );
    auto* sequenceFastxOption =
        app.add_option("--sequence-fastx", sequenceFastxPath, "FASTA/FASTQ file to insert");
    auto* queryFastxOption =
        app.add_option("--query-fastx", queryFastxPath, "FASTA/FASTQ file to query");
    auto* sequencePackedOption = app.add_option(
        "--sequence-packed-binary", sequencePackedPath, "Packed k-mer binary file to insert"
    );
    auto* queryPackedOption = app.add_option(
        "--query-packed-binary", queryPackedPath, "Packed k-mer binary file to query"
    );
    app.add_option("--length", sequenceLength, "Generate a random DNA sequence of this length")
        ->default_val(sequenceLength);
    app.add_option("--seed", seed, "Random seed for generated DNA")->default_val(seed);
    app.add_option(
           "--filter-bits", filterBits, "Total bloom-filter bits before power-of-two rounding"
    )
        ->default_val(filterBits);

    sequenceFastxOption->excludes(sequenceOption);
    sequencePackedOption->excludes(sequenceOption);
    sequencePackedOption->excludes(sequenceFastxOption);
    queryFastxOption->excludes(queryOption);
    queryPackedOption->excludes(queryOption);
    queryPackedOption->excludes(queryFastxOption);

    CLI11_PARSE(app, argc, argv);

    const bool useSequencePacked = !sequencePackedPath.empty();
    const bool useSequenceFastx = !useSequencePacked && !sequenceFastxPath.empty();
    const bool useRawSequence = !useSequencePacked && !useSequenceFastx && !sequence.empty();
    const bool useGeneratedFastx = !useSequencePacked && !useSequenceFastx && !useRawSequence;

    if (useGeneratedFastx) {
        sequence = generateRandomDNA(sequenceLength, seed);
    }

    try {
        bloom::Filter<Config> filter(filterBits);

        auto loadPackedBinary = [](std::string_view path) {
            const auto file = bloom::readPackedKmerBinaryFile(path);
            if (file.k != Config::k) {
                throw std::runtime_error(
                    "Packed k-mer binary uses k=" + std::to_string(file.k) +
                    ", but this example is compiled for k=" + std::to_string(Config::k)
                );
            }
            return file;
        };

        std::optional<bloom::PackedKmerBinaryFile> sequencePackedFile;
        std::optional<bloom::PackedKmerBinaryFile> queryPackedFile;

        uint64_t inserted = 0;
        uint64_t queryKmers = 0;
        uint64_t positives = 0;
        uint64_t insertedBases = 0;
        uint64_t queriedBases = 0;
        uint64_t insertedRecords = 0;
        uint64_t queriedRecords = 0;
        bool insertedBasesKnown = true;
        bool queriedBasesKnown = true;
        bool insertedRecordsKnown = true;
        bool queriedRecordsKnown = true;
        bool usedPackedInput = false;

        if (useSequencePacked) {
            sequencePackedFile = loadPackedBinary(sequencePackedPath);
            inserted = filter.insertPackedKmers(sequencePackedFile->kmers);
            insertedBasesKnown = false;
            insertedRecordsKnown = false;
            usedPackedInput = true;
        } else if (useSequenceFastx) {
            const auto report = filter.insertFastxFile(sequenceFastxPath);
            inserted = report.insertedKmers;
            insertedBases = report.indexedBases;
            insertedRecords = report.recordsIndexed;
        } else if (useRawSequence) {
            inserted = filter.insertSequence(sequence);
            insertedBases = sequence.size();
            insertedRecords = 1;
        } else {
            std::istringstream inputFastx(makeFastaRecord("generated-insert", sequence, 73));
            const auto report = filter.insertFastx(inputFastx);
            inserted = report.insertedKmers;
            insertedBases = report.indexedBases;
            insertedRecords = report.recordsIndexed;
        }

        if (!queryPackedPath.empty()) {
            queryPackedFile = loadPackedBinary(queryPackedPath);
            const auto hits = filter.containsPackedKmers(queryPackedFile->kmers);
            queryKmers = hits.size();
            positives = std::count(hits.begin(), hits.end(), uint8_t{1});
            queriedBasesKnown = false;
            queriedRecordsKnown = false;
            usedPackedInput = true;
        } else if (!queryFastxPath.empty()) {
            const auto report = filter.queryFastxFile(queryFastxPath);
            queryKmers = report.queriedKmers;
            positives = report.positiveKmers;
            queriedBases = report.queriedBases;
            queriedRecords = report.recordsQueried;
        } else if (!query.empty()) {
            const auto hits = filter.containsSequence(query);
            queryKmers = hits.size();
            positives = std::count(hits.begin(), hits.end(), uint8_t{1});
            queriedBases = query.size();
            queriedRecords = 1;
        } else if (useSequencePacked) {
            const auto& packedFile = sequencePackedFile.value();
            const auto hits = filter.containsPackedKmers(packedFile.kmers);
            queryKmers = hits.size();
            positives = std::count(hits.begin(), hits.end(), uint8_t{1});
            queriedBasesKnown = false;
            queriedRecordsKnown = false;
            usedPackedInput = true;
        } else if (useSequenceFastx) {
            const auto report = filter.queryFastxFile(sequenceFastxPath);
            queryKmers = report.queriedKmers;
            positives = report.positiveKmers;
            queriedBases = report.queriedBases;
            queriedRecords = report.recordsQueried;
        } else if (useGeneratedFastx) {
            std::istringstream queryFastx(makeFastqRecord("generated-query", sequence, 59));
            const auto report = filter.queryFastx(queryFastx);
            queryKmers = report.queriedKmers;
            positives = report.positiveKmers;
            queriedBases = report.queriedBases;
            queriedRecords = report.recordsQueried;
        } else {
            const auto hits = filter.containsSequence(sequence);
            queryKmers = hits.size();
            positives = std::count(hits.begin(), hits.end(), uint8_t{1});
            queriedBases = sequence.size();
            queriedRecords = 1;
        }

        std::cout << "Inserted records: ";
        if (insertedRecordsKnown) {
            std::cout << insertedRecords;
        } else {
            std::cout << "n/a (packed k-mer input)";
        }
        std::cout << "\n";
        std::cout << "Queried records: ";
        if (queriedRecordsKnown) {
            std::cout << queriedRecords;
        } else {
            std::cout << "n/a (packed k-mer input)";
        }
        std::cout << "\n";
        std::cout << "\n";
        std::cout << "Inserted bases: ";
        if (insertedBasesKnown) {
            std::cout << insertedBases;
        } else {
            std::cout << "n/a (packed k-mer input)";
        }
        std::cout << "\n";
        std::cout << "Queried bases: ";
        if (queriedBasesKnown) {
            std::cout << queriedBases;
        } else {
            std::cout << "n/a (packed k-mer input)";
        }
        std::cout << "\n";
        std::cout << "\n";
        std::cout << "Inserted k-mers: " << inserted << "\n";
        std::cout << "Query k-mers: " << queryKmers << "\n";
        std::cout << "Positive k-mers: " << positives << "\n";
        if (usedPackedInput) {
            std::cout << "Packed k-mer k: " << Config::k << "\n";
        }
        std::cout << "\n";
        std::cout << "Load factor: " << filter.loadFactor() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
