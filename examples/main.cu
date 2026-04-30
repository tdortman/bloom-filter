#include <CLI/CLI.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>

#include <bloom/BloomFilter.cuh>

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

std::string generateRandomProtein(uint64_t length, uint32_t seed) {
    static constexpr char symbols[] = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                                       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'};

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 19);

    std::string sequence(length, 'A');
    for (char& symbol : sequence) {
        symbol = symbols[dist(rng)];
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

template <typename Config>
int runDemo(
    uint64_t filterBits,
    std::string_view sequence,
    std::string_view query,
    std::string_view insertFastxPath,
    std::string_view queryFastxPath,
    bool useInsertFastx,
    bool useRawSequence,
    bool useGeneratedFastx
) {
    bloom::Filter<Config> filter(filterBits);

    uint64_t inserted = 0;
    uint64_t queryKmers = 0;
    uint64_t positives = 0;
    uint64_t insertedBases = 0;
    uint64_t queriedBases = 0;
    uint64_t insertedRecords = 0;
    uint64_t queriedRecords = 0;

    if (useInsertFastx) {
        const auto report = filter.insertFastxFile(insertFastxPath);
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

    if (!queryFastxPath.empty()) {
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
    } else if (useInsertFastx) {
        const auto report = filter.queryFastxFile(insertFastxPath);
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

    std::cout << "Inserted records: " << insertedRecords << "\n";
    std::cout << "Queried records: " << queriedRecords << "\n";
    std::cout << "\n";
    std::cout << "Inserted bases: " << insertedBases << "\n";
    std::cout << "Queried bases: " << queriedBases << "\n";
    std::cout << "\n";
    std::cout << "Inserted k-mers: " << inserted << "\n";
    std::cout << "Query k-mers: " << queryKmers << "\n";
    std::cout << "Positive k-mers: " << positives << "\n";
    std::cout << "\n";
    std::cout << "Load factor: " << filter.loadFactor() << "\n";
    return 0;
}

int main(int argc, char** argv) {
    using Config = bloom::Config<31, 28, 16, 4, 256>;
    using ProteinConfig = bloom::Config<12, 10, 6, 4, 256, bloom::ProteinAlphabet>;

    CLI::App app{"GPU SuperBloom demo"};

    std::string sequence;
    std::string query;
    std::string mode = "dna";
    std::string insertFastxPath;
    std::string queryFastxPath;
    uint64_t filterBits = 1ULL << 24;
    uint64_t sequenceLength = 1ULL << 16;
    uint32_t seed = 42;

    auto* sequenceOption = app.add_option(
        "sequence",
        sequence,
        "Raw sequence to insert into the filter (optional if --length or --insert-fastx is "
        "used)"
    );
    auto* queryOption =
        app.add_option("query", query, "Raw sequence to query (defaults to the inserted input)");
    auto* insertFastxOption =
        app.add_option("--insert-fastx", insertFastxPath, "FASTA/FASTQ file to insert");
    auto* queryFastxOption =
        app.add_option("--query-fastx", queryFastxPath, "FASTA/FASTQ file to query");
    app.add_option("--length", sequenceLength, "Generate a random sequence of this length")
        ->default_val(sequenceLength);
    app.add_option("--seed", seed, "Random seed for generated sequence")->default_val(seed);
    app.add_option("--mode", mode, "Input alphabet mode: dna or protein")
        ->check(CLI::IsMember({"dna", "protein"}))
        ->default_val(mode);
    app.add_option(
           "--filter-bits", filterBits, "Total bloom-filter bits before power-of-two rounding"
    )
        ->default_val(filterBits);

    insertFastxOption->excludes(sequenceOption);
    queryFastxOption->excludes(queryOption);

    CLI11_PARSE(app, argc, argv);

    const bool useInsertFastx = !insertFastxPath.empty();
    const bool useRawSequence = !useInsertFastx && !sequence.empty();
    const bool useGeneratedFastx = !useInsertFastx && !useRawSequence;

    if (useGeneratedFastx) {
        sequence = mode == "protein" ? generateRandomProtein(sequenceLength, seed)
                                     : generateRandomDNA(sequenceLength, seed);
    }

    try {
        if (mode == "protein") {
            return runDemo<ProteinConfig>(
                filterBits,
                sequence,
                query,
                insertFastxPath,
                queryFastxPath,
                useInsertFastx,
                useRawSequence,
                useGeneratedFastx
            );
        }
        return runDemo<Config>(
            filterBits,
            sequence,
            query,
            insertFastxPath,
            queryFastxPath,
            useInsertFastx,
            useRawSequence,
            useGeneratedFastx
        );
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
