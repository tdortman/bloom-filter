#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark_common.cuh"

#define SUPERBLOOM_FIRST_INSERT_QUERY_FPR_CONFIG(X) X(31, 28, 16, 4)

#define SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR(X) SUPERBLOOM_FIRST_INSERT_QUERY_FPR_CONFIG(X)

#define SUPERBLOOM_CONFIGS_FPR_ONLY(X) \
    X(31, 31, 21, 4)                   \
    X(31, 30, 21, 4)                   \
    X(31, 28, 21, 4)                   \
    X(31, 27, 21, 4)                   \
    X(31, 20, 21, 4)                   \
    X(31, 16, 21, 4)

#define FOR_EACH_SUPERBLOOM_CONFIG(X)      \
    SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR(X) \
    SUPERBLOOM_CONFIGS_FPR_ONLY(X)

FOR_EACH_SUPERBLOOM_CONFIG(BENCHMARK_DEFINE_SUPERBLOOM_CONFIG_AND_FIXTURE)

#define BENCHMARK_DEFINE_SUPERBLOOM_CPU_FIXTURE(K, S, M, H)     \
    using BENCHMARK_SUPERBLOOM_CPU_FIXTURE_SYMBOL(K, S, M, H) = \
        benchmark_common::SuperBloomCpuFixture<BENCHMARK_SUPERBLOOM_CONFIG_SYMBOL(K, S, M, H)>;

FOR_EACH_SUPERBLOOM_CONFIG(BENCHMARK_DEFINE_SUPERBLOOM_CPU_FIXTURE)

#undef BENCHMARK_DEFINE_SUPERBLOOM_CPU_FIXTURE

// GPU SuperBloom benchmark definitions

#define DEFINE_GPU_INSERT_QUERY_FPR(K, S, M, H) \
    BENCHMARK_DEFINE_SUPERBLOOM_ALL(BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

#define DEFINE_GPU_FPR_ONLY(K, S, M, H) \
    BENCHMARK_DEFINE_SUPERBLOOM_FPR_ONLY(BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR(DEFINE_GPU_INSERT_QUERY_FPR)
SUPERBLOOM_CONFIGS_FPR_ONLY(DEFINE_GPU_FPR_ONLY)

#undef DEFINE_GPU_FPR_ONLY
#undef DEFINE_GPU_INSERT_QUERY_FPR

// CPU SuperBloom benchmark definitions

#define DEFINE_CPU_INSERT_QUERY_FPR(K, S, M, H) \
    BENCHMARK_DEFINE_SUPERBLOOM_CPU_ALL(BENCHMARK_SUPERBLOOM_CPU_FIXTURE_SYMBOL(K, S, M, H))

#define DEFINE_CPU_FPR_ONLY(K, S, M, H) \
    BENCHMARK_DEFINE_SUPERBLOOM_CPU_FPR_ONLY(BENCHMARK_SUPERBLOOM_CPU_FIXTURE_SYMBOL(K, S, M, H))

SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR(DEFINE_CPU_INSERT_QUERY_FPR)
SUPERBLOOM_CONFIGS_FPR_ONLY(DEFINE_CPU_FPR_ONLY)

#undef DEFINE_CPU_FPR_ONLY
#undef DEFINE_CPU_INSERT_QUERY_FPR

// GPU SuperBloom registrations

#define REGISTER_GPU_INSERT_QUERY_FPR(K, S, M, H) \
    BENCHMARK_REGISTER_SUPERBLOOM_ALL(BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

#define REGISTER_GPU_FPR_ONLY(K, S, M, H) \
    BENCHMARK_REGISTER_SUPERBLOOM_FPR_ONLY(BENCHMARK_SUPERBLOOM_FIXTURE_SYMBOL(K, S, M, H))

SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR(REGISTER_GPU_INSERT_QUERY_FPR)
SUPERBLOOM_CONFIGS_FPR_ONLY(REGISTER_GPU_FPR_ONLY)

#undef REGISTER_GPU_FPR_ONLY
#undef REGISTER_GPU_INSERT_QUERY_FPR

// CPU SuperBloom registrations

#define REGISTER_CPU_INSERT_QUERY_FPR(K, S, M, H) \
    BENCHMARK_REGISTER_SUPERBLOOM_CPU_ALL(BENCHMARK_SUPERBLOOM_CPU_FIXTURE_SYMBOL(K, S, M, H))

#define REGISTER_CPU_FPR_ONLY(K, S, M, H) \
    BENCHMARK_REGISTER_SUPERBLOOM_CPU_FPR_ONLY(BENCHMARK_SUPERBLOOM_CPU_FIXTURE_SYMBOL(K, S, M, H))

SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR(REGISTER_CPU_INSERT_QUERY_FPR)
SUPERBLOOM_CONFIGS_FPR_ONLY(REGISTER_CPU_FPR_ONLY)

#undef REGISTER_CPU_FPR_ONLY
#undef REGISTER_CPU_INSERT_QUERY_FPR

#undef FOR_EACH_SUPERBLOOM_CONFIG
#undef SUPERBLOOM_CONFIGS_FPR_ONLY
#undef SUPERBLOOM_CONFIGS_INSERT_QUERY_FPR
#undef SUPERBLOOM_FIRST_INSERT_QUERY_FPR_CONFIG

int main(int argc, char** argv) {
    benchmark_common::g_cpuFastxParallelizeRecords = true;

    std::vector<char*> benchmarkArgv;
    benchmarkArgv.reserve(argc);
    benchmarkArgv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        constexpr const char* numRecordsPrefix = "--num-query-records=";
        if (std::strncmp(arg.c_str(), numRecordsPrefix, std::strlen(numRecordsPrefix)) == 0) {
            benchmark_common::g_cpuFastxParallelizeRecords = true;
            benchmark_common::g_cpuFastxNumRecords =
                std::stoull(arg.substr(std::strlen(numRecordsPrefix)));
            continue;
        }
        if (arg == "--num-query-records") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --num-query-records" << std::endl;
                return 1;
            }
            benchmark_common::g_cpuFastxParallelizeRecords = true;
            benchmark_common::g_cpuFastxNumRecords = std::stoull(argv[++i]);
            continue;
        }

        benchmarkArgv.push_back(argv[i]);
    }

    if (benchmark_common::g_cpuFastxParallelizeRecords &&
        benchmark_common::g_cpuFastxNumRecords == 0) {
        std::cerr << "--num-query-records must be >= 1" << std::endl;
        return 1;
    }

    int benchmarkArgc = static_cast<int>(benchmarkArgv.size());
    ::benchmark::Initialize(&benchmarkArgc, benchmarkArgv.data());
    if (::benchmark::ReportUnrecognizedArguments(benchmarkArgc, benchmarkArgv.data())) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    fflush(stdout);
    std::_Exit(0);
}
