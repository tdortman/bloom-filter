benchmark_dep = subproject('google-benchmark').get_variable('google_benchmark_dep')

cmake = import('cmake')

cuco_opts = cmake.subproject_options()
cuco_opts.add_cmake_defines(
    {
        'CMAKE_BUILD_TYPE': 'Release',
        'BUILD_TESTS': 'OFF',
        'BUILD_TESTING': 'OFF',
        'BUILD_BENCHMARKS': 'OFF',
        'BUILD_EXAMPLES': 'OFF',
        'INSTALL_CUCO': 'OFF',
        'CUCO_ENABLE_EXPORTS': 'OFF',
        'RAPIDS_ENABLE_EXPORTS': 'OFF',
    },
)
cuco_dep = cmake.subproject('cuco', options: cuco_opts).dependency('cuco')

bench_deps = [benchmark_dep, bloom_dep]

benchmarks = {}

benchmarks += {
    'superbloom-vs-cuco-bloom': {
        'src': 'superbloom_vs_cuco_bloom.cu',
        'deps': [cuco_dep],
    },
}

foreach name, config : benchmarks
    executable(
        name,
        config['src'],
        dependencies: bench_deps + config.get('deps', []),
        cuda_args: cuda_args + config.get('extra_args', []),
        install: false,
    )
endforeach

kmc_api_dep = subproject('kmc').get_variable('kmc_api_dep')

executable(
    'kmc_to_binary',
    'kmc_to_binary.cpp',
    dependencies: [bloom_dep, kmc_api_dep, cli11_dep],
    install: true,
)

executable(
    'superbloom-gpu-benchmark',
    'superbloom_gpu_benchmark.cu',
    dependencies: [bloom_dep, cli11_dep, benchmark_dep],
    cuda_args: cuda_args,
    install: false,
)

subproject('superbloom-cpu')
