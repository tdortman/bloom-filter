benchmark_dep = subproject('google-benchmark').get_variable('google_benchmark_dep')

cmake = import('cmake')

bench_deps = [benchmark_dep]

benchmarks = {}

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
    dependencies: [kmc_api_dep, cli11_dep],
    install: true,
)
