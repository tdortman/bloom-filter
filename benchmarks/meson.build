benchmark_dep = subproject('google-benchmark').get_variable('google_benchmark_dep')

cuco_dep = subproject('cuco').get_variable('cuco_dep')

cuckoogpu_dep = subproject('cuckoogpu').get_variable('cuckoogpu_dep')

bench_deps = [benchmark_dep, bloom_dep]

benchmarks = {}

benchmarks += {
    'gpu-filter-comparison': {
        'src': 'gpu_filter_comparison.cu',
        'deps': [cuco_dep, cuckoogpu_dep],
    },
    'fpr-fastx-sweep': {
        'src': 'fpr_fastx_sweep.cu',
        'deps': [cuco_dep],
    },
    'sh-sweep-sparse': {
        'src': 'sh_sweep.cu',
        'deps': [],
    },
    'sh-sweep-dense': {
        'src': 'sh_sweep.cu',
        'extra_args': ['-DSH_SWEEP_DENSE'],
        'deps': [],
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