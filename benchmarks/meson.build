benchmark_dep = subproject('google-benchmark').get_variable('google_benchmark_dep')

cuco_dep = subproject('cuco').get_variable('cuco_dep')

cuckoogpu_dep = subproject('cuckoogpu').get_variable('cuckoogpu_dep')

bench_deps = [benchmark_dep, bloom_dep]

superbloom_cpu_dep = subproject('superbloom-cpu').get_variable('superbloom_cpu_dep')

benchmarks = {
    'gpu-filter-comparison': {
        'src': 'gpu_filter_comparison.cu',
        'deps': [cuco_dep, cuckoogpu_dep],
    },
    'fpr-fastx-sweep': {
        'src': 'fpr_fastx_sweep.cu',
        'deps': [cuco_dep, superbloom_cpu_dep],
    },
}

if get_option('param_sweep').enabled()
    alphabet = get_option('param_sweep_alphabet')
    if alphabet == 'protein'
        s_values = [6, 7, 8, 9, 10, 11, 12]
        m_values = [4, 6, 8, 10, 12]
        sweep_extra_args = [
            '-DPARAM_SWEEP_PROTEIN',
            '-DPARAM_SWEEP_K=12',
            '-DPARAM_SWEEP_ALPHABET=bloom::ProteinAlphabet',
        ]
    else
        s_values = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        m_values = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31]
        sweep_extra_args = [
            '-DPARAM_SWEEP_K=31',
            '-DPARAM_SWEEP_ALPHABET=bloom::DnaAlphabet',
        ]
    endif
    num_m = m_values.length()
    foreach s_idx : range(s_values.length())
        foreach m_idx : range(num_m)
            group_idx = s_idx * num_m + m_idx
            benchmarks += {
                'param-sweep-group@0@'.format(group_idx): {
                    'src': 'param_sweep.cu',
                    'extra_args': sweep_extra_args
                    + ['-DPARAM_SWEEP_GROUP=@0@'.format(group_idx)],
                    'deps': [],
                },
            }
        endforeach
    endforeach
endif

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

executable(
    'gpu-filter-profiler',
    'profiler_benchmark.cu',
    dependencies: bench_deps + [cli11_dep, cuco_dep, cuckoogpu_dep],
    cuda_args: cuda_args,
    install: false,
)

executable(
    'cpu-filter-comparison',
    'cpu_filter_comparison.cu',
    dependencies: bench_deps + [superbloom_cpu_dep],
    cuda_args: cuda_args,
    install: false,
)
