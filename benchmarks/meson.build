benchmark_dep = subproject('google-benchmark').get_variable('google_benchmark_dep')

cuco_dep = subproject('cuco').get_variable('cuco_dep')

cuckoogpu_dep = subproject('cuckoogpu').get_variable('cuckoogpu_dep')

gpu_filters_sp = subproject('gpu-filters')
gqf_deps = gpu_filters_sp.get_variable('gqf_deps')
gqf_dep = gqf_deps['16']
tcf_dep = gpu_filters_sp.get_variable('tcf_dep')

bench_deps = [benchmark_dep, cusbf_dep]

superbloom_cpu_dep = subproject('superbloom-cpu').get_variable('superbloom_cpu_dep')

benchmarks = {
    'fastx-line-scan': {
        'src': 'fastx_line_scan.cpp',
        'deps': [],
    },
    'cusbf-host-overhead': {
        'src': 'cusbf_host_overhead.cu',
        'deps': [],
    },
    'dense-packed-throughput': {
        'src': 'dense_packed_throughput.cu',
        'deps': [],
    },
    'filter-comparison': {
        'src': 'filter_comparison.cu',
        'deps': [cuco_dep, cuckoogpu_dep, superbloom_cpu_dep, gqf_dep, tcf_dep],
    },
    'fpr-fastx-sweep': {
        'src': 'fpr_fastx_sweep.cu',
        'deps': [cuco_dep, cuckoogpu_dep, superbloom_cpu_dep, gqf_dep, tcf_dep],
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
            '-DPARAM_SWEEP_ALPHABET=cusbf::ProteinAlphabet',
        ]
    else
        s_values = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        m_values = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31]
        sweep_extra_args = [
            '-DPARAM_SWEEP_K=31',
            '-DPARAM_SWEEP_ALPHABET=cusbf::DnaAlphabet',
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

# gqf_dep carries -rdc=true for device-linking libgqf. When that path is active,
# repeat the CUDA arch list on the final link so nvcc does not emit a default
# single-arch device-link cubin for Thrust/CUB kernels.
tcf_benchmark_cuda_args = ['-Xcompiler=-Wno-implicit-fallthrough']

foreach name, config : benchmarks
    extra_cuda_args = config.get('extra_args', [])
    extra_link_args = []
    deps = config.get('deps', [])
    if tcf_dep in deps
        extra_cuda_args += tcf_benchmark_cuda_args
    endif
    if gqf_dep in deps
        extra_link_args += cuda_arch_args
    endif
    executable(
        name,
        config['src'],
        dependencies: bench_deps + deps,
        cuda_args: cuda_args + extra_cuda_args,
        link_args: extra_link_args,
        install: false,
    )
endforeach

executable(
    'gpu-filter-profiler',
    'profiler_benchmark.cu',
    dependencies: bench_deps + [cli11_dep, cuco_dep, cuckoogpu_dep],
    cuda_args: cuda_args,
    install: false,
)
