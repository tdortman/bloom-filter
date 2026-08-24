gtest_dep = subproject('gtest').get_variable('gtest_dep')
gtest_main_dep = subproject('gtest').get_variable('gtest_main_dep')
benchmark_dep = subproject('google-benchmark').get_variable('google_benchmark_dep')

test_exe = executable(
    'test-cusbf',
    [
        'test_bloom_filter.cu',
        'test_fastx.cu',
        'test_fastx_chunk.cpp',
        'test_dense_record_batch.cpp',
        'test_fastx_dispatch.cpp',
        'test_fastx_buffer_reader.cpp',
        'test_fastx_sequence_scan.cpp',
        'test_fastx_host_limits.cpp',
        'test_normalized_record_batch.cpp',
        'test_query_layout.cpp',
        'test_benchmark_common.cu',
    ],
    dependencies: [cusbf_dep, benchmark_dep, gtest_dep, gtest_main_dep],
    cuda_args: cuda_args,
    install: false,
)

test('cusbf-tests', test_exe)

large_fastx_exe = executable(
    'test-large-fastx',
    ['test_large_fastx.cu'],
    dependencies: [cusbf_dep, gtest_dep, gtest_main_dep],
    cuda_args: cuda_args,
    install: false,
)

if get_option('large_fastx_tests').enabled()
    test(
        'cusbf-large-fastx',
        large_fastx_exe,
        timeout: 14400,
        env: ['CUSBF_LARGE_FASTX_GB=16'],
    )
endif
