gtest_dep = subproject('gtest').get_variable('gtest_dep')
gtest_main_dep = subproject('gtest').get_variable('gtest_main_dep')

test_exe = executable(
    'test-bloom-filter',
    ['test_bloom_filter.cu', 'test_fastx.cu'],
    dependencies: [bloom_dep, gtest_dep, gtest_main_dep],
    cuda_args: cuda_args,
    install: false,
)

test('bloom-filter-tests', test_exe)
