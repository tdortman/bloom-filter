project('superbloom-cpu', 'c',
    version: '0.1.0',
    meson_version: '>= 1.3.0',
)

cargo = find_program('cargo')

superbloom_cpu_benchmark = custom_target(
    'superbloom-cpu-benchmark',
    command: [
        cargo, 'build',
        '--release',
        '--bin', 'benchmark',
        '--manifest-path', meson.current_source_dir() / 'Cargo.toml',
        '--target-dir', '@OUTDIR@',
    ],
    output: 'release',
    console: true,
    build_by_default: true,
    build_always_stale: true,
    env: {'RUSTFLAGS': '-C target-cpu=native'},
)

_cargo_lib_target_dir = meson.current_build_dir() / 'superbloom-cargo-dir'

superbloom_cpu_lib = custom_target(
    'superbloom-cpu-lib',
    output: 'libsuperbloom.a',
    command: [
        'sh', '-c',
        'cargo build --release --lib --manifest-path "$1" --target-dir "$2" && cp "$2/release/$3" "$4"',
        '_',
        meson.current_source_dir() / 'Cargo.toml',
        _cargo_lib_target_dir,
        'libsuperbloom.a',
        '@OUTPUT@',
    ],
    console: true,
    build_by_default: true,
    build_always_stale: true,
    env: {'RUSTFLAGS': '-C target-cpu=native'},
)

cc = meson.get_compiler('c')
lzma_dep = cc.find_library('lzma', required: true)
bz2_dep = cc.find_library('bz2', required: true)

superbloom_cpu_dep = declare_dependency(
    link_with: superbloom_cpu_lib,
    dependencies: [lzma_dep, bz2_dep],
)
