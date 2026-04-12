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
