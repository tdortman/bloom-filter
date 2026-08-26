{
  description = "Bloom Filter";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, rust-overlay, ... }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
    in
    {
      devShells = nixpkgs.lib.genAttrs supportedSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
            config.allowUnfree = true;
          };

          lib = pkgs.lib;
          cudaPkgs = pkgs.cudaPackages_13_2;
          llvmPkgs = pkgs.llvmPackages_22;

          rustToolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [
              "rust-src"
              "rust-analyzer"
            ];
          };

          cudaToolkit = pkgs.symlinkJoin {
            name = "cuda-toolkit";
            paths = with cudaPkgs; [
              cuda_nvcc
              cuda_crt
              cuda_cudart
              cuda_cccl
              cuda_profiler_api.include
              cuda_cuobjdump
              cuda_nvdisasm

              cuda_gdb.bin
              nsight_systems
              nsight_compute

              # I do not know why cuRAND headers are necessary
              # for clangd to not freak out about STL headers when cuda_crt is
              # also present but at least it's a somewhat cheap dependency...
              libcurand.include
            ];
          };

          cuda = {
            arch = "1200";
            smTarget = "sm_120";
            path = cudaToolkit;
            version = {
              complete = cudaPkgs.cudaMajorMinorVersion;
              major = cudaPkgs.cudaMajorVersion;
              minor = lib.lists.last (builtins.splitVersion cuda.version.complete);
            };
          };

          buildInputs = [
            cudaToolkit
            pkgs.stdenv.cc.cc.lib
            pkgs.xz
            pkgs.bzip2
          ];

          nativeBuildInputs =
            with pkgs;
            [
              llvmPkgs.clang-tools
              llvmPkgs.clang
              meson
              uv
              pkg-config
              doxygen
              graphviz

              ninja
              cmake

              texliveFull
              tex-fmt
            ]
            ++ [
              rustToolchain
            ];
        in
        {
          default = pkgs.mkShell {
            inherit buildInputs nativeBuildInputs;

            CPATH = lib.makeIncludePath [ cuda.path ];
            CUDA_HOME = cuda.path;
            CARGO_NET_OFFLINE = true;

            LD_LIBRARY_PATH = "${
              lib.makeLibraryPath (buildInputs ++ nativeBuildInputs)
            }:/run/opengl-driver/lib";

            shellHook = ''
                  if [ ! -e .clangd ]; then
                    cat > .clangd <<EOF
              CompileFlags:
                Compiler: ${cuda.path}/bin/nvcc
                Add:
                  - -std=c++20
                  - -xcuda
                  - --cuda-path=${cuda.path}
                  - -D__INTELLISENSE__
                  - -D__CLANGD__
                  - -I${cuda.path}/include
                  - -I$(pwd)/include
                  - -I$(pwd)/subprojects/cuco/include
                  - -I$(pwd)/subprojects/zlib-1.3.1
                  - -I$(pwd)/subprojects/googletest-1.17.0/googletest/include
                  - -D__LIBCUDAXX__STD_VER=${cuda.version.major}
                  - -D__CUDACC_VER_MAJOR__=${cuda.version.major}
                  - -D__CUDACC_VER_MINOR__=${cuda.version.minor}
                  - -D__CUDA_ARCH__=${cuda.arch}
                  - --cuda-gpu-arch=${cuda.smTarget}
                  - -D__CUDACC_EXTENDED_LAMBDA__
                  - -DPARAM_SWEEP_GROUP
                Remove:
                  - -Xcompiler=*
                  - -G
                  - "-arch=*"
                  - "-Xfatbin*"
                  - "-Xnvlink*"
                  - "-gencode*"
                  - "--generate-code*"
                  - "--generate-line-info"
                  - "--compiler-options*"
                  - "--expt-extended-lambda"
                  - "--expt-relaxed-constexpr"
                  - "-forward-unknown-to-host-compiler"
                  - "-Werror=cross-execution-space-call"

              Diagnostics:
                UnusedIncludes: None
                Suppress:
                  - variadic_device_fn
                  - attributes_not_allowed
                  - undeclared_var_use_suggest
                  - typename_invalid_functionspec
                  - expected_expression
                  - deduction_guide_target_attr
              EOF
                    echo ".clangd created by flake shellHook"
                  fi
            '';
          };
        }
      );
    };
}
