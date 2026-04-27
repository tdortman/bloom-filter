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
          llvm = pkgs.llvmPackages_22;

          rustToolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [ "rust-src" ];
          };

          cudaPackages = with cudaPkgs; [
            cuda_nvcc
            cuda_cudart
            cuda_cccl
            cuda_cuobjdump
          ];

          cuda = {
            arch = "1200";
            smTarget = "sm_120";
            path = pkgs.symlinkJoin {
              name = "cuda-dev-path";
              paths = cudaPackages;
            };
            version = {
              complete = cudaPkgs.cudaMajorMinorVersion;
              major = cudaPkgs.cudaMajorVersion;
              minor = lib.lists.last (builtins.splitVersion cuda.version.complete);
            };
          };

          buildInputs = cudaPackages ++ [
            pkgs.stdenv.cc.cc.lib
          ];

          nativeBuildInputs =
            with pkgs;
            [
              llvm.clang-tools
              llvm.lldb
              meson
              uv
              pkg-config
              doxygen
              graphviz

              cudaPkgs.nsight_systems
              cudaPkgs.nsight_compute

              rust-analyzer
              ninja
              llvm.clang
              mold-unwrapped
            ]
            ++ [
              rustToolchain
            ];
        in
        {
          default = pkgs.mkShell {
            inherit buildInputs nativeBuildInputs;

            CPATH = lib.makeIncludePath [ cuda.path ];

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
                  - -I$(pwd)/subprojects/googletest-1.17.0/googletest/include
                  - -D__LIBCUDAXX__STD_VER=${cuda.version.major}
                  - -D__CUDACC_VER_MAJOR__=${cuda.version.major}
                  - -D__CUDACC_VER_MINOR__=${cuda.version.minor}
                  - -D__CUDA_ARCH__=${cuda.arch}
                  - --cuda-gpu-arch=${cuda.smTarget}
                  - -D__CUDACC_EXTENDED_LAMBDA__
                Remove:
                  - -Xcompiler=*
                  - -G
                  - "-arch=*"
                  - "-Xfatbin*"
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
