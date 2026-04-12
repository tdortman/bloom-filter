{
  description = "Bloom Filter";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    snowfall-lib = {
      url = "github:snowfallorg/lib";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    inputs.snowfall-lib.mkFlake {
      inherit inputs;
      src = ./.;

      snowfall = {
        root = ./nix;
        namespace = "bloom";
      };

      overlays = [
        inputs.rust-overlay.overlays.default
      ];

      channels-config = {
        allowUnfree = true;
      };

      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      alias = {
        packages.default = "bloom";
      };
    };
}
