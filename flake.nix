{
  description = "Bloom Filter";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    snowfall-lib = {
      url = "github:snowfallorg/lib";
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
