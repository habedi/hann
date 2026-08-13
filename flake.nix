{
  description = "Hann: a fast approximate nearest neighbor search library for Go";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      # The distance functions are compiled with -mavx and -mavx2, so the build
      # is x86_64 only.
      supportedSystems = [ "x86_64-linux" "x86_64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              # Go toolchain and build tools. (The core package uses cgo for the
              # SIMD distance functions, so a C compiler is part of the toolchain
              # here.)
              go
              gcc
              gnumake

              # Linting and formatting
              golangci-lint
              go-tools
              gofumpt
              gotools

              # Profiling the examples and the benchmarks
              pprof
              graphviz

              # Git hooks, and the dataset download script
              pre-commit
              uv
            ];

            # Keep the test runs reproducible and quiet by default. Override
            # either one in the shell to change it.
            HANN_SEED = "33";
            HANN_LOG = "1";
          };
        }
      );

      # No packages output: Hann is a library, and the programs under
      # example/cmd are meant to be run from the dev shell with make.
    };
}
