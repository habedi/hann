{
  description = "Hann: a fast approximate nearest neighbor search library for Go";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      # The dev shell is defined for the x86_64 systems used for local
      # development. The library itself also builds and runs on arm64 through
      # the NEON kernel variants, and an arm64 CI cell tests it there.
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

              # Cross-compiling the cgo code (zig cc is a hermetic C cross-toolchain)
              zig

              # Git hooks, and the dataset download script
              pre-commit
              uv
            ];

            # Keep the test runs reproducible by default. Override it in the
            # shell to change it.
            HANN_SEED = "33";
          };
        }
      );

      # No packages output: Hann is a library, and the programs under
      # example/cmd are meant to be run from the dev shell with make.
    };
}
