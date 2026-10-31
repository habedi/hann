{
  description = "Hann: a fast approximate nearest neighbor search library for Go";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
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
              go
              gcc
              gnumake
              golangci-lint
              go-tools
              gofumpt
              gotools
              pprof
              graphviz
              zig
              pre-commit
              uv
              python315
            ];

            # Keep the test runs reproducible by default. Override it in the shell to change it.
            HANN_SEED = "33";

            # Prevent an inherited GOROOT from conflicting with the Go toolchain provided by Nix.
            shellHook = ''
              unset GOROOT
            '';
          };
        }
      );
    };
}
