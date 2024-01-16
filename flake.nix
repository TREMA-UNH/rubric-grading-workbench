{
  description = "ExamPP";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.dspy-nix.url = "git+https://git.smart-cactus.org/ben/dspy-nix";

  outputs = inputs@{ self, nixpkgs, flake-utils, dspy-nix, ... }:
    flake-utils.lib.eachDefaultSystem (system: 
      let
        pkgs = nixpkgs.legacyPackages.${system};
        mkShell = target: (dspy-nix.lib.${system}.mkShell {
          inherit target;
          packages = ps: with ps; [
            pydantic
            fuzzywuzzy
            nltk
            mypy
            jedi
            (ps.callPackage ./pylatex.nix {})
            (ps.callPackage ./trec-car-tools.nix {})
          ];
        }).overrideAttrs (oldAttrs: {
          nativeBuildInputs = oldAttrs.nativeBuildInputs ++ [self.outputs.packages.${system}.trec-eval];
        });

      in {
        packages.trec-eval = pkgs.callPackage ./trec-eval.nix {};

        devShells.default = self.outputs.devShells.${system}.cpu;
        devShells.cpu = mkShell "cpu";
        devShells.rocm = mkShell "rocm";
        devShells.cuda = mkShell "cuda";
      }
    );
}
