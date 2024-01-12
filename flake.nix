{
  description = "ExamPP";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.dspy-nix.url = "git+https://git.smart-cactus.org/ben/dspy-nix";

  outputs = inputs@{ self, nixpkgs, flake-utils, dspy-nix, ... }:
    flake-utils.lib.eachDefaultSystem (system: 
      let pkgs = nixpkgs.legacyPackages.${system}; in {
        devShells.default = dspy-nix.lib.${system}.mkShell {
          target = "cuda";
          packages = ps: with ps; [
            pydantic
            fuzzywuzzy
            nltk
            mypy
            jedi
            (ps.callPackage ./pylatex.nix {})
          ];
        };
      }
    );
}
