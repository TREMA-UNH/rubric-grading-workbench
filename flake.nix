{
  description = "ExamPP";

  # inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/ffb2e65e054b989c55a83f79af0ed4b912e22e14";
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

        devShells.default = self.outputs.devShells.${system}.cuda;
        devShells.cpu = mkShell "cpu";
        devShells.rocm = mkShell "rocm";
        devShells.cuda = mkShell "cuda";
      }
    );
}


        # checks.default = pkgs.stdenv.mkDerivation {
        #   name = "check-exampp";
        #   src = ./.;
        #   nativeBuildInputs =
        #     let py = dspy-nix.outputs.packages.${system}.python-cuda.withPackages (ps: [ (ps.callPackage ./exampp.nix {}) ]);
        #     in [py];
        #   buildPhase = '' python scripts/minimal_tests.py '';
        # };