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

        pythonOverrides = self: super: {
          pylatex = self.callPackage ./pylatex.nix {};
          trec-car-tools = self.callPackage ./trec-car-tools.nix {};
          ir_datasets = self.callPackage ./ir_datasets.nix {};
          unlzw3 = self.callPackage ./unlzw3.nix {};
          pyautocorpus = self.callPackage ./pyautocorpus.nix {};
          zlib-state = self.callPackage ./zlib-state.nix {};
          warc3-wet = self.callPackage ./warc3-wet.nix {};
          warc3-wet-clueweb09 = self.callPackage ./warc3-wet-clueweb09.nix {};
        };

        mkShell = target: (dspy-nix.lib.${system}.mkShell {
          inherit target;
          pythonOverrides = [ pythonOverrides ];
          packages = ps: with ps; [
            pydantic
            fuzzywuzzy
            nltk
            mypy
            jedi
            pylatex
            trec-car-tools
            ir_datasets
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