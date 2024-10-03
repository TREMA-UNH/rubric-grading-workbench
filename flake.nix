{
  description = "ExamPP";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.dspy-nix.url = "git+https://git.smart-cactus.org/ben/dspy-nix";

  outputs = inputs@{ self, nixpkgs, flake-utils, dspy-nix, ... }:
    flake-utils.lib.eachDefaultSystem (system: 
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonOverrides = self: super: {
          pylatex = self.callPackage ./nix/pylatex.nix {};
          trec-car-tools = self.callPackage ./nix/trec-car-tools.nix {};
          ir_datasets = self.callPackage ./nix/ir_datasets.nix {};
          unlzw3 = self.callPackage ./nix/unlzw3.nix {};
          pyautocorpus = self.callPackage ./nix/pyautocorpus.nix {};
          zlib-state = self.callPackage ./nix/zlib-state.nix {};
          warc3-wet = self.callPackage ./nix/warc3-wet.nix {};
          warc3-wet-clueweb09 = self.callPackage ./nix/warc3-wet-clueweb09.nix {};
          exampp = self.buildPythonPackage {
            name = "exampp";
            src = ./.;
            format = "setuptools";
            propagatedBuildInputs = with self; [ 
              pydantic
              nltk
              ir_datasets
              fuzzywuzzy
            ];
            doCheck = false;
          };
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
          nativeBuildInputs = oldAttrs.nativeBuildInputs;
        });

      in {
        lib.pythonOverrides = pythonOverrides;
        packages.exampp = (pkgs.python3.override {packageOverrides = pythonOverrides;}).pkgs.exampp;

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
