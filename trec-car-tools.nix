{ lib, buildPythonPackage, fetchPypi, cbor }:
python3Packages.buildPythonPackage rec {
  pname = "trec-car-tools;
  version = "2.6";
  src = fetchPypi {
    inherit pname version;
    hash = "";
  };
  propagatedBuildInputs = [
    cbor
  ];
};