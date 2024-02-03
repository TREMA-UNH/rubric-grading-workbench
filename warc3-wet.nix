{lib, buildPythonPackage, fetchPypi }:

buildPythonPackage rec {
  pname = "warc3-wet";
  version = "0.2.3";
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-1Dck83Ltu8eZC5w4Skk/tggYx8muYp2Owxwpy7zNAbk=";
  };
  doCheck = false;
  propagatedBuildInputs = [ ];
}