{lib, buildPythonPackage, fetchPypi }:

buildPythonPackage rec {
  pname = "warc3-wet-clueweb09";
  version = "0.2.5";
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-MFS/wH2lJdWWffjKMXX3j6P3hRTIJkP4yB+8qWMAuDY=";
  };
  propagatedBuildInputs = [ ];
}