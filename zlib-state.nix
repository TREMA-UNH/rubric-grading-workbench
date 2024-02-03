{lib, buildPythonPackage, fetchPypi, zlib }:

buildPythonPackage rec {
  pname = "zlib-state";
  version = "0.1.6";
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-3qbMx+NcMUzbr5xDnUeQtX6pLdX7gFNOHUbHL112mK4=";
  };
  propagatedBuildInputs = [ zlib ];
}