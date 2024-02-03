{lib, buildPythonPackage, fetchPypi, setuptools }:

buildPythonPackage rec {
  pname = "unlzw3";
  version = "0.2.2";
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-0DeptoI9GkVdbeHgJYr4wPXb9Aq6OxnMUURI542ncGI=";
  };
  format = "pyproject";
  buildInputs = [ setuptools ];
  propagatedBuildInputs = [ ];
}