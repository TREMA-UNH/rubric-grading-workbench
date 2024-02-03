{lib, buildPythonPackage, fetchPypi, pcre }:

buildPythonPackage rec {
  pname = "pyautocorpus";
  version = "0.1.12";
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-FGkRsmXrTncenUlRb/ZakaTyVGTjZozqb4iF8tKYrVE=";
  };
  propagatedBuildInputs = [ pcre ];
}