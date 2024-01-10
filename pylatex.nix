{ buildPythonPackage, fetchPypi, ordered-set }:

buildPythonPackage rec {
  pname = "PyLaTeX";
  version = "1.4.2";
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-u3shvsV+zbo/b0TIVuvr32VJ/W6AZhvUT9UJQjZykkI=";
  };
  propagatedBuildInputs = [ ordered-set ];
}