{lib, buildPythonPackage, fetchPypi, cbor, numpy }:

buildPythonPackage rec {
  pname = "trec-car-tools";
  version = "2.6";
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-L84t4SAiT9VpsVHVvtNYpO0zTmQ4ibnj3+Plo9FdIcg=";
  };
  propagatedBuildInputs = [ cbor numpy ];
  meta = with lib; {
    description = "Python tools for TREC CAR";
    homepage = "https://trec-car.cs.unh.edu"; 
    license = licenses.cc-by-sa-40;
  };
}