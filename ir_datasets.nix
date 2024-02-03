{lib, buildPythonPackage, fetchPypi 
    , beautifulsoup4
    ,inscriptis
    ,lxml
    ,numpy
    ,pyyaml
    ,requests
    ,tqdm
    ,trec-car-tools
    ,lz4
    ,warc3-wet
    ,warc3-wet-clueweb09
    ,zlib-state
    ,ijson
    ,pyautocorpus
    ,unlzw3 
}:

buildPythonPackage rec {
  pname = "ir_datasets";
  version = "0.5.5";
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-gMZbJJK7zIqwNLFEuNOHJPYoTF27j+NsAcFumtsosOo=";
  };
  propagatedBuildInputs = [ 
    beautifulsoup4
    inscriptis
    lxml
    numpy
    pyyaml
    requests
    tqdm
    trec-car-tools
    lz4
    warc3-wet
    warc3-wet-clueweb09
    zlib-state
    ijson
    pyautocorpus
    unlzw3
   ];
  preCheck = ''
    HOME="`pwd`/tmp"
    mkdir $HOME
  '';
  meta = with lib; {
    description = "ir_datasets is a python package that provides a common interface to many IR ad-hoc ranking benchmarks, training datasets, etc.";
    homepage = "https://github.com/allenai/ir_datasets";
  };
}