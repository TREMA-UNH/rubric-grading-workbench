{ stdenv, fetchFromGitHub }:

stdenv.mkDerivation {
  name = "trec-eval";
  src = fetchFromGitHub {
    owner = "usnistgov";
    repo = "trec_eval";
    rev = "a5211566d0c9e2ec337bacf327b9350ab5b3edde";
    sha256 = "10v9hg9919maz4x088jzv9vb4mhv4qij85kmz3i2vgk5aphnwzp2";
  };
  installPhase = ''
    mkdir -p $out/bin
    cp trec_eval $out/bin
  '';
}
