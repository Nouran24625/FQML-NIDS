"""The post-quantum smoke test must keep passing."""
from fqml.pq.smoke import kem_test, sig_test


def test_ml_kem_768():
    kem_test()


def test_ml_dsa_65():
    sig_test()