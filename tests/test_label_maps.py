"""Label map: the 8 families stay primary; variant c7m merges DDoS and DoS (Phase 0 amendment 5)."""
import numpy as np
import pytest

from fqml.data.label_maps import family_index, load_label_map


def test_label_map_primary_and_merged_variant():
    families, mapping = load_label_map("ciciot2023")
    assert families == ["Benign", "DDoS", "DoS", "Recon", "Web", "BruteForce", "Spoofing", "Mirai"]
    assert len(mapping) == 34
    f7, m7 = load_label_map("ciciot2023", "c7m")
    assert f7 == ["Benign", "DDoS_DoS", "Recon", "Web", "BruteForce", "Spoofing", "Mirai"]
    assert set(m7) == set(mapping)
    for native, fam in mapping.items():
        assert m7[native] == ("DDoS_DoS" if fam in ("DDoS", "DoS") else fam)
    native = ["BenignTraffic", "DDoS-SYN_Flood", "DoS-SYN_Flood", "XSS", "Mirai-udpplain"]
    assert list(family_index(native, "ciciot2023")) == [0, 1, 2, 4, 7]
    assert list(family_index(native, "ciciot2023", "c7m")) == [0, 1, 1, 3, 6]
    assert family_index(np.array(native), "ciciot2023").dtype == np.int64
    with pytest.raises(KeyError):
        load_label_map("ciciot2023", "no_such_variant")
    with pytest.raises(ValueError):
        family_index(["NotALabel"], "ciciot2023")
