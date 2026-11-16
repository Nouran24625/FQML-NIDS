"""Retired fingerprints are kept as a record but no longer verified (Phase 0 amendment 1)."""
import json

import pytest

from fqml.data import checksums


def test_retired_fingerprints_are_kept_but_not_verified(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    (data / "interim").mkdir(parents=True)
    (data / "raw").mkdir()
    (data / "interim" / "old.parquet").write_bytes(b"old")
    (data / "raw" / "a.csv").write_bytes(b"a")
    monkeypatch.setattr(checksums, "DATA", data)
    monkeypatch.setattr(checksums, "DB", data / "checksums.json")
    checksums.record("interim")
    checksums.record("raw")
    checksums.retire("interim", "replaced by cleaner v1")
    (data / "interim" / "old.parquet").unlink()               # the retired file is gone from this machine
    with pytest.raises(SystemExit) as e:
        checksums.verify()
    assert e.value.code == 0 and "retired" in capsys.readouterr().out
    db = json.loads((data / "checksums.json").read_text())
    assert "replaced by cleaner v1" in db["interim"]["retired"] and db["interim"]["files"]
    with pytest.raises(SystemExit) as e:                       # naming it still checks it
        checksums.verify(["interim"])
    assert e.value.code == 1
    with pytest.raises(SystemExit):
        checksums.retire("not_recorded", "x")


def test_repo_checksums_mark_v0_as_retired():
    db = checksums.load()
    assert "retired" in db["interim"] and "v0" in db["interim"]["retired"]
    assert all("retired" not in v for k, v in db.items() if k.startswith("raw/"))
