"""Record and verify SHA-256 fingerprints of data folders -> data/checksums.json (committed).

    python -m fqml.data.checksums record raw/unsw_nb15     # after downloading
    python -m fqml.data.checksums record interim           # after cleaning
    python -m fqml.data.checksums verify                   # check every recorded folder
    python -m fqml.data.checksums verify interim           # check only some folders
Folder names are relative to data/. Only data/checksums.json goes to git, never the data.
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DB = DATA / "checksums.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB at a time
            h.update(chunk)
    return h.hexdigest()


def load():
    return json.loads(DB.read_text()) if DB.exists() else {}


def record(name, source=""):
    folder = DATA / name
    files = sorted(p for p in folder.rglob("*") if p.is_file() and not p.name.startswith("."))
    if not files:
        sys.exit(f"No files found in {folder}")
    entry = {"recorded": datetime.date.today().isoformat(), "source": source,
             "n_files": len(files),
             "total_gb": round(sum(p.stat().st_size for p in files) / 1e9, 3), "files": {}}
    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {p.relative_to(DATA)}", flush=True)
        entry["files"][str(p.relative_to(DATA))] = sha256(p)
    db = load()
    db[name] = entry
    DB.write_text(json.dumps(db, indent=2, sort_keys=True))
    print(f"Recorded {len(files)} files ({entry['total_gb']} GB) for '{name}'")


def verify(names=None):
    db, all_ok = load(), True
    if not db:
        sys.exit("data/checksums.json is empty")
    for name, entry in db.items():
        if names and name not in names:
            continue
        bad = [f for f, h in entry["files"].items()
               if not (DATA / f).exists() or sha256(DATA / f) != h]
        all_ok &= not bad
        print(f"{'OK ' if not bad else 'BAD'} {name}: "
              f"{entry['n_files'] - len(bad)}/{entry['n_files']} files match")
        for f in bad[:10]:
            print(f"     missing or changed: {f}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "record":
        record(args[1], source=" ".join(args[2:]))
    elif args and args[0] == "verify":
        verify(args[1:] or None)
    else:
        sys.exit(__doc__)
