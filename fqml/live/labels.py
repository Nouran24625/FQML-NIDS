"""Feature windows + attack log -> labelled training table. Owner: Nouran.

Rules (written down so they can go in the video's honest note and the thesis):
  * A row of the ATTACKER host is labelled with the attack if at least `min_overlap` (default 50 %)
    of its window lies inside a logged attack run. Rows overlapping a run by less than that are
    AMBIGUOUS and dropped from training (start/end edges).
  * Every other row is Benign: normal devices, the attacker while idle, and the VICTIM's replies during
    an attack (the victim is not attacking; the model must learn not to flag it).
  * Each logged run is a "session" (session id = its index). The split in train.py keeps whole sessions
    together, so the model is always tested on attack runs it never saw (same idea as the project's
    capture-disjoint split, INTERFACES.md 1.2).

    python -m fqml.live.labels --pcap runs/lab/train.pcap --log runs/lab/attack_log.jsonl \
        --out runs/lab/windows.parquet [--window 2]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fqml.live import attack_log
from fqml.live.features import windows_from_packets
from fqml.live.pcap import iter_pcap_file

BENIGN = "Benign"
CANONICAL = [BENIGN, "DoS", "Recon", "BruteForce", "ModbusWrite"]


def label_rows(rows: pd.DataFrame, log: list[dict], min_overlap: float = 0.5) -> pd.DataFrame:
    df = rows.copy()
    df["label"] = BENIGN
    df["session"] = -1                       # -1 = benign background
    df["ambiguous"] = False
    w0 = df["window_start"].to_numpy()
    w1 = w0 + df["window"].to_numpy()
    for sid, e in enumerate(log):
        ov = (pd.Series(w1).clip(upper=e["end"]) - pd.Series(w0).clip(lower=e["start"])).clip(lower=0).to_numpy()
        frac = ov / df["window"].to_numpy()
        is_att = (df["host"] == e["attacker"]).to_numpy()
        hit = is_att & (frac >= min_overlap)
        edge = is_att & (frac > 0) & (frac < min_overlap)
        df.loc[hit, "label"] = e["label"]
        df.loc[hit, "session"] = sid
        df.loc[edge & (df["session"] == -1).to_numpy(), "ambiguous"] = True
    return df


def class_order(labels) -> list[str]:
    present = set(labels)
    if BENIGN not in present:
        raise ValueError("no Benign rows: capture normal traffic too")
    return [c for c in CANONICAL if c in present] + sorted(present - set(CANONICAL))


def build(pcap: str, log: str, window: float = 2.0, min_overlap: float = 0.5) -> pd.DataFrame:
    rows = pd.DataFrame(windows_from_packets(iter_pcap_file(pcap), window))
    if rows.empty:
        raise ValueError(f"{pcap}: no IPv4 packets")
    return label_rows(rows, attack_log.load(log), min_overlap)


def summary(df: pd.DataFrame) -> str:
    use = df[~df["ambiguous"]]
    lines = [f"{len(df):,} windows ({df['ambiguous'].sum()} ambiguous edge rows dropped)"]
    for lab, g in use.groupby("label"):
        n_sess = g.loc[g["session"] >= 0, "session"].nunique()
        lines.append(f"  {lab:12s} {len(g):6,} rows" + (f"  in {n_sess} attack runs" if lab != BENIGN else ""))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--min-overlap", type=float, default=0.5)
    a = ap.parse_args(argv)
    df = build(a.pcap, a.log, a.window, a.min_overlap)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out) if a.out.endswith(".parquet") else df.to_csv(a.out, index=False)
    print(summary(df))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
