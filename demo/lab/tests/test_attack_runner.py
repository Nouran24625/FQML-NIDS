"""Attack-runner checks that need no lab: command shapes, safety (lab-only targets, rate caps),
and that the log entries the runner writes are consumed correctly by the labeller.
The attacks themselves run against the live lab; here we only check the parts that can run offline.
"""
import importlib.util
import random
import sys
from pathlib import Path

import pandas as pd

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "nodes"))                       # modbus
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root for fqml.*

spec = importlib.util.spec_from_file_location("attacks", LAB / "attacker" / "attacks.py")
attacks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attacks)

LAB_HOSTS = {"10.10.0.11", "10.10.0.12", "10.10.0.20", "10.10.0.30", "10.10.0.40", "10.10.0.66"}


def test_commands_target_only_lab_hosts():
    rng = random.Random(0)
    for cmd in (attacks.dos_cmd(20, rng), attacks.recon_cmd(20, rng), attacks.brute_cmd(20, rng, "wl.txt")):
        target = cmd[-1]                                      # may be "10.10.0.11" or "ssh://10.10.0.11"
        ip = target.split("//")[-1].split("/")[0]            # strip scheme and any CIDR suffix
        assert ip.startswith("10.10.0."), f"{cmd[0]} targets {target}, outside the lab"
    # recon scans the lab subnet, never a wider range
    assert attacks.recon_cmd(20, rng)[-1] == "10.10.0.0/24"


def test_recon_rate_is_capped():
    """The scan is paced so it does not out-flood the DoS or overrun the capture."""
    cmd = attacks.recon_cmd(20, random.Random(0))
    assert "--max-rate" in cmd
    assert int(cmd[cmd.index("--max-rate") + 1]) <= 2500      # packets/s, well below a flood


def test_dos_is_rate_limited_not_flood():
    rng = random.Random(1)
    cmd = attacks.dos_cmd(20, rng)
    assert "--flood" not in cmd                               # never the uncapped flood
    i = cmd.index("-i")                                       # -i uN caps the packet rate
    us = int(cmd[i + 1].lstrip("u"))
    assert 500 <= us <= 2000                                  # ~500-2000 packets/s


def test_brute_uses_small_wordlist_and_few_threads():
    cmd = attacks.brute_cmd(20, random.Random(0), "/app/wordlist.txt")
    assert cmd[0] == "hydra" and "-t" in cmd
    assert int(cmd[cmd.index("-t") + 1]) <= 8                 # gentle, so it is a demo not a real crack
    assert "-f" not in cmd                                    # no early stop: the attack must last
    wl = (LAB / "attacker" / "wordlist.txt").read_text().split()
    assert "admin2025" in wl and len(wl) <= 50                # the real password is present, list is short


def test_run_logged_fills_the_requested_duration():
    """A fast tool is repeated until the duration is reached, so scans/brute force give enough windows."""
    logs = []
    e = attacks.run_logged("Test", ["true"], 1.0, None, lambda log, entry: logs.append(entry), timeout_pad=1)
    assert e["end"] - e["start"] >= 1.0 and len(logs) == 1    # one log line spanning the whole run
    assert e["label"] == "Test" and e["attacker"] == attacks.ATTACKER


def test_log_entries_feed_the_labeller(tmp_path):
    """The runner's log lines must be exactly what fqml.live.labels expects."""
    from fqml.live import attack_log, labels
    from fqml.live.features import FEATURES
    log = tmp_path / "attack_log.jsonl"
    # entries shaped like the runner writes (run_logged / modbus_write_burst)
    attack_log.append(log, {"label": "DoS", "attacker": "10.10.0.66", "target": "10.10.0.11",
                            "start": 1000.0, "end": 1020.0, "command": "hping3 ...", "exit_code": 0})
    attack_log.append(log, {"label": "ModbusWrite", "attacker": "10.10.0.66", "target": "10.10.0.20",
                            "start": 1100.0, "end": 1130.0, "command": "modbus write burst", "writes": 90})
    loaded = attack_log.load(log)
    assert [e["label"] for e in loaded] == ["DoS", "ModbusWrite"]
    # a window from the attacker during the DoS gets labelled DoS; outside, Benign
    rows = pd.DataFrame([
        {"host": "10.10.0.66", "window_start": 1000.0, "window": 2.0, **{f: 1 for f in FEATURES}},
        {"host": "10.10.0.66", "window_start": 1050.0, "window": 2.0, **{f: 1 for f in FEATURES}},
        {"host": "10.10.0.30", "window_start": 1100.0, "window": 2.0, **{f: 1 for f in FEATURES}},
    ])
    out = labels.label_rows(rows, loaded)
    assert out.iloc[0]["label"] == "DoS"                     # attacker during the DoS run
    assert out.iloc[1]["label"] == "Benign"                  # attacker between runs
    assert out.iloc[2]["label"] == "Benign"                  # the HMI is never the attacker


def test_full_recording_gives_a_trainable_split(tmp_path):
    """A multi-run recording must yield several sessions per attack, so whole runs can be held out."""
    from fqml.live import synth, labels
    from fqml.live.train import session_split
    pcap, log = synth.generate(tmp_path, minutes=8, seed=5, sessions_per_attack=4, attack_seconds=(35, 45))
    df = labels.build(str(pcap), str(log))
    use = df[~df["ambiguous"]]
    counts = use["label"].value_counts()
    assert counts.get("Benign", 0) > 500
    for lab in ("DoS", "Recon", "BruteForce", "ModbusWrite"):
        assert counts.get(lab, 0) >= 40, f"{lab} too thin: {counts.get(lab, 0)}"
    keep, held, untested = session_split(use.reset_index(drop=True), 0.34, seed=0)
    assert not untested, f"some attacks have too few runs to test on: {untested}"
    assert held.sum() > 0 and keep.sum() > 0
