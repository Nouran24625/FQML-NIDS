"""Checks on the lab's wiring (no Docker needed): isolation, fixed IPs, capture setup."""
import io
from pathlib import Path

import yaml

LAB = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((LAB / "compose.yml").read_text())


def test_lab_network_is_isolated_with_fixed_bridge_name():
    net = CFG["networks"]["lab"]
    assert net["internal"] is True                                   # no route out of the lab
    assert net["driver_opts"]["com.docker.network.bridge.name"] == "br-fqml"
    assert net["ipam"]["config"][0]["subnet"] == "10.10.0.0/24"


def test_every_lab_machine_is_only_on_the_lab_network_with_its_ip():
    want = {"broker": "10.10.0.40", "plc": "10.10.0.20", "camera": "10.10.0.11",
            "sensor": "10.10.0.12", "hmi": "10.10.0.30", "attacker": "10.10.0.66"}
    for svc, ip in want.items():
        nets = CFG["services"][svc]["networks"]
        assert list(nets) == ["lab"], f"{svc} must not join any other network"
        assert nets["lab"]["ipv4_address"] == ip
        assert "ports" not in CFG["services"][svc], f"{svc} must not publish ports"


def test_ids_listens_on_host_namespace_and_is_on_demand():
    ids = CFG["services"]["ids"]
    assert ids["network_mode"] == "host" and "NET_RAW" in ids["cap_add"]
    assert ids["profiles"] == ["ids"]


def test_watch_view_shows_lab_hosts(tmp_path):
    from fqml.live import synth, watch
    pcap, _ = synth.generate(tmp_path, minutes=1, seed=3)
    out = io.StringIO()
    w = watch.Watcher(2.0, {"10.10.0.30": "hmi", "10.10.0.20": "plc"}, out=out)
    from fqml.live.detect import run
    with open(pcap, "rb") as f:
        run(f, w)
    text = out.getvalue()
    assert w.windows >= 25 and "hmi 10.10.0.30" in text and "plc 10.10.0.20" in text
