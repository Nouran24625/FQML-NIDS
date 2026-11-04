"""Live-demo pipeline: pcap reader, features, labels, split, training, detector, dashboard.
Uses SYNTHETIC traffic only (fqml.live.synth); torch tests are skipped where torch is missing."""
import io
import json
import struct
import threading
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from fqml.live import attack_log, dashboard, detect, labels, synth, train
from fqml.live.features import FEATURES, FOOTPRINT, WindowAggregator, windows_from_packets
from fqml.live.model import Classifier, Preprocess
from fqml.live.pcap import ACK, PSH, SYN, iter_pcap_file, read_pcap


def _pcap(frames):
    """In-memory pcap built with the same writer the synthetic generator uses."""
    class _Mem(synth.PcapWriter):
        def __init__(self):
            self.f = io.BytesIO()
            self.f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    w = _Mem()
    for ts, fr in frames:
        w.write(ts, fr)
    w.f.seek(0)
    return w.f


def test_pcap_roundtrip_fields():
    f = synth.frame("10.0.0.1", "10.0.0.2", 6, 1234, 502, PSH | ACK, synth.modbus(6, 7))
    u = synth.frame("10.0.0.1", "10.0.0.3", 17, 5000, 53, 0, b"x" * 30)
    pk = list(read_pcap(_pcap([(100.5, f), (100.75, u)])))
    assert [p.proto for p in pk] == [6, 17]
    assert pk[0].src == "10.0.0.1" and pk[0].dport == 502 and pk[0].modbus_fc == 6 and pk[0].payload == 12
    assert abs(pk[0].ts - 100.5) < 1e-6 and pk[1].payload == 30 and pk[1].modbus_fc == -1


def test_not_pcap_is_rejected():
    with pytest.raises(ValueError):
        list(read_pcap(io.BytesIO(b"\x0a\x0d\x0d\x0a" + b"\0" * 40)))


def test_window_features_scan_vs_benign():
    frames = [(10.0 + i * 0.001, synth.frame("10.0.0.66", "10.0.0.2", 6, 5555, 1 + i, SYN)) for i in range(200)]
    frames += [(10.0 + i * 0.1, synth.frame("10.0.0.5", "10.0.0.2", 6, 4000, 1883, PSH | ACK, b"m" * 40)) for i in range(15)]
    frames.sort()
    rows = {r["host"]: r for r in windows_from_packets(read_pcap(_pcap(frames)), window=2.0)}
    assert set(FEATURES) <= set(rows["10.0.0.66"])
    assert rows["10.0.0.66"]["n_dst_ports"] == 200 and rows["10.0.0.66"]["syn_frac"] == 1.0
    assert rows["10.0.0.5"]["n_dst_ports"] == 1 and rows["10.0.0.5"]["payload_frac"] == 1.0
    assert "10.0.0.2" not in rows                        # hosts that only receive get no row
    assert set(FOOTPRINT) <= set(FEATURES) and len(FOOTPRINT) == 8


def test_aggregator_closes_windows_and_flushes():
    agg = WindowAggregator(2.0)
    def mk(t):
        return next(read_pcap(_pcap([(t, synth.frame("1.1.1.1", "2.2.2.2", 17, 1, 2))])))
    assert agg.feed(mk(0.5)) == []
    done = agg.feed(mk(4.1))                              # windows [0,2) and empty [2,4) close
    assert [t for t, _ in done] == [0.0, 2.0] and len(done[0][1]) == 1 and done[1][1] == []
    assert agg.flush(now=5.0) == [] and len(agg.flush(now=6.1)) == 1


def test_labels_attacker_victim_and_edges():
    rows = pd.DataFrame([{"host": h, "window_start": t, "window": 2.0, **{f: 1 for f in FEATURES}}
                         for t in (0, 2, 4, 6) for h in ("A", "V")])
    log = [{"label": "DoS", "attacker": "A", "start": 1.5, "end": 6.0}]
    df = labels.label_rows(rows, log)
    a = df[df.host == "A"].set_index("window_start")
    assert list(a["label"]) == ["Benign", "DoS", "DoS", "Benign"]
    assert bool(a.loc[0, "ambiguous"]) and not bool(a.loc[6, "ambiguous"])
    assert (df[df.host == "V"]["label"] == "Benign").all()                 # the victim is not attacking


def test_attack_log_wrapper(tmp_path):
    log = tmp_path / "log.jsonl"
    assert attack_log.main(["--log", str(log), "--label", "Recon", "--attacker", "1.2.3.4", "--", "true"]) == 0
    attack_log.main(["--log", str(log), "--label", "DoS", "--attacker", "1.2.3.4", "--start", "5", "--end", "9"])
    e = attack_log.load(log)
    assert [x["label"] for x in e] == ["Recon", "DoS"] and e[0]["end"] >= e[0]["start"]


def test_preprocess_range_and_constant_column():
    X = np.column_stack([np.arange(100.0), np.full(100, 3.0)])
    pre = Preprocess.fit(X, ["n_out", "syn_frac"])
    Z = pre.transform(np.array([[1e9, 3.0], [-5, 3.0]]))
    assert Z.min() >= 0 and Z.max() <= np.pi + 1e-6 and (Z[:, 1] == 0).all()


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    pcap, log = synth.generate(d, minutes=5, seed=1)
    df = labels.build(str(pcap), str(log))
    csv = d / "windows.csv"
    df.to_csv(csv, index=False)
    return d, pcap, csv, df


def test_session_split_keeps_runs_whole(lab):
    _, _, _, df = lab
    keep, held, untested = train.session_split(df, 0.34, seed=0)
    s_tr = set(df.loc[keep & (df.session >= 0), "session"])
    s_te = set(df.loc[held & (df.session >= 0), "session"])
    assert s_te and not (s_tr & s_te) and untested == []


def _serve(state):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(dashboard.Handler, state=state))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_rf_train_detect_dashboard_end_to_end(lab, tmp_path):
    d, pcap, csv, _ = lab
    bundle = tmp_path / "bundle"
    train.main(["--windows", str(csv), "--out", str(bundle), "--models", "rf", "--deploy", "rf"])
    rep = json.loads((bundle / "report.json").read_text())
    assert rep["models"]["rf"]["macro_f1"] > 0.5
    clf = Classifier(bundle)
    state = tmp_path / "live"
    det = detect.Detector(clf, state, threshold=0.6, source="test", hosts={}, attackers=[synth.HOSTS["attacker"]],
                          color=False)
    with open(pcap, "rb") as f:
        detect.run(f, det)
    det.close()
    wins = [json.loads(x) for x in (state / "detections.jsonl").read_text().splitlines()]
    assert len(wins) > 100 and sum(w["pkts"] for w in wins) == sum(1 for _ in iter_pcap_file(str(pcap)))
    alerts = [r for w in wins for r in w["rows"] if r["pred"] != "Benign" and r["conf"] >= 0.6]
    assert alerts and sum(r["host"] == synth.HOSTS["attacker"] for r in alerts) / len(alerts) > 0.9
    srv = _serve(state)
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "FQML-Guard" in page
        api = json.loads(urllib.request.urlopen(base + "/api/windows?from=3").read())
        assert api["from"] == 3 and api["next"] == len(wins) and len(api["windows"]) == len(wins) - 3
        assert json.loads(urllib.request.urlopen(base + "/api/status").read())["running"] is False
    finally:
        srv.shutdown()


def test_vqc_bundle_trains_and_predicts(lab, tmp_path):
    pytest.importorskip("torch")
    _, _, csv, df = lab
    bundle = tmp_path / "vqc"
    train.main(["--windows", str(csv), "--out", str(bundle), "--models", "vqc,mlp", "--deploy", "vqc",
                "--seeds", "1", "--steps", "150"])
    rep = json.loads((bundle / "report.json").read_text())
    assert rep["models"]["vqc"]["n_params"] == 2 * 8 * 3 + 2 * len(rep["classes"])
    assert abs(rep["models"]["mlp"]["n_params"] - rep["models"]["vqc"]["n_params"]) < 30
    clf = Classifier(bundle)
    P = clf.predict_proba(df.head(20))
    assert P.shape == (20, len(clf.classes)) and np.allclose(P.sum(1), 1, atol=1e-5)


def test_vqc_federated_option(lab, tmp_path):
    pytest.importorskip("torch")
    _, _, csv, _ = lab
    train.main(["--windows", str(csv), "--out", str(tmp_path / "fed"), "--models", "vqc", "--deploy", "vqc",
                "--seeds", "1", "--steps", "60", "--federated-sites", "3", "--rounds", "3"])
    meta = json.loads((tmp_path / "fed" / "bundle.json").read_text())
    assert meta["federated_sites"] == 3 and Classifier(tmp_path / "fed").kind == "vqc"
