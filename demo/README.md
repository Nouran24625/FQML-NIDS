# ITIDA prototype video: detection, dashboard and A8

Plan: `FQML-Guard_ITIDA_Video_Plan.pdf` (Version 2). Deadline: video link in the proposal, Sun 8 Nov 2026.

| What | Where | Video step |
|---|---|---|
| VQC vs matched MLP training on GPU | `demo/a8_vqc_train.py`, `demo/A8_vqc_gpu_colab.ipynb` | A8 (Version A and B) |
| Live detection pipeline | `fqml/live/` | B2–B7 |
| Dashboard (runs offline, no installs) | `python -m fqml.live.dashboard` | B2–B7 |
| Tests (synthetic traffic only) | `tests/test_live.py` | – |

Everything uses the existing image and libraries. **No change to `requirements/lock.txt`.** The dashboard
needs only Python's standard library, so it can also run outside Docker.

---

## Thursday: record A8 (Version A)

1. Open `demo/A8_vqc_gpu_colab.ipynb` in Colab, runtime **T4 GPU**.
2. Run the cells. The data cell checks the parquet's SHA-256 against `data/checksums.json`.
3. Rehearse once, then record the last cell. It prints: GPU, data and split, the 8 features, parameter
   counts (VQC 64, MLP of the same size), loss going down, then macro-F1 and per-family F1 for both.
4. Keep `results/demo/a8_vqc.json` with the video files. The numbers on screen must match it.

On screen it says **random row split**: the capture-disjoint split (DATA-3) isn't merged yet. Say
"current results", never "final".

---

## What the detection side needs from the lab

1. **A pcap stream of the lab segment**, readable from the laptop running the detector:
   - EVE-NG: capture on the lab bridge on the EVE host (that's what EVE's own capture does), e.g.
     `ssh root@<eve> "tcpdump -i <vunl/vnet iface> -n -U -s 128 -w - not port 22"`.
     A sensor *node* on a normal EVE network only sees its own traffic unless the switch mirrors ports.
   - Docker lab: `sudo tcpdump -i br-<lab network id> -n -U -s 128 -w -` on the laptop.
2. **Fixed IPs**: attacker, PLC, HMI, IoT devices. They go in `--hosts` for the dashboard labels.
3. **Same clock** for the attacker node and the capture (NTP, or log the attacks on the capture host).
4. **Every training attack started through the logger**, so labels come from what really happened:

```bash
L="python -m fqml.live.attack_log --log runs/lab/attack_log.jsonl --attacker 10.10.0.66"
$L --label DoS        -- hping3 -S -p 80 -i u2000 -c 15000 10.10.0.11                  # ~500 pkt/s, NOT --flood
$L --label Recon      -- nmap -sS -T4 -p 1-2000 10.10.0.0/24
$L --label BruteForce -- hydra -l admin -P small_wordlist.txt -t 4 ssh://10.10.0.11
$L --label ModbusWrite --timeout 40 -- sh -c 'while true; do mbpoll -m tcp -1 -r 1 -t 4 10.10.0.20 $RANDOM; sleep 0.3; done'
```

If the attack tool isn't on the logger's machine, log by hand right after:
`$L --label Recon --start <epoch> --end <epoch>`. Check the tool flags on the lab images before Friday.

Why rate-limited DoS: `hping3 --flood` can exceed what the capture and a Python reader keep up with, and
can freeze small VMs. ~500–2000 pkt/s is clearly a flood for the model and stays stable on camera.

---

## Friday: record training traffic and train the demo model

**Recording (about 35 minutes):**
- Start the capture to a file: `tcpdump ... -w runs/lab/train.pcap` (same command as above, with a file
  instead of `-w -`).
- 10 minutes of **normal traffic only** (HMI reading the PLC, cameras, sensors).
- Then **3 runs of each attack**, 30–60 s each, with different settings each time (rate, ports, targets),
  and at least 30 s of normal traffic between runs.
- Another 5 minutes of normal traffic at the end.

**Label and train:**
```bash
python -m fqml.live.labels --pcap runs/lab/train.pcap --log runs/lab/attack_log.jsonl --out runs/lab/windows.parquet
python -m fqml.live.train  --windows runs/lab/windows.parquet --out models/lab_demo --deploy vqc
#   optional: --federated-sites 3 --rounds 30   (the VQC is trained by 3 simulated sites with our FedAvg)
```
`train` prints, for the VQC, the matched MLP and a random-forest reference: test macro-F1, per-class F1
and benign false-positive rate. The test set is **whole attack runs the model never saw**, plus held-out
normal time blocks. Everything is in `models/lab_demo/report.json`.

**Which model goes in the video** (team decision Friday night, based on `report.json`):
- If the VQC detects every attack type on the held-out runs with few false alarms, deploy it (default).
- If it doesn't, either deploy the MLP or RF and **say so** in B1/B8 ("our classical model; the quantum
  model is still being improved"), or drop the attack type it misses from the video. Never fake it.

---

## Saturday: the live demo (B1–B7)

Three windows on screen: attacker console, detector terminal, dashboard in the browser.

```bash
# 1. dashboard (standard library only; works on the host or in the container)
python3 -m fqml.live.dashboard --state runs/live          # open http://localhost:8050

# 2. detector (container has torch for the VQC); -T so the pcap stream can be piped in
<capture command from above> | docker compose -f docker/compose.yml exec -T dev \
  python -m fqml.live.detect --stdin --bundle models/lab_demo --state runs/live \
    --hosts 10.10.0.66=attacker,10.10.0.20=PLC,10.10.0.30=HMI,10.10.0.11=camera,10.10.0.12=sensor \
    --attacker 10.10.0.66
```

`--attacker` is lab ground truth used **only** by the dashboard to count false alarms (alerts from any
other host). The model never sees it.

**Rehearse without the lab:** record a short demo capture, then replay it at real speed:
`python -m fqml.live.detect --pcap runs/lab/demo.pcap --realtime --bundle models/lab_demo --state runs/live ...`
The dashboard then shows **REPLAY**. A replay is only shown in the video if we say it's a replay.

**Order for recording** (matches B2–B7): 30 s normal → DoS (stop it, wait for the banner to go green) →
port scan → brute force → Modbus write (only if it works) → summary.

---

## Dry run with synthetic traffic (pipeline check only, never for the video)

```bash
python -m fqml.live.synth  --out runs/synth --minutes 6
python -m fqml.live.labels --pcap runs/synth/SYNTHETIC_lab.pcap --log runs/synth/SYNTHETIC_attack_log.jsonl --out runs/synth/windows.csv
python -m fqml.live.train  --windows runs/synth/windows.csv --out runs/synth/bundle
python -m fqml.live.detect --pcap runs/synth/SYNTHETIC_lab.pcap --realtime --bundle runs/synth/bundle --state runs/live --attacker 10.10.0.66
```

## How it works (for B1/B8 and questions from ITIDA)

- Traffic is cut into **2-second windows**. For each host that sent packets, the detector computes 23
  counts and ratios (no IPs, ports or timestamps as features).
- The model uses **8 of them, one per qubit**, fixed before seeing lab data, each aimed at one attack
  footprint: packet volume, SYN share, distinct ports, refused connections, new connections, focus on one
  service, share of packets with data, Modbus writes. (`--features mrmr` instead picks 8 by mutual
  information on the training rows.)
- Preprocessing is fitted on training rows only (log, 1st–99th percentile clip, scale to [0, π]), the
  same rules as `docs/INTERFACES.md` §1.3.
- The VQC is the project's model: 8 qubits, 3 layers, CNOT chain, circuit v2, on our PyTorch simulator.

## Known limits (said in the video's honest note)

- The network is emulated (EVE-NG or Docker), the PLC is a software PLC, the demo model was trained on
  this lab's traffic, and the quantum model runs on a simulator.
- Lab traffic is far simpler than a real plant; scores on it say nothing about field performance.
