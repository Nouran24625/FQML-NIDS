# FQML-Guard demo lab

A small industrial network in Docker, used to show FQML-NIDS detecting attacks live. Everything runs on
one laptop, isolated from the internet. This is the live demo behind the ITIDA prototype video.

## What it is
- **Six machines on one isolated network** (10.10.0.0/24): a software PLC running a water-tank process
  over Modbus, an IP camera (web + SSH), an MQTT sensor and broker, an operator HMI, and an attacker.
- **Normal traffic** from the HMI reading the PLC, camera snapshots and MQTT, all the time.
- **Four attacks** from the attacker, each logged so the traffic can be labelled: DoS, Recon,
  BruteForce, ModbusWrite.
- **A sensor (IDS) node** that sees every packet, runs the trained quantum model, and feeds a
  **dashboard** you open in the browser. Detection only (it alerts; it does not block).

## Quick start (lab must be built once; see the steps below)
```
# 1. lab + dashboard
docker compose -f demo/lab/compose.yml up -d
docker compose -f demo/lab/compose.yml --profile dash up -d dashboard      # http://localhost:8050
# 2. live detector (leave running)
docker compose -f demo/lab/compose.yml run --rm ids sh demo/lab/tap/detect.sh
# 3. attacks (another terminal)
docker exec lab-attacker python3 /app/attacks.py one DoS --seconds 20
# stop
docker compose -f demo/lab/compose.yml --profile dash down
```

## The guides, in order
| File | Step |
|---|---|
| this README | overview + quick start |
| `RECORDING.md` | record the training dataset (normal + logged attacks) |
| `TRAINING.md` | train the VQC / MLP / RF and read the honest report |
| `LIVE.md` | run live detection + dashboard, drive attacks |
| `tap/watch.sh` | live per-host traffic view (no model), to check the lab/attacks |

## Parts
- `compose.yml` - the whole lab, one isolated network, on-demand ids and dashboard services
- `nodes/` - PLC, camera, sensor, HMI (standard library)
- `attacker/attacks.py` - the logged attack runner (standard tools, lab targets only)
- `tap/` - capture helpers (watch, record, detect) run inside the ids node
- model code and dashboard are the project's `fqml/live/` package

## Rules (kept throughout)
- The attacker reaches only the lab; the network has no internet route.
- Train and test on different attack runs; numbers are shown exactly as measured.
- `runs/` (captures) and `models/` (trained models) are not committed - back them up separately.
