# Step 8: live detection + dashboard (lab only)

The trained model (models/lab_demo, the VQC) classifies lab traffic live. The detector runs in the ids
node; the dashboard runs in its own container that publishes a port, so you open it in the browser on
the computer running Docker.

## Start (two terminals + a browser)

**Terminal 1 - dashboard** (open http://localhost:8050 after it starts):
```
docker compose -f demo/lab/compose.yml --profile dash up -d dashboard
```

**Terminal 2 - live detector** (captures br-fqml, runs the model, writes runs/live/):
```
docker compose -f demo/lab/compose.yml run --rm ids sh demo/lab/tap/detect.sh
```
It prints one line per window ("ok ... all Benign", or a red ALERT line). The dashboard updates within
a second or two.

## See it detect
In a third terminal, run an attack and watch the dashboard turn red:
```
docker exec lab-attacker python3 /app/attacks.py one DoS --seconds 20
docker exec lab-attacker python3 /app/attacks.py one Recon --seconds 20
docker exec lab-attacker python3 /app/attacks.py one ModbusWrite --seconds 20
docker exec lab-attacker python3 /app/attacks.py one BruteForce --seconds 20
```
The dashboard shows: a red/green banner, detections per attack type, a timeline, recent alerts, and a
false-alarm count (alerts from any host that is not the attacker). It names the model (VQC) in the footer.

## Stop
- Detector: Ctrl+C in Terminal 2.
- Dashboard: `docker compose -f demo/lab/compose.yml --profile dash down`

## Notes
- `--attacker 10.10.0.66` is only used by the dashboard to count false alarms; the model never sees it.
- Rehearse without touching the lab by replaying a recording instead of live capture:
  `... run --rm ids python -m fqml.live.detect --pcap runs/lab/train.pcap --realtime --bundle models/lab_demo --state runs/live --hosts ... --attacker 10.10.0.66`
  (the dashboard then shows REPLAY, which you only show in the video if you say it is a replay).
- BruteForce is the hardest class (see report.json). If it is missed on camera, cut that attack or say
  it is subtler for the current model. Never fake a detection.
