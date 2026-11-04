# Step 6: record the training dataset (lab only)

Goal: ~25-30 minutes of labelled lab traffic -> `runs/lab/train.pcap` + `runs/lab/attack_log.jsonl`,
the input to training (Step 7). Normal traffic runs the whole time; attacks are logged as they happen.

## One-time: make sure the lab is up and quiet
```
docker compose -f demo/lab/compose.yml up -d
del runs\lab\attack_log.jsonl        # remove any earlier test log (PowerShell)
```

## Run it (two terminals)

**Terminal 1 - record** (captures until you stop it, or pass minutes):
```
docker compose -f demo/lab/compose.yml run --rm ids sh demo/lab/tap/record.sh runs/lab/train.pcap 0
```
Leave this running for the whole session. It writes to `runs/lab/train.pcap` on the host.

**Let normal traffic run alone for ~10 minutes first** (so the model sees plenty of normal).

**Terminal 2 - attacks** (4 runs of each attack, 40 s each, 25 s normal gaps ~= 18 min):
```
docker exec lab-attacker python3 /app/attacks.py run --log /workspace/runs/lab/attack_log.jsonl --repeat 4 --seconds 40 --gap 25
```
More runs = a stronger model (training holds out whole runs to test on). 4 is a good minimum; use
`--repeat 5` if you have time. Fewer than 3 is not enough.
When it finishes, **let normal traffic run ~5 more minutes**, then stop Terminal 1 with Ctrl+C.

## Check what you captured
```
docker compose -f demo/lab/compose.yml run --rm ids python -m fqml.live.labels --pcap runs/lab/train.pcap --log runs/lab/attack_log.jsonl --out runs/lab/windows.parquet
```
This prints how many windows each class got. Good: Benign in the thousands, and each attack
(DoS, Recon, BruteForce, ModbusWrite) with **~60+ windows across at least 3 runs** (one 40 s run is
~19 windows). If a class is thin, run a few more of it (`attacks.py one <Label> --seconds 40`) while
still recording.

## Notes
- Keep every attack inside the lab (the runner only targets 10.10.0.x; the lab has no internet route).
- One continuous recording is simplest, but several files work too: label each, they merge in training.
- The .pcap can get large (tens to a few hundred MB). That's fine; it is not committed to git.
