#!/bin/sh
# Live detection on the lab network (lab only). Runs inside the ids container.
# Captures br-fqml and feeds it to the trained model; writes detections to runs/live/ for the dashboard.
#   sh demo/lab/tap/detect.sh [extra args passed to fqml.live.detect]
set -e
IFACE=$(python demo/lab/tap/lab_iface.py)
HOSTS="10.10.0.11=camera,10.10.0.12=sensor,10.10.0.20=plc,10.10.0.30=hmi,10.10.0.40=broker,10.10.0.66=attacker,10.10.0.1=docker-gw"
echo "detecting on $IFACE with model models/lab_demo" >&2
mkdir -p runs/live
tcpdump -i "$IFACE" -n -U -s 160 -Z root -w - 2>/dev/null \
  | python -m fqml.live.detect --stdin --bundle models/lab_demo --state runs/live \
      --hosts "$HOSTS" --attacker 10.10.0.66 "$@"
