#!/bin/sh
# Capture the lab network to a pcap file (lab only). Runs inside the ids container.
#   sh demo/lab/tap/record.sh [OUTPUT_PCAP] [MINUTES]
# Default: runs/lab/train.pcap, stop with Ctrl+C (or it stops after MINUTES if given).
set -e
OUT="${1:-runs/lab/train.pcap}"
MINUTES="${2:-0}"
IFACE=$(python demo/lab/tap/lab_iface.py)
mkdir -p "$(dirname "$OUT")"
echo "recording $IFACE -> $OUT   (Ctrl+C to stop)" >&2
if [ "$MINUTES" = "0" ]; then
  exec tcpdump -i "$IFACE" -n -U -s 160 -Z root -w "$OUT"
else
  echo "will stop after $MINUTES min" >&2
  exec timeout "$(( MINUTES * 60 ))" tcpdump -i "$IFACE" -n -U -s 160 -Z root -w "$OUT"
fi
