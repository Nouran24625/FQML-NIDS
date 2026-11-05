#!/bin/sh
# Live view of lab traffic, one line per host every 2 s (lab only). Stop with Ctrl+C.
set -e
IFACE=$(python demo/lab/tap/lab_iface.py)
echo "capturing on $IFACE" >&2
tcpdump -i "$IFACE" -n -U -s 160 -Z root -w - 2>/dev/null | python -m fqml.live.watch "$@"
