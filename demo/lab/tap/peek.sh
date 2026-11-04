#!/bin/sh
# Print the first packets seen on the lab network, as text (lab only).
set -e
IFACE=$(python demo/lab/tap/lab_iface.py)
echo "capturing on $IFACE" >&2
exec tcpdump -i "$IFACE" -n -c "${1:-15}"
