"""Print the name of the lab bridge interface (the one holding 10.10.0.1). Lab only.
Normally it is br-fqml (set in compose.yml); this finds it even if Docker named it br-<id>."""
import fcntl
import os
import socket
import struct
import sys

GATEWAY = sys.argv[1] if len(sys.argv) > 1 else "10.10.0.1"
SIOCGIFADDR = 0x8915
if len(sys.argv) == 1 and os.path.exists("/sys/class/net/br-fqml"):
    print("br-fqml")
    sys.exit(0)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for name in sorted(os.listdir("/sys/class/net")):
    try:
        raw = fcntl.ioctl(s.fileno(), SIOCGIFADDR, struct.pack("256s", name[:15].encode()))
    except OSError:
        continue
    if socket.inet_ntoa(raw[20:24]) == GATEWAY:
        print(name)
        sys.exit(0)
sys.exit(f"no interface holds {GATEWAY}: is the lab running? (docker compose -f demo/lab/compose.yml up -d)")
