"""HMI / operator workstation (lab only): the source of normal industrial traffic.

Every cycle it behaves like a real control-room screen:
    * reads the PLC tank registers over Modbus every second (function 3)
    * fetches a camera snapshot over HTTP every few seconds
    * stays subscribed to the sensor's MQTT topic
    * occasionally writes a new setpoint to the PLC (a real operator adjusting the process),
      roughly every 2 minutes -- so "a write" is normal, but a BURST of writes from another host is not.

Prints the tank level each cycle, so `docker logs lab-hmi` looks like an operator console.
"""
from __future__ import annotations

import os
import random
import subprocess
import threading
import time
import urllib.request

from modbus import Client

PLC = os.environ.get("PLC", "10.10.0.20")
CAMERA = os.environ.get("CAMERA", "10.10.0.11")
BROKER = os.environ.get("BROKER", "10.10.0.40")
TOPIC = os.environ.get("TOPIC", "plant/tank/temperature")
SNAP_EVERY = float(os.environ.get("SNAP_EVERY", "3"))
SETPOINT_EVERY = float(os.environ.get("SETPOINT_EVERY", "120"))


def subscribe_mqtt():
    """Stay subscribed so the broker keeps sending the sensor's readings (steady background traffic)."""
    while True:
        try:
            subprocess.run(["mosquitto_sub", "-h", BROKER, "-t", TOPIC, "-i", "hmi-console"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            return
        time.sleep(2)


def snapshots():
    while True:
        try:
            urllib.request.urlopen(f"http://{CAMERA}/snapshot.jpg", timeout=3).read()
        except OSError:
            pass
        time.sleep(SNAP_EVERY)


def main():
    rng = random.Random(30)
    threading.Thread(target=subscribe_mqtt, daemon=True).start()
    threading.Thread(target=snapshots, daemon=True).start()
    plc = Client(PLC)
    print(f"HMI running: reading PLC {PLC} every 1 s, camera {CAMERA} every {SNAP_EVERY:g} s", flush=True)
    last_setpoint = time.time()
    while True:
        try:
            level, setpoint, pump, mode, alarm = plc.read_holding(0, 5)
            line = (f"{time.strftime('%H:%M:%S')}  level {level / 10:5.1f} %  setpoint {setpoint / 10:5.1f} %  "
                    f"pump {'ON ' if pump else 'OFF'}  {['ok', 'HIGH', 'OVERFLOW'][min(alarm, 2)]}")
            if time.time() - last_setpoint > SETPOINT_EVERY:      # operator nudges the setpoint
                new = max(350, min(650, setpoint + rng.choice([-30, -20, 20, 30])))
                plc.write_register(1, new)
                last_setpoint = time.time()
                line += f"  -> operator set setpoint {new / 10:.1f} %"
            print(line, flush=True)
        except OSError:
            print(f"{time.strftime('%H:%M:%S')}  PLC unreachable, retrying", flush=True)
            plc.close()
        time.sleep(1)


if __name__ == "__main__":
    main()
