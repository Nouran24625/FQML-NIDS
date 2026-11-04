"""MQTT temperature sensor (lab only): publishes one reading per second to the broker.

Uses one persistent connection via `mosquitto_pub -l`, which sends one message per line read from
stdin, the way a real device keeps a session open. Reconnects if the broker restarts.
"""
from __future__ import annotations

import math
import os
import random
import subprocess
import time

BROKER = os.environ.get("BROKER", "10.10.0.40")
TOPIC = os.environ.get("TOPIC", "plant/tank/temperature")
PERIOD = float(os.environ.get("PERIOD", "1.0"))


def reading(t: float, rng: random.Random) -> float:
    return 24.0 + 1.5 * math.sin(t / 120) + rng.gauss(0, 0.15)


def main():
    rng = random.Random(12)
    print(f"sensor running: publishing {TOPIC} to {BROKER} every {PERIOD:g} s", flush=True)
    while True:
        try:
            proc = subprocess.Popen(
                ["mosquitto_pub", "-h", BROKER, "-t", TOPIC, "-i", "temp-sensor-01", "-l"],
                stdin=subprocess.PIPE, text=True)
        except FileNotFoundError:
            print("mosquitto_pub not installed", flush=True)
            return
        try:
            while proc.poll() is None:
                line = f"{reading(time.time(), rng):.2f}\n"
                proc.stdin.write(line)
                proc.stdin.flush()
                time.sleep(PERIOD)
        except (BrokenPipeError, OSError):
            pass
        finally:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(2)   # broker gone: wait and reconnect


if __name__ == "__main__":
    main()
