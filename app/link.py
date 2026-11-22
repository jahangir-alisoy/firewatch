"""Serial link to the Arduino node (plus a simulator for running without hardware)."""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass

import serial
import serial.tools.list_ports

ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4}  # Arduino, CH340, FTDI, CP210x


@dataclass
class Reading:
    smoke_raw: int
    smoke_base: int
    temp: float
    flame_raw: int
    local: int
    shown: int
    muted: bool
    warm: bool
    t: float

    @property
    def smoke_delta(self) -> int:
        return max(0, self.smoke_raw - self.smoke_base)

    @property
    def flame(self) -> float:
        """IR intensity 0..1 (module output drops when it sees a flame)."""
        return max(0.0, min(1.0, (1023 - self.flame_raw) / 1023))


def parse(line: str) -> Reading | None:
    p = line.strip().split(",")
    if len(p) != 9 or p[0] != "D":
        return None
    try:
        v = [int(x) for x in p[1:]]
    except ValueError:
        return None
    return Reading(v[0], v[1], v[2] / 10, v[3], v[4], v[5], bool(v[6]), bool(v[7]), time.time())


class ArduinoLink:
    sim = False

    def __init__(self, port: str = "auto", baud: int = 115200):
        self.port_arg = port
        self.baud = baud
        self.port: str | None = None
        self.reading: Reading | None = None
        self.level = 0
        self._mute = False
        self._stop = threading.Event()

    @property
    def online(self) -> bool:
        return self.reading is not None and time.time() - self.reading.t < 2.0

    def set_level(self, level: int):
        self.level = int(level)

    def mute(self):
        self._mute = True

    def find_port(self) -> str | None:
        if self.port_arg != "auto":
            return self.port_arg
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if p.vid in ARDUINO_VIDS:
                return p.device
        for p in ports:
            if "ACM" in p.device or "USB" in p.device:
                return p.device
        return None

    def run(self):
        while not self._stop.is_set():
            dev = self.find_port()
            if not dev:
                time.sleep(2)
                continue
            try:
                with serial.Serial(dev, self.baud, timeout=0.1) as ser:
                    self.port = dev
                    last_tx = 0.0
                    buf = b""
                    while not self._stop.is_set():
                        buf += ser.read(256)
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            r = parse(line.decode(errors="ignore"))
                            if r:
                                self.reading = r
                        now = time.time()
                        if now - last_tx >= 0.25:
                            last_tx = now
                            ser.write(f"L{self.level}\n".encode())
                        if self._mute:
                            self._mute = False
                            ser.write(b"M\n")
            except (serial.SerialException, OSError):
                pass
            self.port = None
            time.sleep(2)

    def stop(self):
        self._stop.set()


def local_level(smoke_delta: float, temp: float, flame_raw: int) -> int:
    """Same rule as computeLocal() in firewatch.ino (without rate-of-rise)."""
    def scale(v, lo, hi):
        if v < lo:
            return 0
        if v >= hi:
            return 10
        return 1 + int((v - lo) * 9 / (hi - lo))

    s, t, f = scale(smoke_delta, 60, 400), scale(temp, 45, 70), scale(1023 - flame_raw, 250, 850)
    lvl = max(s, t, f)
    if sum(x >= 3 for x in (s, t, f)) >= 2:
        lvl = min(10, lvl + 2)
    return lvl


class SimLink(ArduinoLink):
    """Fake Arduino for testing the UI. Sensors slowly react to what the camera sees."""

    sim = True

    def __init__(self, vision):
        super().__init__()
        self.vision = vision
        self.port = "simulated"

    def run(self):
        base, temp, smoke, flame, mute_until, start = 180.0, 23.0, 180.0, 1010.0, 0.0, time.time()
        while not self._stop.is_set():
            fire = self.vision.fire
            vsmoke = self.vision.smoke
            temp += ((23 + 45 * fire) - temp) * 0.015 + random.uniform(-0.05, 0.05)
            smoke += ((base + 380 * max(fire * 0.8, vsmoke)) - smoke) * 0.04 + random.uniform(-3, 3)
            flame += ((1010 - 800 * fire) - flame) * 0.2 + random.uniform(-4, 4)
            if self._mute:
                self._mute = False
                mute_until = time.time() + 60
            warm = time.time() - start < 3
            local = 0 if warm else local_level(smoke - base, temp, int(flame))
            self.reading = Reading(int(smoke), int(base), round(temp, 1), int(flame), local,
                                   max(local, self.level), time.time() < mute_until, warm, time.time())
            time.sleep(0.25)
