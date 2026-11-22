"""FireWatch — camera AI + Arduino sensors, fused into one 0..10 level, live dashboard.

    python app/server.py                       # webcam 0, Arduino auto-detected
    python app/server.py --sim                 # no Arduino: simulated sensors
    python app/server.py --source video.mp4    # test with a video file
    python app/server.py --model models/fire_smoke.pt
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

from detector import Vision
from link import ArduinoLink, SimLink

ROOT = Path(__file__).resolve().parent.parent
LEVEL_WARN, LEVEL_ALARM = 4, 7


def state_of(level: int) -> str:
    return "ALARM" if level >= LEVEL_ALARM else "WARNING" if level >= LEVEL_WARN else "CLEAR"


class Hub:
    def __init__(self, args):
        self.args = args
        model = args.model or next((str(p) for p in (ROOT / "models" / "fire_smoke.onnx",
                                                    ROOT / "models" / "fire_smoke.pt") if p.exists()), None)
        self.vision = Vision(model)
        self.link = SimLink(self.vision) if args.sim else ArduinoLink(args.port)
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.cam_ok = False
        self.fps = 0.0
        self.level = 0
        self.shown = 0
        self.vision_level = 0
        self.sensor_level = 0
        self.state = "CLEAR"
        self.history: deque = deque(maxlen=1200)  # 5 min at 4 Hz
        self.events: deque = deque(maxlen=80)
        self.event_id = 0
        self._link_was = None
        self._last_drop = 0.0
        self._last_snap = 0.0
        (ROOT / "logs").mkdir(exist_ok=True)
        (ROOT / "captures").mkdir(exist_ok=True)
        self.event("info", "start", model=self.vision.name)
        if self.vision.error:
            self.event("warn", "model_err", err=self.vision.error)

    # ---------------------------------------------------------------- events
    def event(self, kind: str, code: str, **args):
        """Language-neutral event: the UI turns `code` + `args` into AZ/EN text."""
        self.event_id += 1
        self.events.append({"id": self.event_id, "t": time.time(), "kind": kind, "code": code, "args": args})

    # ---------------------------------------------------------------- camera
    def camera_loop(self):
        src = self.args.source
        if src == "none":
            return
        src = int(src) if src.isdigit() else src
        is_file = isinstance(src, str)
        while True:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                self._set_cam(False)
                time.sleep(2)
                continue
            self._set_cam(True)
            delay = 1 / (cap.get(cv2.CAP_PROP_FPS) or 25) if is_file else 0
            misses = 0
            while True:
                t0 = time.time()
                ok, frame = cap.read()
                if not ok:
                    misses += 1
                    if is_file and misses < 3:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                misses = 0
                if frame.shape[1] > 960:
                    frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
                self.vision.process(frame)
                self.vision.draw(frame)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self.lock:
                    self.jpeg = buf.tobytes()
                    self.frame_id += 1
                dt = time.time() - t0
                if delay > dt:
                    time.sleep(delay - dt)
                self.fps = self.fps * 0.9 + (1 / max(1e-3, time.time() - t0)) * 0.1
            cap.release()
            self._set_cam(False)
            time.sleep(1)

    def _set_cam(self, ok: bool):
        if ok != self.cam_ok:
            self.cam_ok = ok
            self.event("info" if ok else "warn", "cam_on" if ok else "cam_off")

    # ---------------------------------------------------------------- fusion
    def fusion_loop(self):
        last_log = 0.0
        log_day, writer, fh = None, None, None
        while True:
            now = time.time()
            r = self.link.reading if self.link.online else None

            v = max(self.vision.fire, 0.85 * self.vision.smoke) if self.cam_ok else 0.0
            s = r.local / 10 if (r and not r.warm) else 0.0
            fused = 1 - (1 - v) * (1 - s)           # noisy-OR: agreement pushes the level up
            target = round(fused * 10)
            self.vision_level, self.sensor_level = round(v * 10), (r.local if r else 0)

            if target >= self.level:
                self.level = target
            elif now - self._last_drop >= 1.0:      # fall back at most 1 step per second
                self.level -= 1
                self._last_drop = now
            self.link.set_level(self.level)
            self.shown = max(self.level, r.local if r else 0)

            new_state = state_of(self.shown)
            if new_state != self.state:
                self.event(new_state.lower() if new_state != "CLEAR" else "ok", "state", state=new_state,
                           level=self.shown, vision=self.vision_level, sensor=self.sensor_level)
                if new_state == "ALARM" and now - self._last_snap > 20:
                    self._snapshot()
                self.state = new_state

            online = self.link.online
            if online != self._link_was:
                if self._link_was is not None or online:
                    port = "sim" if self.link.sim else self.link.port
                    self.event("info" if online else "warn", "link_on" if online else "link_off", port=port)
                self._link_was = online

            self.history.append(self._point(now, r))

            if now - last_log >= 1.0:
                last_log = now
                day = datetime.now().strftime("%Y-%m-%d")
                if day != log_day:
                    if fh:
                        fh.close()
                    path = ROOT / "logs" / f"{day}.csv"
                    new = not path.exists()
                    fh = open(path, "a", newline="")
                    writer = csv.writer(fh)
                    if new:
                        writer.writerow(["time", "level", "state", "vision_fire", "vision_smoke",
                                         "smoke_raw", "smoke_base", "temp_c", "flame_raw", "sensor_level"])
                    log_day = day
                writer.writerow([datetime.now().isoformat(timespec="seconds"), self.shown, self.state,
                                 f"{self.vision.fire:.3f}", f"{self.vision.smoke:.3f}",
                                 *((r.smoke_raw, r.smoke_base, r.temp, r.flame_raw, r.local) if r else ("",) * 5)])
                fh.flush()
            time.sleep(0.25)

    def _snapshot(self):
        with self.lock:
            jpg = self.jpeg
        if not jpg:
            return
        self._last_snap = time.time()
        name = datetime.now().strftime("alarm-%Y%m%d-%H%M%S.jpg")
        (ROOT / "captures" / name).write_bytes(jpg)
        self.event("info", "snapshot", file=f"captures/{name}")

    def _point(self, now, r):
        return {"t": round(now, 2), "level": self.shown,
                "fire": round(self.vision.fire, 3), "smoke": round(self.vision.smoke, 3),
                "temp": r.temp if r else None,
                "gas": r.smoke_delta if r else None,
                "flame": round(r.flame, 3) if r else None}

    def snapshot_state(self) -> dict:
        r = self.link.reading if self.link.online else None
        return {
            "t": time.time(),
            "level": self.shown,
            "state": self.state,
            "vision": {"fire": self.vision.fire, "smoke": self.vision.smoke, "level": self.vision_level,
                       "model": self.vision.name, "cam": self.cam_ok, "fps": round(self.fps, 1),
                       "boxes": len(self.vision.detections)},
            "sensors": None if not r else {
                "smoke_raw": r.smoke_raw, "smoke_base": r.smoke_base, "gas": r.smoke_delta,
                "temp": r.temp, "flame": r.flame, "local": r.local, "leds": r.shown,
                "muted": r.muted, "warm": r.warm},
            "link": {"online": self.link.online, "sim": self.link.sim, "port": self.link.port},
            "events": list(self.events)[-30:],
        }


# -------------------------------------------------------------------- web
def build_app(hub: Hub) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)
    static = Path(__file__).parent / "static"

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/video")
    async def video():
        async def gen():
            last = -1
            while True:
                with hub.lock:
                    fid, jpg = hub.frame_id, hub.jpeg
                if jpg is None or fid == last:
                    await asyncio.sleep(0.03)
                    continue
                last = fid
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/mute")
    def mute():
        hub.link.mute()
        hub.event("info", "mute")
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            await sock.send_json({"type": "init", "history": list(hub.history)})
            while True:
                msg = hub.snapshot_state()
                msg["type"] = "tick"
                msg["point"] = hub.history[-1] if hub.history else None
                await sock.send_json(msg)
                await asyncio.sleep(0.25)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


def main():
    ap = argparse.ArgumentParser(description="FireWatch")
    ap.add_argument("--source", default="0", help="camera index, video file, RTSP/HTTP url, or 'none'")
    ap.add_argument("--model", default=None, help="YOLO weights (.pt/.onnx) with fire/smoke classes")
    ap.add_argument("--port", default="auto", help="serial port, e.g. COM3 or /dev/ttyACM0")
    ap.add_argument("--sim", action="store_true", help="simulate the Arduino sensors")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--http", type=int, default=8000)
    args = ap.parse_args()

    hub = Hub(args)
    for fn in (hub.camera_loop, hub.fusion_loop, hub.link.run):
        threading.Thread(target=fn, daemon=True).start()
    print(f"FireWatch → http://localhost:{args.http}   vision: {hub.vision.name}")
    uvicorn.run(build_app(hub), host=args.host, port=args.http, log_level="warning")


if __name__ == "__main__":
    main()
