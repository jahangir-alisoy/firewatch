"""Fire / smoke detection on camera frames.

Backends, same output:
  OnnxDetector       — trained YOLOv8 model exported to ONNX (models/fire_smoke.onnx), runs on OpenCV only.
  YoloDetector       — same model as ultralytics .pt (needs `pip install ultralytics`).
  HeuristicDetector  — fallback when there is no model: flame colour + flicker, grey moving regions.

Vision(...) wraps either one and smooths the scores over time.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import json

import cv2
import numpy as np


@dataclass
class Detection:
    label: str            # "fire" | "smoke"
    conf: float           # 0..1
    box: tuple[int, int, int, int]


def _label(name: str) -> str | None:
    n = name.lower()
    return "fire" if ("fire" in n or "flame" in n) else "smoke" if "smoke" in n else None


class OnnxDetector:
    """YOLOv8 ONNX inference with cv2.dnn: letterbox -> forward -> decode -> NMS."""

    def __init__(self, path: str, conf: float = 0.30, iou: float = 0.45):
        meta = Path(path).with_suffix(".json")
        info = json.loads(meta.read_text()) if meta.exists() else {}
        self.names = info.get("names", ["smoke", "fire"])
        self.size = int(info.get("imgsz", 640))
        self.net = cv2.dnn.readNetFromONNX(path)
        self.conf, self.iou = conf, iou
        self.name = f"YOLOv8 · {Path(path).name}"

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        S = self.size
        r = min(S / h, S / w)
        nw, nh = round(w * r), round(h * r)
        dx, dy = (S - nw) // 2, (S - nh) // 2
        canvas = np.full((S, S, 3), 114, np.uint8)
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.net.setInput(cv2.dnn.blobFromImage(canvas, 1 / 255.0, (S, S), swapRB=True))
        out = self.net.forward()[0].T                      # (anchors, 4 + classes)
        scores = out[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        if not keep.any():
            return []
        out, cls, conf = out[keep], cls[keep], conf[keep]
        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        x1 = (cx - bw / 2 - dx) / r
        y1 = (cy - bh / 2 - dy) / r
        boxes = np.stack([x1, y1, bw / r, bh / r], 1)
        idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), conf.tolist(), cls.tolist(), self.conf, self.iou)
        dets = []
        for i in np.array(idx).reshape(-1):
            label = _label(self.names[int(cls[i])])
            if label:
                x, y, ww, hh = boxes[i]
                dets.append(Detection(label, float(conf[i]),
                                      (max(0, int(x)), max(0, int(y)), min(w, int(x + ww)), min(h, int(y + hh)))))
        return dets


class YoloDetector:
    def __init__(self, path: str, conf: float = 0.30, imgsz: int = 640):
        from ultralytics import YOLO  # optional dependency

        self.model = YOLO(path, task="detect")
        self.conf = conf
        self.imgsz = imgsz
        self.name = f"YOLO · {Path(path).name}"

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        names = res.names
        out = []
        for b in res.boxes:
            label = _label(str(names[int(b.cls)]))
            if label:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                out.append(Detection(label, float(b.conf), (x1, y1, x2, y2)))
        return out


class HeuristicDetector:
    """Classical CV fallback. Static orange objects score low because they don't flicker."""

    name = "OpenCV evristik (modelsiz)"
    W = 320

    def __init__(self):
        self.fire_hist: deque[np.ndarray] = deque(maxlen=8)
        self.bg: np.ndarray | None = None
        self.smoke_persist: np.ndarray | None = None
        self.k3 = np.ones((3, 3), np.uint8)
        self.k7 = np.ones((7, 7), np.uint8)

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        s = self.W / w
        small = cv2.GaussianBlur(cv2.resize(frame, (self.W, int(h * s))), (5, 5), 0)
        area_total = small.shape[0] * small.shape[1]

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        b, g, r = (c.astype(np.int16) for c in cv2.split(small))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # motion against a slow background model
        if self.bg is None or self.bg.shape != gray.shape:
            self.bg = gray.astype(np.float32)
            self.smoke_persist = np.zeros_like(self.bg)
            self.fire_hist.clear()
        motion = cv2.absdiff(gray, cv2.convertScaleAbs(self.bg)) > 10
        cv2.accumulateWeighted(gray, self.bg, 0.01)

        # ---- fire: red/orange/yellow, bright, saturated, R > G > B
        fire = (H <= 35) & (S >= 80) & (V >= 170) & (r >= 170) & (r > g) & (g > b)
        fire = cv2.morphologyEx(fire.astype(np.uint8) * 255, cv2.MORPH_OPEN, self.k3)
        fire = cv2.dilate(fire, self.k3)
        self.fire_hist.append(fire > 0)

        if len(self.fire_hist) >= 4:
            on = np.mean(np.stack(self.fire_hist), axis=0)
            flicker = (on > 0.12) & (on < 0.88)   # pixels that keep toggling
        else:
            flicker = np.zeros_like(fire, dtype=bool)

        dets: list[Detection] = []
        for c in cv2.findContours(fire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            a = cv2.contourArea(c) / area_total
            if a < 0.002:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            roi_fire = fire[y:y + bh, x:x + bw] > 0
            roi_flick = flicker[y:y + bh, x:x + bw]
            flick_ratio = roi_flick.sum() / max(1, roi_fire.sum())
            area_f = min(1.0, a / 0.02) ** 0.5
            flick_f = min(1.0, flick_ratio / 0.30)
            conf = area_f * (0.08 + 0.92 * flick_f)   # no flicker -> almost ignored
            dets.append(Detection("fire", float(conf), self._up(x, y, bw, bh, s)))

        # ---- smoke: low saturation, mid brightness, moving, soft (few edges), persistent
        greyish = (S < 45) & (V > 90) & (V < 235)
        cand = (greyish & motion).astype(np.float32)
        self.smoke_persist = self.smoke_persist * 0.92 + cand * 0.08
        sm = (self.smoke_persist > 0.35).astype(np.uint8) * 255
        sm = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, self.k7)
        edges = cv2.Canny(gray, 60, 140) > 0
        for c in cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            a = cv2.contourArea(c) / area_total
            if a < 0.01:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            edge_density = edges[y:y + bh, x:x + bw].mean()
            softness = max(0.0, 1.0 - edge_density / 0.12)
            conf = min(1.0, a / 0.08) ** 0.5 * softness * 0.8
            if conf > 0.15:
                dets.append(Detection("smoke", float(conf), self._up(x, y, bw, bh, s)))
        return dets

    @staticmethod
    def _up(x, y, w, h, s):
        return int(x / s), int(y / s), int((x + w) / s), int((y + h) / s)


class Vision:
    """Runs a detector and keeps smoothed fire/smoke scores (fast attack, slow release)."""

    COLORS = {"fire": (59, 59, 208), "smoke": (183, 194, 195)}  # BGR

    def __init__(self, model_path: str | None = None):
        self.detector = None
        self.error = ""
        if model_path and Path(model_path).exists():
            try:
                cls = OnnxDetector if model_path.endswith(".onnx") else YoloDetector
                self.detector = cls(model_path)
            except Exception as e:  # missing dependency or bad weights -> fall back
                self.error = f"Model yüklənmədi: {e}"
        if self.detector is None:
            self.detector = HeuristicDetector()
        self.fire = 0.0
        self.smoke = 0.0
        self.detections: list[Detection] = []

    @property
    def name(self) -> str:
        return self.detector.name

    def process(self, frame: np.ndarray) -> list[Detection]:
        dets = self.detector(frame)
        f = max((d.conf for d in dets if d.label == "fire"), default=0.0)
        s = max((d.conf for d in dets if d.label == "smoke"), default=0.0)
        self.fire = self._ema(self.fire, f)
        self.smoke = self._ema(self.smoke, s)
        self.detections = dets
        return dets

    @staticmethod
    def _ema(old: float, new: float) -> float:
        a = 0.45 if new > old else 0.08
        return old + (new - old) * a

    def draw(self, frame: np.ndarray) -> np.ndarray:
        for d in self.detections:
            if d.conf < 0.3:
                continue
            x1, y1, x2, y2 = d.box
            col = self.COLORS[d.label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
            txt = f"{d.label} {d.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 6, y1)
            cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty), col, -1)
            cv2.putText(frame, txt, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (20, 20, 20) if d.label == "smoke" else (255, 255, 255), 1, cv2.LINE_AA)
        return frame
