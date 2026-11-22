"""Build the FireWatch fire/smoke model.

    pip install ultralytics
    python train.py                    # full pipeline (CPU works, GPU is much faster)
    python train.py --epochs 30 --device 0

Pipeline
  1. base model   : YOLOv8n already trained on D-Fire (21k fire/smoke images)
                    https://huggingface.co/rabahdev/fire-smoke-yolov8n
  2. indoor data  : ISFire-4 (7,370 indoor images), 4 classes merged to 2 -> 0 smoke, 1 fire
                    https://huggingface.co/datasets/shahriar-5/ISFire-4
  3. fine-tune the base model on the indoor data
  4. evaluate base vs fine-tuned on the indoor test split, keep the better one
  5. export -> models/fire_smoke.onnx  (the app runs it with OpenCV only, no torch needed)
"""
import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
BASE_URL = "https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt"
DATA_URL = "https://huggingface.co/datasets/shahriar-5/ISFire-4/resolve/main/Dataset.zip"
# ISFire-4: 0 black smoke, 1 blue fire, 2 white-gray smoke, 3 yellow-orange fire
CLASS_MAP = {0: 0, 1: 1, 2: 0, 3: 1}


def download(url: str, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)


def safe_import(src: Path, dst: Path):
    """Load third-party weights WITHOUT running pickle code (weights_only + audited allowlist),
    copy the tensors into a freshly built YOLOv8n and save our own clean checkpoint."""
    if dst.exists():
        return
    import collections
    import torch
    import torch.nn as nn
    from torch.serialization import add_safe_globals
    from ultralytics.nn.modules.block import C2f, DFL, SPPF, Bottleneck
    from ultralytics.nn.modules.conv import Concat, Conv
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.tasks import DetectionModel

    add_safe_globals([DetectionModel, Bottleneck, C2f, DFL, SPPF, Concat, Conv, Detect, nn.SiLU, nn.BatchNorm2d,
                      nn.ModuleList, nn.Sequential, nn.Conv2d, nn.Identity, nn.MaxPool2d, nn.Upsample,
                      set, collections.OrderedDict])
    ck = torch.load(src, map_location="cpu", weights_only=True)
    old = (ck.get("ema") or ck["model"]).float()
    names = dict(old.names)
    model = DetectionModel("yolov8n.yaml", nc=len(names), verbose=False)
    model.load_state_dict(old.state_dict(), strict=True)
    model.names = names
    model.args = {"imgsz": 640, "task": "detect"}
    torch.save({"model": model.half(), "train_args": {"imgsz": 640, "task": "detect", "model": "yolov8n.yaml"},
                "date": None, "version": "clean"}, dst)


def convert_label(src: Path) -> list[str]:
    out = []
    for line in src.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = CLASS_MAP.get(int(float(p[0])))
        if cls is None:
            continue
        v = [float(x) for x in p[1:]]
        if len(v) > 4:  # polygon -> bounding box
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            v = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        out.append(f"{cls} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {v[3]:.6f}")
    return out


def prepare_dataset() -> Path:
    yaml = DATA / "indoor2.yaml"
    if yaml.exists():
        return yaml
    zpath = DATA / "isfire4.zip"
    download(DATA_URL, zpath)
    raw = DATA / "raw"
    if not raw.exists():
        zipfile.ZipFile(zpath).extractall(raw)
    src_root = next(raw.glob("*/data.yaml")).parent
    dst_root = DATA / "indoor2"
    counts = {"smoke": 0, "fire": 0, "images": 0}
    for split in ("train", "valid", "test"):
        (dst_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_root / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in (src_root / split / "images").iterdir():
            lbl = src_root / split / "labels" / (img.stem + ".txt")
            lines = convert_label(lbl) if lbl.exists() else []
            shutil.copy(img, dst_root / split / "images" / img.name)
            (dst_root / split / "labels" / (img.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            counts["images"] += 1
            for ln in lines:
                counts["fire" if ln[0] == "1" else "smoke"] += 1
    yaml.write_text(f"path: {dst_root}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
                    "names:\n  0: smoke\n  1: fire\n")
    print("dataset ready:", counts)
    return yaml


def evaluate(model_path: Path, yaml: Path, imgsz: int, device: str) -> dict:
    from ultralytics import YOLO
    m = YOLO(str(model_path))
    r = m.val(data=str(yaml), split="test", imgsz=imgsz, batch=16, device=device,
              plots=False, verbose=False, project=str(ROOT / "runs"), name="eval", exist_ok=True)
    per = {m.names[int(c)]: round(float(r.box.maps[i]), 3) for i, c in enumerate(r.box.ap_class_index)}
    return {"mAP50": round(float(r.box.map50), 3), "mAP50-95": round(float(r.box.map), 3),
            "precision": round(float(r.box.mp), 3), "recall": round(float(r.box.mr), 3), "per_class_mAP50-95": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="cpu, or 0 for the first GPU")
    ap.add_argument("--fraction", type=float, default=1.0, help="use part of the train set (0.05 = quick test)")
    ap.add_argument("--eval-only", action="store_true", help="only score the base model")
    args = ap.parse_args()

    from ultralytics import YOLO

    MODELS.mkdir(exist_ok=True)
    downloaded = MODELS / "dfire_yolov8n.pt"
    download(BASE_URL, downloaded)
    base = MODELS / "base_clean.pt"
    safe_import(downloaded, base)
    yaml = prepare_dataset()

    report = {"base (D-Fire)": evaluate(base, yaml, args.imgsz, args.device)}
    print("base:", report["base (D-Fire)"])
    if args.eval_only:
        return

    model = YOLO(str(base))
    res = model.train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                      device=args.device, fraction=args.fraction, lr0=0.002, warmup_epochs=1, close_mosaic=2, patience=0,
                      workers=4, project=str(ROOT / "runs"), name="indoor", exist_ok=True, plots=False)
    tuned = Path(res.save_dir) / "weights" / "best.pt"
    report["fine-tuned (D-Fire + indoor)"] = evaluate(tuned, yaml, args.imgsz, args.device)
    print("fine-tuned:", report["fine-tuned (D-Fire + indoor)"])

    winner_name, winner = max(((k, v) for k, v in report.items()), key=lambda kv: kv[1]["mAP50"])
    src = base if winner_name.startswith("base") else tuned
    shutil.copy(src, MODELS / "fire_smoke.pt")
    onnx = YOLO(str(src)).export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False)
    shutil.move(onnx, MODELS / "fire_smoke.onnx")
    (MODELS / "fire_smoke.json").write_text(json.dumps({"names": ["smoke", "fire"], "imgsz": args.imgsz}))
    report["selected"] = winner_name
    (MODELS / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("-> models/fire_smoke.onnx")


if __name__ == "__main__":
    main()
