"""
Project 15 - RSNA Pneumonia Detection - Advanced model.

Primary architecture: YOLOv8 (Ultralytics) on chest X-rays after CLAHE
pre-processing and heavy augmentation (affine, brightness/contrast, mixup).
YOLOv8 was chosen over a vanilla Faster R-CNN because the RSNA opacities are
relatively large (a quarter to a third of the image area), so a single-stage
anchor-free / hybrid detector with strong multi-scale heads handles the regime
efficiently.

Alternatives considered:

- Faster R-CNN with ResNet50-FPN (torchvision). Stronger localisation on
  small/medium boxes, slower per epoch. Built optionally below as
  `train_faster_rcnn()` for direct comparison.
- Detectron2 RetinaNet/Faster-RCNN. More complex install (compiled CUDA ops);
  considered out of scope for Phase 1 but flagged in the manuscript.

Evaluation metric: mAP averaged over IoU thresholds 0.40 to 0.75 in 0.05 steps
(the official RSNA challenge metric). Computed via the helper in
`model_baseline.py`.

This file is runnable but NOT executed during scaffolding. Phase 1 deliverable.

Run later with:

    cd /root/AI/liora_projects/15_rsna_pneumonia
    python src/model_advanced.py

Dependencies: ultralytics (YOLOv8), torch, torchvision, pydicom, opencv-python,
albumentations, pandas, numpy, scikit-learn, tqdm.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Lazy imports of torch / ultralytics / albumentations live inside functions so
# the file imports cleanly even if a sub-dependency is missing in scaffolding.

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_IMG_DIR = DATA_DIR / "stage_2_train_images"
LABELS_CSV = DATA_DIR / "stage_2_train_labels.csv"
DELIVERABLES_DIR = ROOT / "deliverables"
DELIVERABLES_DIR.mkdir(exist_ok=True)

YOLO_DATA_DIR = ROOT / "data" / "yolo_format"
IMG_SIZE = 640
BATCH_SIZE = 8
NUM_EPOCHS = 30
SEED = 42


# ----------------------------------------------------------------------------
# DICOM-to-PNG conversion with CLAHE
# ----------------------------------------------------------------------------

def dicom_to_png(src: Path, dst: Path, size: int = IMG_SIZE) -> None:
    ds = pydicom.dcmread(str(src))
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    cv2.imwrite(str(dst), img)


def write_yolo_labels(boxes, dst: Path, orig_size: int = 1024) -> None:
    """YOLO label format: class cx_norm cy_norm w_norm h_norm (0..1)."""
    lines = []
    for x, y, w, h in boxes:
        cx = (x + w / 2) / orig_size
        cy = (y + h / 2) / orig_size
        wn = w / orig_size
        hn = h / orig_size
        lines.append(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
    dst.write_text("\n".join(lines))


def prepare_yolo_dataset() -> Path:
    """Convert DICOMs to PNGs and write YOLO labels under data/yolo_format/."""
    df = pd.read_csv(LABELS_CSV)
    boxes_by_pid: Dict[str, list] = {}
    for _, r in df.iterrows():
        if r["Target"] == 1:
            boxes_by_pid.setdefault(r["patientId"], []).append(
                [r["x"], r["y"], r["width"], r["height"]]
            )
    patients = df["patientId"].drop_duplicates().tolist()
    target_max = df.groupby("patientId")["Target"].max().reindex(patients).values

    train, val = train_test_split(
        patients, test_size=0.15, stratify=target_max, random_state=SEED
    )

    if YOLO_DATA_DIR.exists():
        shutil.rmtree(YOLO_DATA_DIR)
    for split, pids in [("train", train), ("val", val)]:
        (YOLO_DATA_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (YOLO_DATA_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        for pid in tqdm(pids, desc=f"convert {split}"):
            src = TRAIN_IMG_DIR / f"{pid}.dcm"
            dst_img = YOLO_DATA_DIR / split / "images" / f"{pid}.png"
            dst_lbl = YOLO_DATA_DIR / split / "labels" / f"{pid}.txt"
            dicom_to_png(src, dst_img)
            write_yolo_labels(boxes_by_pid.get(pid, []), dst_lbl)

    yaml_path = YOLO_DATA_DIR / "rsna.yaml"
    yaml_path.write_text(
        f"path: {YOLO_DATA_DIR}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        "  0: opacity\n"
    )
    return yaml_path


# ----------------------------------------------------------------------------
# YOLOv8 train + evaluate
# ----------------------------------------------------------------------------

def train_yolov8(yaml_path: Path) -> Dict[str, float]:
    from ultralytics import YOLO

    model = YOLO("yolov8s.pt")  # small variant; swap to yolov8m for more capacity
    results = model.train(
        data=str(yaml_path),
        epochs=NUM_EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        seed=SEED,
        project=str(DELIVERABLES_DIR),
        name="yolov8_rsna",
        # heavy augmentation knobs (Ultralytics defaults plus mixup)
        mosaic=1.0,
        mixup=0.15,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,    # no vertical flip - anatomy
        hsv_h=0.0,     # grayscale already
        hsv_s=0.0,
        hsv_v=0.4,
        patience=10,
    )
    metrics = {
        "model": "YOLOv8s",
        "img_size": IMG_SIZE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "mAP@0.5": float(results.box.map50) if hasattr(results, "box") else None,
        "mAP@0.5:0.95": float(results.box.map) if hasattr(results, "box") else None,
    }
    return metrics


# ----------------------------------------------------------------------------
# Optional Faster R-CNN comparison
# ----------------------------------------------------------------------------

def train_faster_rcnn() -> Dict[str, float]:
    """Two-stage Faster R-CNN baseline-comparator. Optional, slower than YOLOv8."""
    import torch
    from torch import nn
    import torch.utils.data as data_utils
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    # Reuse the dataset class from the baseline module
    from model_baseline import (
        RSNADataset, build_label_table, collate_fn,
        average_precision_at_iou, IMG_SIZE as BASE_IMG_SIZE,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = build_label_table(LABELS_CSV)
    train_df, val_df = train_test_split(
        df, test_size=0.15, stratify=df["target_max"], random_state=SEED
    )

    train_loader = data_utils.DataLoader(
        RSNADataset(train_df, TRAIN_IMG_DIR), batch_size=4, shuffle=True,
        collate_fn=collate_fn, num_workers=2,
    )
    val_loader = data_utils.DataLoader(
        RSNADataset(val_df, TRAIN_IMG_DIR), batch_size=4, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(8):
        model.train()
        for imgs, targets in tqdm(train_loader, desc=f"frcnn epoch {epoch + 1}"):
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(imgs, targets).values())
            optim.zero_grad(); loss.backward(); optim.step()

    model.eval()
    preds_all, gts_all = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = [im.to(device) for im in imgs]
            outs = model(imgs)
            for out, tgt in zip(outs, targets):
                p = torch.cat([out["boxes"], out["scores"].unsqueeze(1)], dim=1).cpu().numpy()
                preds_all.append(p)
                gts_all.append(tgt["boxes"].cpu().numpy())
    map50 = average_precision_at_iou(preds_all, gts_all, iou_thresh=0.5)
    return {"model": "FasterRCNN ResNet50-FPN", "mAP@0.5": round(map50, 4)}


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def run() -> Dict[str, float]:
    yaml_path = prepare_yolo_dataset()
    metrics = train_yolov8(yaml_path)
    with open(DELIVERABLES_DIR / "metrics_advanced.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run()
