"""
Project 15 - RSNA Pneumonia Detection - Baseline model.

Architecture: RetinaNet (Lin 2017) with ResNet50 backbone and Feature Pyramid
Network (FPN), trained with focal loss for class imbalance. Implementation uses
the torchvision reference detector for transparency and auditability.

Evaluation metric for the baseline: mean Average Precision at IoU threshold 0.5
(mAP@0.5). The official RSNA challenge metric averages across IoU 0.40 to 0.75
in 0.05 steps; that is computed in the advanced script.

This file is runnable but NOT executed during implementation. v1.0 deliverable.

Run later (main session) with:

    cd /root/AI/project_root
    python src/model_baseline.py

Dependencies: torch, torchvision, pydicom, numpy, pandas, opencv-python, tqdm,
scikit-learn. Tested mental-model: torch 2.x, torchvision 0.18+.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.transforms import functional as F
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_IMG_DIR = DATA_DIR / "stage_2_train_images"
LABELS_CSV = DATA_DIR / "stage_2_train_labels.csv"
DELIVERABLES_DIR = ROOT / "deliverables"
DELIVERABLES_DIR.mkdir(exist_ok=True)

IMG_SIZE = 512        # downscale from 1024 for speed
BATCH_SIZE = 4
NUM_EPOCHS = 6
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 2       # background + opacity
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_dicom_image(path: Path, size: int = IMG_SIZE) -> np.ndarray:
    """Read a DICOM file, normalise to uint8, resize, apply CLAHE."""
    ds = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = np.stack([img, img, img], axis=-1)  # 3-channel for ImageNet backbone
    return img


def build_label_table(labels_csv: Path) -> pd.DataFrame:
    """One row per patient with grouped boxes."""
    df = pd.read_csv(labels_csv)
    grouped = df.groupby("patientId").agg(
        boxes=("Target", lambda _: None),
        target_max=("Target", "max"),
    ).reset_index()
    boxes_by_pid: Dict[str, List[List[float]]] = defaultdict(list)
    for _, r in df.iterrows():
        if r["Target"] == 1:
            boxes_by_pid[r["patientId"]].append(
                [r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]]
            )
    grouped["boxes"] = grouped["patientId"].map(lambda p: boxes_by_pid.get(p, []))
    return grouped


class RSNADataset(data_utils.Dataset):
    """Loads DICOM, scales boxes to IMG_SIZE, returns torchvision detection target."""

    def __init__(self, df: pd.DataFrame, img_dir: Path, size: int = IMG_SIZE):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.size = size
        self.scale = size / 1024.0  # original DICOM resolution

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        row = self.df.iloc[idx]
        img = load_dicom_image(self.img_dir / f"{row['patientId']}.dcm", self.size)
        img_t = F.to_tensor(img)

        boxes = [[b[0] * self.scale, b[1] * self.scale,
                  b[2] * self.scale, b[3] * self.scale]
                 for b in row["boxes"]]
        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.ones((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
        }
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def build_retinanet(num_classes: int = NUM_CLASSES) -> nn.Module:
    """RetinaNet ResNet50-FPN, ImageNet-pretrained backbone, focal-loss head."""
    model = retinanet_resnet50_fpn_v2(weights="DEFAULT", num_classes=91)
    # Replace classification head for opacity vs background.
    in_channels = model.head.classification_head.cls_logits.in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes
    cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes,
                           kernel_size=3, stride=1, padding=1)
    nn.init.normal_(cls_logits.weight, std=0.01)
    nn.init.constant_(cls_logits.bias, -4.6)  # focal-loss bias init
    model.head.classification_head.cls_logits = cls_logits
    return model


# ----------------------------------------------------------------------------
# Evaluation - mAP@0.5 (single threshold)
# ----------------------------------------------------------------------------

def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    xa = max(box1[0], box2[0]); ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2]); yb = min(box1[3], box2[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def average_precision_at_iou(all_preds, all_gts, iou_thresh: float = 0.5) -> float:
    """Pascal-VOC style AP averaged across images at one IoU threshold."""
    aps = []
    for preds, gts in zip(all_preds, all_gts):
        if len(gts) == 0 and len(preds) == 0:
            aps.append(1.0)
            continue
        if len(gts) == 0 and len(preds) > 0:
            aps.append(0.0)
            continue
        if len(preds) == 0:
            aps.append(0.0)
            continue
        order = np.argsort(-preds[:, 4])
        preds = preds[order]
        matched = np.zeros(len(gts), dtype=bool)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        for i, p in enumerate(preds):
            best, best_iou = -1, 0.0
            for j, g in enumerate(gts):
                if matched[j]:
                    continue
                v = iou(p[:4], g)
                if v > best_iou:
                    best, best_iou = j, v
            if best_iou >= iou_thresh and best >= 0:
                tp[i] = 1.0; matched[best] = True
            else:
                fp[i] = 1.0
        if tp.sum() == 0:
            aps.append(0.0); continue
        cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
        recall = cum_tp / max(len(gts), 1)
        precision = cum_tp / (cum_tp + cum_fp)
        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recall >= t
            ap += (precision[mask].max() if mask.any() else 0.0) / 11
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


# ----------------------------------------------------------------------------
# Train and evaluate
# ----------------------------------------------------------------------------

def run() -> Dict[str, float]:
    torch.manual_seed(SEED); np.random.seed(SEED)

    df = build_label_table(LABELS_CSV)
    train_df, val_df = train_test_split(
        df, test_size=0.15, stratify=df["target_max"], random_state=SEED
    )

    train_ds = RSNADataset(train_df, TRAIN_IMG_DIR)
    val_ds = RSNADataset(val_df, TRAIN_IMG_DIR)

    train_loader = data_utils.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=2,
    )
    val_loader = data_utils.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    model = build_retinanet().to(DEVICE)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running = 0.0
        for imgs, targets in tqdm(train_loader, desc=f"epoch {epoch + 1}/{NUM_EPOCHS}"):
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            running += float(loss)
        scheduler.step()
        print(f"epoch {epoch + 1} mean loss: {running / max(len(train_loader), 1):.4f}")

    # Evaluation
    model.eval()
    preds_all, gts_all = [], []
    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc="eval"):
            imgs = [img.to(DEVICE) for img in imgs]
            outputs = model(imgs)
            for out, tgt in zip(outputs, targets):
                p = torch.cat([out["boxes"], out["scores"].unsqueeze(1)], dim=1).cpu().numpy()
                preds_all.append(p)
                gts_all.append(tgt["boxes"].cpu().numpy())

    map50 = average_precision_at_iou(preds_all, gts_all, iou_thresh=0.5)

    metrics = {
        "model": "RetinaNet ResNet50-FPN",
        "loss": "focal",
        "img_size": IMG_SIZE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "mAP@0.5": round(map50, 4),
    }

    torch.save(model.state_dict(), DELIVERABLES_DIR / "retinanet_baseline.pt")
    with open(DELIVERABLES_DIR / "metrics_baseline.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run()
