# Project 15 - RSNA Pneumonia Detection

**Track:** Computer Vision / Medical Imaging
**Difficulty:** 8/10
**Status:** Phase 1 scaffold

## Goal

Build a CV object-detection pipeline that localises pneumonia opacities on frontal-chest X-rays from the RSNA Pneumonia Detection Challenge (Kaggle 2018). The model should both detect and localise (bounding box) regions of opacity consistent with pneumonia, supporting clinician workflows in which radiologists verify and adjudicate findings.

## Twin context

This project is the medical-imaging twin of project #10 Severstal Steel (industrial defect detection). Both are object-detection tasks under heavy class imbalance with imperfect ground truth. The contrast (industrial vs medical, photometric vs radiographic, surface defect vs anatomical pathology) is a deliberate teaching pair across the cohort. It also sits alongside project #8 (COVID-19 chest X-rays, classification) on the medical-imaging axis - same modality, different task formulation, shared dataset-bias literature.

## Dataset

- Source: Kaggle - `rsna-pneumonia-detection-challenge` (Radiological Society of North America, 2018, in partnership with Kaggle and the NIH).
- Size: ~3.7 GB (training + test sets, DICOM format).
- Records: ~26,684 patient studies in `stage_2_train_images/`, ~3,000 in `stage_2_test_images/`. One image per patient.
- Labels: `stage_2_train_labels.csv` with `patientId`, `x`, `y`, `width`, `height`, `Target` (0 = no opacity, 1 = opacity present, with bounding box). Multiple rows per patient possible (multi-box).
- Class metadata: `stage_2_detailed_class_info.csv` with three classes: `Normal`, `No Lung Opacity / Not Normal`, `Lung Opacity`. Only `Lung Opacity` rows have non-null bounding boxes.
- Image format: DICOM, single-channel, typically 1024 x 1024 pixels.

Per Liora rules (>2 GB and Kaggle competition acceptance), the dataset is **not downloaded**. The exact `kaggle competitions download` command and post-download layout are documented in `data/README.md`.

## Target

Bounding-box localisation of pneumonia opacities. Evaluation metric is mean Average Precision averaged over IoU thresholds {0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75} (the official RSNA challenge metric). For this scaffold the baseline reports mAP@0.5 to keep the implementation auditable.

## Deliverables

- [x] `brief.md` (this file)
- [x] `data/README.md` - Kaggle CLI command + setup notes
- [x] `notebooks/01_EDA.ipynb` - DICOM loading, label distribution, image checks (skeleton, not executed)
- [x] `reports/references.md` - 20+ verified references
- [x] `src/model_baseline.py` - RetinaNet ResNet50-FPN + focal loss
- [x] `src/model_advanced.py` - YOLOv8 (ultralytics) with CLAHE pre-processing and heavy augmentation
- [x] `manuscripts/manuscript.md` - IMRaD draft (~4,500 words)
- [x] `deliverables/presentation.html` - 8-12 slide self-contained HTML
- [x] `checkpoint.json` - Phase 1 status

## Open questions

1. Detection vs classification framing: stick with detection (bounding box) per challenge, but the manuscript will discuss the classification-only fallback used in many top-10 RSNA solutions.
2. Class imbalance handling: opacity-positive rate is roughly 22-30 percent of patient IDs. Focal loss in baseline; class-weighted sampling and mixup in advanced.
3. Single-stage vs two-stage detector: baseline uses single-stage (RetinaNet) per Lin 2017 focal-loss motivation; advanced compares one-stage YOLOv8 to two-stage Faster R-CNN as a discussion point.
4. Test-time augmentation: standard horizontal-flip TTA is in scope for advanced; multi-scale TTA out of scope for Phase 1.
