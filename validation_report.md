# Validation Report - Project 15 RSNA Pneumonia Detection

## Summary

**Overall status: PASS-WITH-WARNINGS**

Scaffold artefacts are structurally sound: notebook JSON parses, both Python scripts compile, presentation HTML is fully self-contained (zero external resources), em-dash count is zero, no AI-tell phrases present, and checkpoint schema covers all four required keys. All 25 unique inline citations map to entries in `reports/references.md` and 5 randomly sampled DOIs resolve live on CrossRef with matching titles. IMRaD coverage is complete. Methods named in the manuscript (RetinaNet, ResNet50-FPN, focal loss, CLAHE, YOLOv8, Faster R-CNN, AdamW, cosine schedule, mosaic, mixup) are all present in the model scripts. Two warnings: manuscript word count is 4,104 (inside 4,000-5,000 target but near the lower bound), and `checkpoint.json` uses the key `phase` instead of (or alongside) the spec's expected schema; `project_number`, `title`, `methodology`, `status` are all present.

---

## Findings

### 1. Notebook validity
- [PASS] `notebooks/01_EDA.ipynb` parses as valid JSON.

### 2. Python script syntax
- [PASS] `src/model_baseline.py` parses with `ast.parse` (no SyntaxError).
- [PASS] `src/model_advanced.py` parses with `ast.parse` (no SyntaxError).

### 3. Manuscript word count
- [PASS] `manuscripts/manuscript.md` is 4,104 words (target 4,000-5,000). Inside range, but on the lower bound.

### 4. Self-contained HTML
- [PASS] `deliverables/presentation.html` has 0 hits for `href="http` or `src="http`. No external resources; fully inline.

### 5. IMRaD completeness
- [PASS] Title present (line 1).
- [PASS] Abstract present (Section "Abstract").
- [PASS] Introduction present (Section 1).
- [PASS] Methods present (Section 3).
- [PASS] Results present (Section 4).
- [PASS] Discussion present (Section 5).
- [PASS] Conclusion present (Section 6).
- [PASS] References present (Section "References", with pointer to `reports/references.md`).
- Note: an additional Section 2 "Background and related work" sits between Introduction and Methods, which is consistent with extended IMRaD.

### 6. Method drift
Methods named in manuscript Methods section (3.1-3.7), each verified against `src/`:
- [PASS] DICOM read with `pydicom` -> present in `model_baseline.py` (line 38, `import pydicom`) and `model_advanced.py`.
- [PASS] CLAHE clipLimit=2.0, tileGridSize=(8,8) -> present in baseline line 76 and advanced line 75.
- [PASS] RetinaNet ResNet50-FPN-v2 -> baseline line 39, line 144 (`retinanet_resnet50_fpn_v2`).
- [PASS] Focal-loss bias init `-4.6` (= `-log((1-0.01)/0.01)`) -> baseline line 152.
- [PASS] AdamW lr=1e-4, weight_decay=1e-4 -> baseline line 240, advanced line 208.
- [PASS] CosineAnnealingLR -> baseline line 243.
- [PASS] YOLOv8s with mosaic=1.0, mixup=0.15 -> advanced lines 138, 148, 149.
- [PASS] Faster R-CNN ResNet50-FPN-v2 comparator -> advanced line 179, 203 (`fasterrcnn_resnet50_fpn_v2`).
- [PASS] mAP@0.5 evaluation with eleven-point interpolation -> baseline line 171 (`average_precision_at_iou`).
- [PASS] Patient-level 85/15 split with seed=42 -> referenced in manuscript Section 3.2; seed pinning present in scripts.

### 7. Citation drift
- 25 unique inline citations extracted: Doi 2007, Dreyer 2017, Everingham 2009, He 2016, He 2017, He 2019, Huang 2017, Krizhevsky 2017, Lakhani 2017, LeCun 2015, Lin 2017a, Lin 2017b, Litjens 2017, Liu 2016, Majkowska 2020, Rajpurkar 2018, Redmon 2016, Redmon 2017, Ren 2017, Ronneberger 2015, Sandler 2018, Shih 2019, Shin 2016, Wang 2017, Zech 2018.
- [PASS] All 25 map to entries in `reports/references.md` (Lin 2017a -> ref 1 Focal Loss; Lin 2017b -> ref 2 FPN).
- [PASS] Zero orphan citations.

### 8. Re-verify 5 random references against CrossRef
Sampled with `random.seed(42)` from the 26 DOIs in `references.md`:
- [PASS] 10.1109/TMI.2016.2528162 -> HTTP 200, title "Deep Convolutional Neural Networks for Computer-Aided Detection: CNN Architectures, Dataset Characteristics and Transfer..." (matches Shin 2016 entry).
- [PASS] 10.1109/CVPR.2016.91 -> HTTP 200, title "You Only Look Once: Unified, Real-Time Object Detection" (matches Redmon 2016 entry).
- [PASS] 10.1109/ICCV.2017.324 -> HTTP 200, title "Focal Loss for Dense Object Detection" (matches Lin 2017a entry).
- [PASS] 10.1148/radiol.2017171183 -> HTTP 200, title "When Machines Think: Radiology's Next Frontier" (matches Dreyer 2017 entry).
- [PASS] 10.1109/CVPR.2017.243 -> HTTP 200, title "Densely Connected Convolutional Networks" (matches Huang 2017 entry).

### 9. Em-dash scan
Per-file count of U+2014 across `brief.md`, `notebooks/01_EDA.ipynb`, `reports/references.md`, `src/model_baseline.py`, `src/model_advanced.py`, `manuscripts/manuscript.md`, `deliverables/presentation.html`:
- [PASS] Total em-dash count = 0.

### 10. AI-tell scan
- [PASS] `grep -riE 'verified by [0-9]+ agents|AI-verified|cross-checked by Claude'` in project root returns zero hits.

### 11. Checkpoint schema
- Keys present: `project_number`, `title`, `methodology`, `phase`, `status`, `needs_main_session_execution`, `blockers`.
- [PASS] `project_number` present.
- [PASS] `title` present.
- [PASS] `methodology` present.
- [PASS] `status` present.
- [WARN] Key `phase` is present in addition to `status`; this is a benign extension. All four required keys are accounted for.

### 12. Saved-model artefacts (project #15 is scaffold-only)
- [PASS-NA] Project #15 is not in the executed range (#1-#8), so absence of saved `.pt` / `.pkl` weights under `deliverables/` is expected. Only `presentation.html` is present, which is correct.

---

## Blockers
- None.

## Role A complete
