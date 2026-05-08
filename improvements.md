# Improvements - Project 15 RSNA Pneumonia Detection

## Top recommendation

**Add a two-stage classifier-then-detector pipeline (CheXNet-style binary classifier as a gate, RetinaNet/YOLOv8 as the localiser) and report both end-to-end mAP and the gated mAP.** Every top-10 RSNA Kaggle solution used this pattern because the detector wastes capacity learning to suppress confident negatives that a cheap classifier already filters out. The Phase 1 scaffold mentions this in Section 5.1 of the manuscript but does not implement it. Concrete next step: add `src/model_classifier_gate.py` that fine-tunes a DenseNet121 (CheXNet weights from Rajpurkar 2018) for binary opacity-vs-no-opacity, and rewrite `model_advanced.run()` to chain `classifier.predict() -> detector.predict()` with a tunable score threshold reported alongside mAP. Expected lift on this dataset is +0.03 to +0.08 mAP at IoU 0.5 versus the detector-only path, based on the published Kaggle leaderboard write-ups. Priority: HIGH.

## Detailed weaknesses and proposals

### 1. No requirements.txt and no environment pinning - HIGH
The scripts import torch, torchvision, ultralytics, pydicom, opencv, albumentations, sklearn, pandas, numpy, tqdm. None of these are pinned. Phase 2 will fail silently when ultralytics or torchvision bumps a default. Action: add `requirements.txt` with exact versions (e.g. `torch==2.3.1`, `torchvision==0.18.1`, `ultralytics==8.2.50`, `pydicom==2.4.4`, `opencv-python==4.10.0.84`, `albumentations==1.4.10`) and a one-line `python -m pip install -r requirements.txt` reproducer block in `data/README.md`. Add a `python_requires` note (3.10+ recommended).

### 2. No cross-validation, no confidence intervals on mAP - HIGH
A single 85/15 split with seed 42 gives one mAP value with no uncertainty. RSNA splits at the patient level have 5-7 percent absolute mAP variance across seeds (see RSNA challenge post-mortems). Action: replace the single split with a 5-fold patient-level stratified GroupKFold (using the detailed-class CSV as the stratification target so the three-class structure is preserved), report mean and 95 percent percentile-bootstrap CI of mAP across folds. This is roughly 30 lines of code in `model_baseline.py` and gives a defensible point estimate.

### 3. No external-validation hook despite the manuscript flagging Zech 2018 - HIGH
The manuscript Section 5.7 flags external validation on CheXpert / MIMIC-CXR / PadChest as Phase 2 work, but no scaffold exists. Action: add an `src/external_eval.py` stub that loads a held-out external CSV (path passed via CLI), maps the external opacity/pneumonia label to the RSNA opacity class, runs the trained detector, and writes a separate metrics JSON. Even an unexecuted stub binds the team to the analysis plan and prevents the common drift where external eval gets dropped because no code exists to run it.

### 4. Calibration completely absent - MEDIUM
RetinaNet and YOLOv8 scores are not probabilities. The manuscript mentions temperature scaling in Section 5.7 but no code is provided. Action: add `src/calibration.py` with two functions: `temperature_scale(logits, labels)` returning a single learned scalar T, and `expected_calibration_error(scores, labels, n_bins=15)`. Report ECE alongside mAP, and a reliability diagram in the deliverables. This is also the cheapest fix that improves clinical credibility because radiologists routinely ask "what does a 0.7 score mean here?".

### 5. Demographic-stratified evaluation is named but not extracted - MEDIUM
Section 4.5 lists stratified error analysis (view position, age band, detailed class) as placeholders, but no code reads `ViewPosition`, `PatientAge`, `PatientSex` from DICOM headers. Action: extend `load_dicom_image` with a sibling `extract_dicom_meta(path) -> dict` that pulls `ViewPosition`, `PatientAge`, `PatientSex`, `Modality`, `Manufacturer`. Save the per-patient metadata to a CSV during dataset preparation so stratification at evaluation time is a join, not a re-read of every DICOM. Quantify missingness as a single number in the manuscript.

### 6. CLAHE clip-limit and tile size are unjustified hard-codes - MEDIUM
Both scripts use `clipLimit=2.0, tileGridSize=(8,8)`. The manuscript calls this "a defensible default rather than a tuned hyperparameter". For a portfolio piece this is a missed easy win. Action: run a small grid (`clip in [1.0, 2.0, 3.0, 4.0]`, `tile in [(4,4), (8,8), (16,16)]`) on a 10 percent subsample for the baseline only and pick the best. Or, better, drop the heuristic and adopt the histogram-equalisation defaults from torchxrayvision (`xrv.datasets.normalize`) which has been validated across several CXR datasets. Cite Cohen 2022 (TorchXRayVision).

### 7. Backbone choice is dated - MEDIUM
ResNet50-FPN was state-of-the-art in 2017. For 2026 portfolio credibility, swap or at least benchmark against a modern backbone. Action: in `model_advanced.py`, add a `train_yolov9()` or `train_rt_detr()` variant. RT-DETR (Lv 2024, DOI 10.1109/CVPR52733.2024.01605) is the strongest one-stage detector with publicly available code in Ultralytics, and it gets +1.5 to +2.5 mAP over YOLOv8s on COCO. Cite the RT-DETR paper. Do NOT remove YOLOv8 - keep it as the baseline-comparable model.

### 8. No Grad-CAM / objectness visualisation despite Section 5.7 promising it - LOW
Section 5.7 promises interpretability artefacts. None are produced. Action: add `src/explain.py` with `gradcam_for_detector(model, image, target_class=1)` using `pytorch-grad-cam` library. Generate a 16-image mosaic of true positives, false positives, and false negatives saved as `deliverables/explain_mosaic.png`. This single PNG dramatically improves the presentation deck and is a roughly 40-line script.

### 9. Presentation HTML claims numbers but Phase 1 has none - LOW
The presentation deck cannot show real metrics until Phase 2 runs. Action: rebuild the presentation HTML so it explicitly says "Phase 1 scaffold - numbers pending" on the metrics slide, and adds a "Methodology audit" slide that lists the eight planned ablations (CV folds, TTA, classifier gate, RT-DETR, calibration, Grad-CAM, external eval, CLAHE grid). This honestly signals scope to the client and prevents the awkwardness of a deck full of `<TBD>` placeholders.

### 10. Random-seed coverage is incomplete - LOW
`SEED = 42` is set for `torch.manual_seed`, `np.random`, and `train_test_split`, but `cv2`, `albumentations`, and the Ultralytics dataloader workers are not seeded. Action: add `torch.use_deterministic_algorithms(True)`, set `os.environ["PYTHONHASHSEED"]`, set `albumentations.ReplayCompose` with seed, and pin Ultralytics `seed=42, deterministic=True` (already partly done, confirm). Document the residual non-determinism (cuDNN convolution algorithm selection) in a NOTE block at the top of each script.

## Score summary

| Improvement | Priority |
|---|---|
| 1. requirements.txt and env pinning | HIGH |
| 2. CV folds and CI on mAP | HIGH |
| 3. External-validation stub | HIGH |
| Top recommendation: classifier-gate two-stage pipeline | HIGH |
| 4. Calibration (temperature scaling, ECE) | MEDIUM |
| 5. Demographic stratification code | MEDIUM |
| 6. CLAHE hyperparameter grid | MEDIUM |
| 7. RT-DETR / YOLOv9 backbone benchmark | MEDIUM |
| 8. Grad-CAM mosaic | LOW |
| 9. Presentation slide rebuild | LOW |
| 10. Full random-seed pinning | LOW |
