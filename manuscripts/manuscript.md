# Localising Pneumonia Opacities on Frontal Chest Radiographs: A Comparative Study of Single-Stage and Two-Stage Object Detectors on the RSNA 2018 Challenge Dataset

**Authors:** Sandeep Grover, Liora MLE Programme, Cohort 6973

## Abstract

Pneumonia is among the most common causes of paediatric and adult hospital admission worldwide, and chest radiography remains the first-line imaging modality despite its known interpretive variability. The 2018 Radiological Society of North America (RSNA) Pneumonia Detection Challenge crowdsourced a curated subset of the National Institutes of Health (NIH) ChestX-ray dataset, re-annotated by a panel of board-certified radiologists with bounding boxes around opacities consistent with pneumonia. The challenge framed pneumonia detection as an object-detection problem rather than the more common image-level classification, foregrounding spatial localisation as a clinically useful auxiliary signal. We re-implement two competitive detection pipelines on this dataset: a baseline RetinaNet with a ResNet50 backbone and Feature Pyramid Network trained with focal loss [Lin 2017a, Lin 2017b], and an advanced YOLOv8 detector with Contrast Limited Adaptive Histogram Equalisation (CLAHE) pre-processing and heavy augmentation. We compare both against a two-stage Faster R-CNN [Ren 2017] reference. Mean Average Precision at IoU 0.5 is reported for the baseline; the advanced configuration reports the official RSNA challenge metric averaged over IoU 0.40 to 0.75. Beyond raw performance, the manuscript audits the well-documented dataset-bias and generalisation risks of chest-radiograph models [Zech 2018, Majkowska 2020], situates the task within the broader medical-imaging literature [Litjens 2017, Shin 2016, Doi 2007], and contrasts the medical-imaging detection regime with the industrial-defect twin (Severstal steel) of this cohort. Numerical results are placeholders pending Phase 2 model runs and are reported as `<TBD after model run>` throughout. (Word count for abstract: approximately 250.)

## 1. Introduction

Chest radiography (CXR) is the highest-volume medical imaging examination in most health systems, and pneumonia is among its commonest indications. Despite extensive clinical experience, inter-radiologist agreement on the presence and extent of pneumonic opacities is moderate at best, and demand frequently outstrips the supply of subspecialty-trained chest radiologists. Computer-aided diagnosis (CAD) for chest imaging therefore has a long history [Doi 2007], with deep-learning-based approaches dominating since the publication of CheXNet-style classifiers and the release of the NIH ChestX-ray dataset [Wang 2017]. The 2018 RSNA Pneumonia Detection Challenge [Shih 2019] extended this line of work in two ways: it cleaned a subset of the NIH dataset by panel-adjudicated relabelling, and it reframed pneumonia detection as a bounding-box localisation task rather than a multilabel classification problem. The localisation framing is clinically meaningful: a heatmap or box that aligns with a radiologist's reading is far more interpretable than a scalar output, and supports workflows in which the model assists rather than replaces the radiologist.

Object detection in computer vision is a mature subfield, with two main families: two-stage detectors that first propose regions and then classify them (e.g. Faster R-CNN [Ren 2017], Mask R-CNN [He 2017]), and single-stage detectors that predict boxes and classes in a single forward pass (e.g. YOLO [Redmon 2016], SSD [Liu 2016], RetinaNet [Lin 2017a]). Single-stage detectors historically traded accuracy for speed; the introduction of focal loss in RetinaNet closed much of the accuracy gap by addressing the foreground-background class imbalance that dense detectors face during training. Feature Pyramid Networks (FPN) [Lin 2017b] became a standard backbone augmentation across both families, and ResNet [He 2016] and DenseNet [Huang 2017] are now default backbones for medical-imaging detection.

This manuscript reports a Phase 1 scaffold for a comparative study on the RSNA dataset. We motivate the task clinically, describe the data and its known biases [Zech 2018], detail two reproducible training pipelines (RetinaNet baseline; YOLOv8 advanced with optional Faster R-CNN comparator), and discuss expected results and limitations. The pipeline mirrors the structure of the cohort's industrial-anomaly twin (Severstal steel surface defects), making the comparison between industrial and medical computer vision concrete. Both regimes share heavy class imbalance, imperfect ground truth, and high deployment cost for false negatives; they differ sharply on photometric properties, annotator consistency, and regulatory exposure.

The contribution of Phase 1 is therefore not new state-of-the-art numbers (none are claimed), but a clean, auditable scaffold consistent with the project rules: documented data access, an EDA notebook describing the analytic plan, two runnable but unexecuted modelling scripts, and a manuscript that can be filled with numbers from Phase 2 model runs.

A note on framing: the RSNA dataset can be approached as either a classification or a detection problem, and many top-performing solutions in the original Kaggle leaderboard were ensembles that combined a strong classifier (does this radiograph contain pneumonia?) with a separate detector (where is it?). The classification head filters out negative images cheaply and reduces the number of false-positive boxes the detector has to worry about; the detector localises the finding for the radiologist. We discuss the merits of this two-stage pipeline in Section 5 but do not implement it in Phase 1 to keep the methodological comparison crisp. The single-stage RetinaNet baseline and YOLOv8 advanced pipeline are both end-to-end detectors with no separate classification stage.

## 2. Background and related work

### 2.1 Object-detection architectures

Single-shot detectors trade off accuracy for inference speed [Redmon 2016, Redmon 2017, Liu 2016]. The first YOLO formulation [Redmon 2016] reframed detection as a single regression on a coarse spatial grid, exchanging some localisation accuracy for a step-change in inference latency. YOLO9000 [Redmon 2017] introduced anchor priors derived by k-means clustering of training-set boxes, multi-scale training, and a joint detection-classification objective; that combination of changes is still visible in the YOLOv8 head used in this study. SSD [Liu 2016] generalised the single-stage idea to multi-scale feature maps with convolutional default boxes, and is widely deployed in production embedded settings.

RetinaNet [Lin 2017a] introduced focal loss to down-weight easy negatives, eliminating the standard hard-negative-mining step required by SSD and competing with two-stage detectors on accuracy. The focal-loss bias initialisation matters: the head bias is initialised to `-log((1 - pi) / pi)` for a small prior `pi` (commonly 0.01), a detail that, if missed, causes the loss to diverge in the first few iterations as the dense head over-predicts foreground. FPN [Lin 2017b] attaches lateral connections to a backbone, producing multi-scale features at near-constant compute, and is now standard in nearly all dense detectors regardless of whether the head is one-stage or two-stage.

Faster R-CNN [Ren 2017] remains the canonical two-stage detector and is competitive on small objects, especially when paired with FPN. The two-stage design (region-proposal network followed by per-proposal classification and refinement) is slower per image than RetinaNet or YOLO but tends to localise small objects more precisely because the second stage operates on cropped features. Mask R-CNN [He 2017] extends Faster R-CNN with an instance-segmentation head; while pneumonia opacities are not annotated as masks in RSNA, Mask R-CNN is commonly used as a detection-only backbone because of its strong RPN and the option to convert weakly-supervised opacity boxes into pseudo-masks. YOLOv8 (Ultralytics, 2023) is the latest in the YOLO family and ships a feature-pyramid neck, anchor-free heads, decoupled classification and box-regression branches, and a tightly engineered training loop, making it a strong default for practitioners. Backbone choices matter: ResNet50 [He 2016] is the de-facto default and the only backbone we use here; DenseNet [Huang 2017] performs well on medical imaging due to its feature-reuse pattern and is the backbone of the original CheXNet pipeline; MobileNetV2 [Sandler 2018] is an efficient option for edge deployment, useful when a CXR model has to run on a portable digital-radiography unit rather than a hospital GPU.

### 2.2 Deep learning on chest radiographs

Modern CXR pipelines descend from a small canonical literature. Krizhevsky 2017 [Krizhevsky 2017] established the practical use of deep convolutional networks for natural-image classification, and the resulting architectural conventions (alternating convolution and pooling, ReLU activations, dropout regularisation, ImageNet pretraining) propagated rapidly into medical-imaging research. LeCun 2015 [LeCun 2015] reviewed the architectural ideas behind that wave and is still the most-cited general introduction to deep learning in medical-imaging methods sections.

Lakhani and Sundaram [Lakhani 2017] applied deep CNNs to pulmonary tuberculosis classification on chest radiographs; their study is one of the earliest demonstrations that an off-the-shelf ImageNet-pretrained network can produce radiologist-comparable performance on a chest-radiograph task with a few thousand training images. Wang et al. [Wang 2017] released ChestX-ray8, the foundation of the broader CXR benchmark community: roughly one hundred thousand frontal radiographs with weakly-supervised labels mined from radiology reports. Although the labels are noisy, the scale of ChestX-ray8 made supervised pretraining feasible for the first time on radiographs.

Rajpurkar et al. [Rajpurkar 2018] demonstrated that a CNN classifier could match or exceed practising radiologists on selected chest-radiograph tasks; the methodology has since become a template for radiologist-comparison studies, and the limitations of that template (selected tasks, limited reader pool, single dataset) are themselves an active area of methodological discussion. Litjens et al. [Litjens 2017] surveyed the broader medical-imaging deep-learning field, providing a useful taxonomy of tasks (classification, detection, segmentation, registration, retrieval) that maps cleanly onto modern pipelines. Shin et al. [Shin 2016] discussed how transfer learning with ImageNet pretraining transfers to medical imaging, with particular attention to layer-wise fine-tuning strategies and architecture choice. Doi 2007 [Doi 2007] frames the entire CAD enterprise historically: the field predates deep learning by two decades, and many of the lessons (operator dependence, false-alarm fatigue, integration into reporting workflow) survived the transition unchanged.

### 2.3 The RSNA challenge dataset

The RSNA Pneumonia Detection Challenge dataset [Shih 2019] is a re-annotated subset of NIH ChestX-ray, with bounding-box ground truth for opacities consistent with pneumonia. The dataset comprises roughly 26,684 patient studies in DICOM format, with one frontal view per patient. The detailed-class CSV exposes a three-class structure that the binary labels hide: `Normal`, `No Lung Opacity / Not Normal` (abnormal but non-pneumonia, e.g. cardiomegaly, effusion, atelectasis), and `Lung Opacity` (the positive class). A binary model that does not see the middle class will silently underperform on real clinical mixes.

### 2.4 Generalisation and dataset bias

A persistent finding in CXR deep learning is that models that perform well on the training-distribution test set often degrade markedly on radiographs from new sites. Zech et al. [Zech 2018] showed that a pneumonia-detection CNN exploited site-specific tokens (e.g. chest tubes, view position, image markers, scanner-specific image-header patterns) rather than disease features, and that performance on a held-out hospital was substantially worse than performance on the same hospital's test split. The implication is uncomfortable: aggregate test mAP on the original distribution is a weak indicator of clinical performance on a new deployment site.

Majkowska et al. [Majkowska 2020] examined radiologist-adjudicated reference standards and population-adjusted evaluation, arguing for both more rigorous ground truth and demographic-stratified reporting. They show that the same model, evaluated on the same dataset, can look better or worse depending on whether the prevalence in the test split matches the deployment prevalence. This is a particularly important finding for pneumonia detection because the prevalence in a busy emergency department is much higher than the prevalence in routine outpatient imaging. He et al. [He 2019] discussed the practical implementation of medical AI in the clinic, foregrounding bias, workflow integration, regulatory pathways, and the often-overlooked question of who is liable when the model is wrong. Dreyer and Geis [Dreyer 2017] argued for a measured deployment posture in radiology AI more broadly, distinguishing between assistive use (where the radiologist remains the decision-maker) and autonomous use (where they do not). We treat all of these findings as binding constraints on what Phase 2 results can claim, and the limitations section makes the distinction between intra-dataset and external performance explicit.

### 2.5 Medical-imaging segmentation and adjacent tasks

Although RSNA is a detection task, the segmentation literature is closely adjacent. U-Net [Ronneberger 2015] is the canonical biomedical segmentation architecture and underpins many CXR pipelines that emit pixel-level outputs. The Pascal VOC challenge [Everingham 2009] established the modern detection-evaluation conventions (mAP, IoU thresholds) reused here.

## 3. Methods

### 3.1 Data preparation

The RSNA challenge ships DICOM imagery with 16-bit pixel encoding. Each DICOM is read with `pydicom`, the pixel array is min-max normalised to `[0, 1]`, scaled to uint8, resized to 512 pixels (baseline) or 640 pixels (advanced), and CLAHE-equalised with a clip limit of 2.0 and an 8x8 tile grid. CLAHE is the standard contrast-enhancement step for CXR pipelines and is documented in the EDA notebook (Section 7) with a before/after preview. The clip-limit choice of 2.0 is a defensible default rather than a tuned hyperparameter; the literature [Litjens 2017, Shin 2016] reports that values in the 1.5 to 3.0 range produce comparable downstream performance.

The grayscale image is replicated across three channels before being passed to the ImageNet-pretrained backbone. This is wasteful in storage but standard practice: it avoids the need to retrain the input convolution from scratch, which would discard the ImageNet-learned low-level edge filters. Alternative single-channel-input strategies (averaging the three input convolutions, learning a 1x1 reduction) were not explored in Phase 1 and are flagged in Section 5.7 as a small but easy ablation.

Bounding-box coordinates are scaled from the original 1024-pixel resolution to the model input resolution. For the YOLOv8 pipeline, boxes are normalised to `[0, 1]` and written in the YOLO label format `class cx_norm cy_norm w_norm h_norm` to a per-image text file alongside the converted PNG. The Ultralytics tooling ingests the dataset YAML at training time and constructs the train/val splits from the per-image label paths; we deliberately produce that layout from the canonical CSV in `prepare_yolo_dataset()` rather than at training time, so that the conversion step is auditable and reproducible.

### 3.2 Train/validation split

We use a 85/15 patient-level stratified split with the random seed fixed at 42. Stratification is on the per-patient maximum of the `Target` column. Splits never cross patients, and the detailed-class metadata is preserved alongside each split for downstream stratified error analysis.

### 3.3 Baseline: RetinaNet ResNet50-FPN with focal loss

The baseline is the torchvision reference RetinaNet ResNet50-FPN-v2 detector with ImageNet-pretrained backbone weights. The classification head is replaced for the two-class problem (background, opacity) with the focal-loss bias initialisation `b = -log((1 - pi) / pi)` for `pi = 0.01`, evaluated as `-4.6` (per Lin 2017a). The model is trained with AdamW at learning rate `1e-4`, weight decay `1e-4`, batch size 4 (memory-bound at 512x512), six epochs, and a cosine learning-rate schedule. Evaluation is at IoU 0.5 with eleven-point interpolated AP averaged across validation images.

### 3.4 Advanced: YOLOv8 with heavy augmentation

The advanced pipeline uses Ultralytics YOLOv8s, trained on PNG-converted CLAHE images at 640x640, batch size 8, 30 epochs, with Ultralytics's stock augmentation (mosaic 1.0, mixup 0.15, translate 0.1, scale 0.5, horizontal flip 0.5, no vertical flip, no HSV hue/saturation perturbation, brightness perturbation 0.4). Vertical flips are disabled because frontal anatomy is asymmetric (heart on the left, gastric bubble on the left under the diaphragm), and a vertical flip changes the disease prior. Early stopping after ten patience epochs.

### 3.5 Optional: Faster R-CNN comparator

A two-stage Faster R-CNN ResNet50-FPN-v2 [Ren 2017] is implemented as `train_faster_rcnn()` in the advanced script, sharing the dataset class with the baseline. Eight epochs at AdamW `1e-4`, batch size 4. Used only for like-for-like comparison and not the primary advanced model.

### 3.6 Evaluation metrics

For the baseline we report mAP@0.5 as a single, easy-to-audit number. For the advanced model we report both mAP@0.5 and mAP@0.5:0.95 (Pascal-style), and the official RSNA challenge metric (mean AP over IoU 0.40 to 0.75 in 0.05 steps). The Pascal-VOC eleven-point interpolation [Everingham 2009] is used in the auditable single-IoU code path; Ultralytics provides the COCO-style metric internally.

### 3.7 Reproducibility

All seeds are pinned to 42. Pre-processing is deterministic. The pipeline writes:

- `deliverables/retinanet_baseline.pt` (RetinaNet weights)
- `deliverables/metrics_baseline.json`
- `deliverables/yolov8_rsna/` (Ultralytics run directory with weights and plots)
- `deliverables/metrics_advanced.json`

## 4. Results

Phase 1 deliverables are scaffolded but not executed. Numerical results are reported below as placeholders. The placeholders will be filled in Phase 2 by re-running the scripts and reading the resulting JSON files programmatically into this manuscript at build time, per the project's no-hallucination rule.

### 4.1 Dataset summary (placeholder)

- Total patient studies: <TBD after EDA run>
- Positive (Lung Opacity) patient fraction: <TBD after EDA run>
- Box count: <TBD after EDA run>
- Median box area as fraction of image: <TBD after EDA run>
- AP/PA view balance: <TBD after EDA run>

### 4.2 Baseline RetinaNet (placeholder)

- Validation mAP@0.5: <TBD after model run>
- Mean training loss at final epoch: <TBD after model run>
- Inference time per image (CPU, 512x512): <TBD after model run>

### 4.3 Advanced YOLOv8 (placeholder)

- Validation mAP@0.5: <TBD after model run>
- Validation mAP@0.5:0.95: <TBD after model run>
- RSNA challenge metric (mean AP over IoU 0.40-0.75): <TBD after model run>

### 4.4 Faster R-CNN comparator (placeholder)

- Validation mAP@0.5: <TBD after model run>
- Inference latency vs YOLOv8: <TBD after model run>

### 4.5 Stratified error analysis (placeholder)

- Recall on `No Lung Opacity / Not Normal` confounders: <TBD after model run>
- Recall by view position (AP vs PA): <TBD after model run>
- Recall by patient age band: <TBD after model run>

Tables and figures referenced by name in this section will be produced by Phase 2 scripts and saved alongside the metrics JSON: `Figure 1 - per-class precision-recall curves`, `Figure 2 - score-distribution histogram`, `Figure 3 - error mosaic`, `Table 1 - mAP by detector`, `Table 2 - mAP by stratum`.

## 5. Discussion

### 5.1 Architecture choice

We chose RetinaNet as the baseline because focal loss directly addresses the foreground-background imbalance that single-stage dense detectors face on RSNA. The focal-loss bias initialisation matters: without `-4.6`, the loss explodes in the first few iterations as the dense head over-predicts foreground. YOLOv8 was selected for the advanced model because the opacities are large relative to the image and YOLOv8's anchor-free head plus mosaic/mixup augmentation works well in this regime; a Faster R-CNN comparator is included to keep the methodological contrast between one-stage and two-stage detectors honest. Detectron2 RetinaNet/Faster-RCNN was not used in Phase 1 because of installation complexity (compiled CUDA ops) and is flagged as a Phase 2 candidate.

### 5.2 Dataset bias

The Zech et al. [Zech 2018] result is the elephant in the room for any chest-radiograph deep-learning manuscript. Our scaffold mitigates the most obvious failure modes by stratifying the validation split on the patient-level positive label and preserving the detailed-class info column for stratified error analysis. We do not claim that intra-dataset performance generalises to other hospitals or scanners. The Majkowska et al. [Majkowska 2020] critique of population-adjusted evaluation also applies: aggregate mAP can hide systematic underperformance on subgroups. Phase 2 should report stratified AP by view position, age band, and detailed-class membership.

### 5.3 Ground-truth quality

The RSNA labels are panel-adjudicated and therefore of higher quality than the original ChestX-ray weak labels. Inter-annotator agreement on bounding-box extent, however, is not perfect: opacities are diffuse, edges are ambiguous, and box geometry is a coarse approximation of true findings. This bounds the achievable IoU even for an oracle model. Discussion of this issue appears in the original RSNA dataset descriptor [Shih 2019] and is consistent with the broader medical-imaging-CAD literature [Doi 2007, Litjens 2017].

### 5.4 Comparison with the industrial-defect twin (Severstal)

The Severstal steel-defect competition (project 10 in this cohort) and RSNA pneumonia (this project) are both large-image, class-imbalanced, IoU-evaluated detection or segmentation tasks. The technical differences are instructive. Industrial defects have crisp edges, deterministic illumination, and replicable ground truth: a defect either is or is not in a region. Medical opacities have soft, diffuse edges and probabilistic ground truth: two radiologists can disagree on extent and even presence. The implication is that competitive Severstal pipelines extract more value from heavy augmentation and segmentation heads, while competitive RSNA pipelines extract more value from CLAHE, demographic stratification, and panel-adjudicated labels. The same architecture (e.g. ResNet50-FPN backbone) is competitive in both, but the failure modes differ: industrial models fail when illumination changes, medical models fail when site or scanner changes [Zech 2018].

A concrete example of the regime gap: in the Severstal task, a 10 percent random crop with brightness perturbation usually improves test accuracy because it forces the model to be invariant to acquisition conditions that vary on the steel mill. The same crop applied to a frontal CXR can rotate a pneumonia opacity out of view or chop the costophrenic angle, both of which are clinically meaningful losses of context. Medical-imaging augmentation must therefore be more conservative and anatomy-aware. Vertical flip is the most obvious trap: it is harmless in industrial CV and disastrous in CXR, where it flips the heart silhouette. Our YOLOv8 configuration disables vertical flip explicitly for this reason. A second example: industrial pipelines tolerate aggressive colour jitter, while CXR pipelines are single-channel and gain nothing from it. We replicate the grayscale-equivalent perturbation with brightness only.

### 5.5 Comparison with the COVID chest-X-ray neighbour (project 8)

Project 8 in this cohort tackles COVID-19 chest-X-ray classification on a different dataset. The methodological lessons travel: same pre-processing (CLAHE), same backbone family, same dataset-bias risk, but a classification rather than detection framing. Cross-pollination between projects 8 and 15 is therefore high; both should cite the same Zech 2018 critique and the same Majkowska 2020 evaluation framework.

### 5.6 Limitations

1. Single dataset, single site distribution. External validation is out of scope.
2. The RSNA test set is held back by Kaggle and not used here; the public validation split is the only number we can report.
3. No ensembling, no test-time augmentation beyond Ultralytics defaults, no multi-scale inference.
4. Demographic stratification depends on DICOM headers being consistent across the dataset; missingness has not been quantified.
5. mAP averages can hide localisation-quality issues; per-image error inspection is needed alongside aggregate metrics.

### 5.7 Future work

Three Phase 2 directions are worth flagging:

- **Self-supervised pretraining on unlabelled CXR.** A growing literature shows that contrastive or masked-autoencoder pretraining on large unlabelled CXR collections improves downstream detection sample efficiency, sometimes by a factor of two or three on the labelling budget. Pretraining on the unlabelled NIH ChestX-ray cohort (the parent dataset of RSNA) would be a controlled experiment because the data lineage is explicit.
- **Radiologist-in-the-loop evaluation.** A modest user study (five radiologists, fifty cases each, mAP-paired and reading-time-paired) would substantially improve the credibility of any deployment claim, even one limited to assistive use. The Majkowska 2020 [Majkowska 2020] reader study is a useful template, although run at a much smaller scale than what they report.
- **External validation on a second public dataset.** CheXpert, MIMIC-CXR, or PadChest are obvious candidates; the picture they give is consistent with [Zech 2018] and would be informative. The Phase 2 plan is to keep RSNA as the training distribution and use one external dataset, with re-mapped opacity labels, as a held-out test set. Any reported deployment claim should rest on the external metric, not the in-distribution one.

A secondary thread worth flagging is calibration. Detection scores from RetinaNet and YOLOv8 are not calibrated probabilities by default, and a poorly calibrated score is harder to fold into a clinical-decision-support workflow. Temperature scaling on the validation split is the cheapest fix; isotonic regression on a held-out calibration split is the more conservative one. Neither is in scope for Phase 1 but both are short, isolated changes for Phase 2.

A tertiary thread is interpretability. Grad-CAM and other attribution methods are easy to add to a CNN classifier; for a detector, the equivalent is showing the per-anchor focal-loss heatmap or the YOLO objectness map. These visualisations are imperfect but help radiologists triage false positives, and the cost of producing them is low because the data is already in memory at inference time.

## 6. Conclusion

This manuscript scaffolds a comparative study of pneumonia-opacity detection on the RSNA 2018 dataset. The baseline (RetinaNet ResNet50-FPN with focal loss) and the advanced model (YOLOv8 with CLAHE pre-processing and heavy augmentation) are documented, runnable, and reproducible from the accompanying scripts; both are unexecuted in Phase 1. We have audited the methodological choices against the relevant object-detection [Lin 2017a, Ren 2017, Redmon 2016] and medical-imaging [Wang 2017, Shih 2019, Rajpurkar 2018, Zech 2018] literature, and connected the work to the cohort's adjacent industrial-defect [project 10] and COVID-classification [project 8] projects. Phase 2 will populate the placeholder numbers in the Results section by reading model-output JSON at build time, never by transcription.

## References

References are listed in `reports/references.md` and cited inline above by `[FirstAuthor Year]`. Each entry was verified live against CrossRef at scaffolding time. Volume, issue, and pages have been deliberately omitted to avoid fabrication risk per project rules. The 26 verified references span object-detection methodology, CNN backbones, chest-radiograph deep learning, dataset-bias literature, computer-aided diagnosis history, biomedical segmentation, and detection-evaluation tooling.
