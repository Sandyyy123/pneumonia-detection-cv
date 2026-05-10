![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Medical CV](https://img.shields.io/badge/Medical-Imaging-red) ![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)

# RSNA Pneumonia Detection — Medical CV Object Detection

Detects and localises pneumonia opacity regions in chest X-rays using Faster R-CNN and RetinaNet on the RSNA 2018 challenge dataset.

---

## Task

**Object Detection (Medical Imaging)**

---

## Architecture

```
DICOM X-ray → Preprocessing → Faster R-CNN (ResNet-50 FPN) → Opacity BBox → mAP Eval
```

---

## Key Features

- Bounding-box detection of pneumonia opacities in 26,684 chest X-rays
- Faster R-CNN with ResNet-50 FPN backbone (torchvision)
- RetinaNet alternative for comparison on small/dense opacities
- DICOM preprocessing pipeline (windowing, normalisation)
- mAP@IoU[0.4:0.75] evaluation matching RSNA competition metric

---

## Dataset

[RSNA Pneumonia Detection Challenge (Kaggle)](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/pneumonia-detection-cv.git
cd pneumonia-detection-cv
pip install -r requirements.txt

# See data/README.md for dataset download
python src/model_baseline.py
python src/model_advanced.py
```

---

## Tech Stack

`PyTorch · torchvision · pydicom · albumentations · OpenCV`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Mössingen, Germany.

---

## License

MIT
