# Dataset - RSNA Pneumonia Detection Challenge

## Source

Kaggle competition: [rsna-pneumonia-detection-challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)

Hosted by the Radiological Society of North America (RSNA) in partnership with Kaggle, the US National Institutes of Health (NIH), and the Society of Thoracic Radiology, 2018.

## Why not downloaded in v1.0

- Total size: ~3.7 GB compressed (DICOM imagery is large per file).
- Access: Kaggle competition rules require the user to log in, accept the competition rules, and join the competition once before the CLI download will succeed.
- Per Project layout

## Download instructions (run in main session, not in implementation)

1. Accept the rules at <https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules>.
2. Confirm Kaggle CLI auth is in place: `~/.kaggle/kaggle.json` (chmod 600).
3. From this folder, run:

```bash
cd data/
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip -q rsna-pneumonia-detection-challenge.zip
unzip -q stage_2_train_images.zip -d stage_2_train_images
unzip -q stage_2_test_images.zip -d stage_2_test_images
rm rsna-pneumonia-detection-challenge.zip stage_2_train_images.zip stage_2_test_images.zip
```

## Expected layout after download

```
data/
├── README.md                              (this file)
├── stage_2_train_labels.csv               (bounding-box labels, multi-row per patient)
├── stage_2_detailed_class_info.csv        (3-class metadata)
├── stage_2_sample_submission.csv
├── stage_2_train_images/                  (~26,684 DICOM files, ~3.0 GB)
│   └── *.dcm
└── stage_2_test_images/                   (~3,000 DICOM files, ~0.4 GB)
    └── *.dcm
```

## Schema

### `stage_2_train_labels.csv`

| Column     | Type    | Notes                                                              |
| ---------- | ------- | ------------------------------------------------------------------ |
| patientId  | string  | DICOM SOP Instance UID, used as image filename stem                |
| x          | float   | Bounding-box top-left x in pixel coords; NaN if Target = 0         |
| y          | float   | Bounding-box top-left y in pixel coords; NaN if Target = 0         |
| width      | float   | Box width in pixels; NaN if Target = 0                             |
| height     | float   | Box height in pixels; NaN if Target = 0                            |
| Target     | int     | 1 = opacity present, 0 = no opacity                                |

Multiple rows per patient if more than one opacity bounding box. ~22-30 percent of patient IDs are positive.

### `stage_2_detailed_class_info.csv`

| Column     | Type    | Values                                                            |
| ---------- | ------- | ----------------------------------------------------------------- |
| patientId  | string  | as above                                                          |
| class      | string  | `Normal`, `No Lung Opacity / Not Normal`, `Lung Opacity`          |

The `No Lung Opacity / Not Normal` class is critical: these are abnormal X-rays with non-pneumonia findings (e.g. atelectasis, cardiomegaly, effusions). Models trained as binary opacity-vs-normal will see distribution shift if deployed clinically. Discussed at length in the manuscript.

## DICOM specifics

- Single-channel grayscale, photometric interpretation typically `MONOCHROME2`.
- Pixel size: usually 1024 x 1024 with `BitsStored = 16` and `BitsAllocated = 16`, but actual stored values often fit in 8 bits.
- Read with `pydicom.dcmread(path).pixel_array`. For training, rescale to `[0, 255]` uint8 or normalise to `[0, 1]` float32.
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation, OpenCV `cv2.createCLAHE`) before model input - standard practice in chest-X-ray pipelines (CheXNet pre-processing convention).

## Label and dataset caveats

- Ground truth was produced by a panel of board-certified radiologists; bounding-box-level inter-rater agreement is moderate, not high. Discussed in the manuscript and in `references.md` (e.g. Rajpurkar 2017 CheXNet).
- The dataset is a re-annotated subset of the NIH ChestX-ray8 / 14 release (Wang 2017). Inheriting any label noise or selection bias from that source is a documented limitation.
- Patient demographics (age, sex) are in the DICOM headers; the CSV labels do not include them.
- Site / scanner provenance is not disclosed at patient granularity in the public release, limiting external-validation analyses.
