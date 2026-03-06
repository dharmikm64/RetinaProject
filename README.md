# IDRiD Retinal Exudate Analysis

A complete end-to-end pipeline for analyzing diabetic retinopathy severity from the IDRiD (Indian Diabetic Retinopathy Image Dataset). Built to help a retinal specialist explore the relationship between hard macular exudates and DR grade across 516 fundus photographs.

---

## Project Summary

Diabetic retinopathy (DR) is a leading cause of blindness, and the presence and extent of hard exudates on the retina is a key indicator of disease severity and macular risk. This project ingests the IDRiD dataset, computes pixel-level exudate statistics from expert segmentation masks, trains an image classifier to predict DR grade, and presents everything through an interactive web dashboard.

The pipeline is broken into five modules:

| File | Role |
|---|---|
| data_pipline.py | Data ingestion -- extracts the segmentation ZIP, merges grading labels with mask statistics, and exports processed_dataset.csv (516 rows) |
| analysis.py | Visualization -- generates 4 diagnostic charts saved as PNGs |
| classifier.py | ML model -- fine-tunes a pretrained EfficientNet-B0 on the grading images for 5-class DR grade prediction |
| progression.py | Progression analysis -- tracks hard exudate coverage across multiple patient visits, detects foveal threat clusters, and returns figures + a summary report |
| server.py + dashboard.html | Web dashboard -- Flask API backend with a three-page HTML/CSS/JS frontend |

---

## Dataset

IDRiD contains 516 retinal fundus images (413 train / 103 test) graded across 5 DR severity levels:

| Grade | Label | Description |
|---|---|---|
| 0 | No DR | No signs of diabetic retinopathy |
| 1 | Mild DR | Microaneurysms only |
| 2 | Moderate DR | More than microaneurysms but less than severe |
| 3 | Severe DR | Extensive haemorrhages and vessel abnormalities |
| 4 | Proliferative DR | New vessel growth; most severe |

Expert pixel-level segmentation masks are provided for 81 images covering 5 lesion types: microaneurysms, haemorrhages, hard exudates, soft exudates, and optic disc.

---

## Charts Generated

Running python analysis.py produces four PNG charts:

- grade_distribution.png -- bar chart of image counts per DR grade across all 516 images
- exudate_vs_grade.png -- mean hard exudate pixel coverage (%) per grade with std error bars
- sample_overlays.png -- one representative fundus image per grade with yellow hard exudate overlay
- exudate_presence_rate.png -- percentage of images per grade that contain any hard exudates

---

## Classifier

classifier.py fine-tunes EfficientNet-B0 (pretrained on ImageNet) for 5-class DR grade prediction:

- All backbone layers frozen; only the final classification head is trained
- 10 epochs, Adam optimizer, cross-entropy loss, CPU training
- ~66% train accuracy, ~42% test accuracy -- expected given the small dataset and class imbalance
- Saved weights: classifier.pth
- Public API: predict(image_array) returns (grade: int, confidence: float)

---

## Dashboard

    python server.py

Opens at http://localhost:5000. Three pages:

Dataset Overview
- Summary metrics (total images, train/test split, segmentation mask count)
- All 4 analysis charts
- Grade breakdown table

Analyze an Image
- Drag-and-drop or click-to-upload a retinal fundus image
- Returns predicted DR grade, confidence score, clinical context, and a probability bar chart across all 5 grades

Patient Progression
- Upload 2 or more dated retinal fundus images from the same patient
- Optional: upload expert segmentation masks per visit; if omitted, exudates are auto-detected using green-channel CLAHE thresholding
- Optional: specify fovea coordinates to enable macular threat analysis
- Outputs: coverage trend chart, visit-to-visit diff overlays (new/resolved/persistent exudates), per-visit fovea proximity maps, and a plain-English progression summary

---

## Setup

    pip install numpy pandas matplotlib pillow tifffile torch torchvision flask flask-cors opencv-python

Then run in order:

    python data_pipline.py   # build processed_dataset.csv
    python analysis.py       # generate the 4 charts
    python classifier.py     # train the model (saves classifier.pth)
    python server.py         # launch the dashboard at http://localhost:5000

Progression analysis runs automatically via the dashboard. To test it standalone:

    python progression.py    # runs a synthetic 3-visit demo and saves figures

---

## Project Structure

    RetinaProject/
    |-- data_pipline.py          # data ingestion pipeline
    |-- analysis.py              # chart generation
    |-- classifier.py            # EfficientNet-B0 classifier
    |-- progression.py           # multi-visit exudate progression analysis
    |-- server.py                # Flask API server
    |-- dashboard.html           # three-page web dashboard
    |-- processed_dataset.csv    # master dataset (516 rows)
    |-- classifier.pth           # trained model weights
    |-- grade_distribution.png
    |-- exudate_vs_grade.png
    |-- sample_overlays.png
    |-- exudate_presence_rate.png
    |-- A. Segmentation/         # IDRiD segmentation masks
    |-- B. Disease Grading/      # IDRiD grading images and labels
