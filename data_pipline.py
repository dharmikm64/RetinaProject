"""
IDRiD Retinal Image Data Pipeline

Loads, extracts, and processes the IDRiD dataset
for retinopathy grading and macular exudate analysis.

Outputs:
  A. Segmentation/   - extracted segmentation folder
  processed_dataset.csv - master table
"""

import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import tifffile


# -- Paths -------------------------------------------------------

PROJECT_DIR = Path(__file__).parent

SEG_ZIP = PROJECT_DIR / "A. Segmentation.zip"
SEG_DIR = PROJECT_DIR / "A. Segmentation"

GRADING_DIR        = PROJECT_DIR / "B. Disease Grading"
GRADING_TRAIN_IMGS = GRADING_DIR / "1. Original Images" / "a. Training Set"
GRADING_TEST_IMGS  = GRADING_DIR / "1. Original Images" / "b. Testing Set"
GRADING_TRAIN_CSV  = GRADING_DIR / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
GRADING_TEST_CSV   = GRADING_DIR / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"

SEG_TRAIN_IMGS = SEG_DIR / "1. Original Images" / "a. Training Set"
SEG_TEST_IMGS  = SEG_DIR / "1. Original Images" / "b. Testing Set"
SEG_MASKS_DIR  = SEG_DIR / "2. All Segmentation Groundtruths"

# -- Constants --------------------------------------------------

LESION_TYPES = {
    "microaneurysms": ("1. Microaneurysms", "_MA"),
    "haemorrhages":   ("2. Haemorrhages",   "_HE"),
    "hard_exudates":  ("3. Hard Exudates",  "_EX"),
    "soft_exudates":  ("4. Soft Exudates",  "_SE"),
    "optic_disc":     ("5. Optic Disc",     "_OD"),
}

RETINOPATHY_GRADE_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

MACULAR_EDEMA_LABELS = {
    0: "No risk",
    1: "Low risk",
    2: "High risk",
}


# -- Step 1: Extract segmentation zip ---------------------------

def extract_segmentation_zip(force=False):
    """Extract A. Segmentation.zip to the project directory."""
    if SEG_DIR.exists() and not force:
        print(f"[skip] Already extracted: {SEG_DIR.name}/")
        return SEG_DIR
    if not SEG_ZIP.exists():
        raise FileNotFoundError(f"Segmentation zip not found: {SEG_ZIP}")
    print(f"Extracting {SEG_ZIP.name}  (558 MB - this may take a minute) ...")
    with zipfile.ZipFile(SEG_ZIP, "r") as z:
        z.extractall(PROJECT_DIR)
    print(f"[done] Extracted to: {SEG_DIR.name}/")
    return SEG_DIR


# -- Step 2: Load disease grading labels ------------------------

def load_grading_labels():
    """Load training + testing CSVs into a single DataFrame."""
    def _load(csv_path, split):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        df = df[["Image name", "Retinopathy grade", "Risk of macular edema"]].copy()
        df.columns = ["image_id", "retinopathy_grade", "macular_edema_risk"]
        df["split"] = split
        return df

    train = _load(GRADING_TRAIN_CSV, "train")
    test  = _load(GRADING_TEST_CSV,  "test")
    df = pd.concat([train, test], ignore_index=True)

    df["retinopathy_label"]   = df["retinopathy_grade"].map(RETINOPATHY_GRADE_LABELS)
    df["macular_edema_label"] = df["macular_edema_risk"].map(MACULAR_EDEMA_LABELS)

    def _img_path(row):
        folder = GRADING_TRAIN_IMGS if row["split"] == "train" else GRADING_TEST_IMGS
        img_id = row["image_id"]
        return folder / f"{img_id}.jpg"

    df["image_path"]   = df.apply(_img_path, axis=1)
    df["image_exists"] = df["image_path"].apply(lambda p: p.exists())

    print(f"[loaded] Grading labels -- {len(train)} train, {len(test)} test")
    return df


# -- Step 3: Compute lesion statistics from masks ---------------

def _seg_id_to_grading_id(seg_id):
    """Convert segmentation ID: IDRiD_01 -> IDRiD_001"""
    prefix, num = seg_id.rsplit("_", 1)
    return f"{prefix}_{int(num):03d}"


def _retinal_pixel_count(img_path):
    """Count non-background pixels for area % denominator."""
    if not img_path.exists():
        return 1
    img = np.array(Image.open(img_path).convert("L"))
    return max(int(np.sum(img > 10)), 1)


def compute_mask_stats(seg_image_id, split):
    """
    Load all lesion masks for one segmentation-set image and compute:
      <lesion>_pixels    : positive mask pixel count
      <lesion>_area_pct  : lesion area as % of total retinal area
      has_<lesion>       : bool flag
    """
    img_folder  = SEG_TRAIN_IMGS if split == "train" else SEG_TEST_IMGS
    mask_subdir = "a. Training Set" if split == "train" else "b. Testing Set"

    img_path     = img_folder / f"{seg_image_id}.jpg"
    total_pixels = _retinal_pixel_count(img_path)

    row = {
        "seg_image_id":         seg_image_id,
        "grading_image_id":     _seg_id_to_grading_id(seg_image_id),
        "total_retinal_pixels": total_pixels,
    }

    for lesion_key, (folder_name, suffix) in LESION_TYPES.items():
        mask_path = (
            SEG_MASKS_DIR / mask_subdir / folder_name
            / f"{seg_image_id}{suffix}.tif"
        )
        if mask_path.exists():
            mask = tifffile.imread(str(mask_path))
            pixel_count = int(np.sum(mask > 0))
        else:
            pixel_count = 0

        area_pct = round(pixel_count / total_pixels * 100, 5)
        row[f"{lesion_key}_pixels"]   = pixel_count
        row[f"{lesion_key}_area_pct"] = area_pct
        row[f"has_{lesion_key}"]      = pixel_count > 0

    return row


def load_segmentation_stats():
    """Compute mask statistics for all images in the segmentation folder."""
    rows = []
    for split, img_dir in [("train", SEG_TRAIN_IMGS), ("test", SEG_TEST_IMGS)]:
        if not img_dir.exists():
            continue
        for img_file in sorted(img_dir.glob("*.jpg")):
            rows.append(compute_mask_stats(img_file.stem, split))
    df = pd.DataFrame(rows)
    print(f"[loaded] Segmentation stats -- {len(df)} images processed")
    return df


# -- Step 4: Build master dataset -------------------------------

def build_master_dataset():
    """
    Join grading labels (516 images) with segmentation mask stats (81 images).
    Images without masks have NaN for lesion columns and has_masks=False.
    """
    grading_df = load_grading_labels()
    seg_df     = load_segmentation_stats()

    master = grading_df.merge(
        seg_df.rename(columns={"grading_image_id": "image_id"}),
        on="image_id",
        how="left",
    )

    master["has_masks"] = master["seg_image_id"].notna()

    area_cols = [c for c in master.columns if c.endswith("_area_pct")]
    master["total_lesion_area_pct"] = master[area_cols].fillna(0).sum(axis=1)

    n_masked = int(master["has_masks"].sum())
    print(f"[done]   Master dataset -- {len(master)} total images, {n_masked} with segmentation masks")
    return master


# -- Step 5: Save processed CSV ----------------------------------

def save_processed_data(df, output_path=None):
    """Write master DataFrame to CSV."""
    if output_path is None:
        output_path = PROJECT_DIR / "processed_dataset.csv"
    export = df.drop(columns=["image_path"], errors="ignore")
    export.to_csv(output_path, index=False)
    print(f"[saved]  {output_path.name} -- {len(export)} rows, {len(export.columns)} columns")
    return output_path


# -- Summary report ----------------------------------------------

def print_summary(df):
    train  = df[df["split"] == "train"]
    masked = df[df["has_masks"]]
    sep = "-" * 52

    print("\n" + sep)
    print("  DATASET SUMMARY")
    print(sep)
    print(f"  Total images       : {len(df)}")
    print(f"  Training           : {len(train)}")
    print(f"  Testing            : {len(df) - len(train)}")
    print(f"  With seg masks     : {len(masked)}")

    print("\nRetinopathy grade distribution (training):")
    for label, count in train["retinopathy_label"].value_counts().sort_index().items():
        pct = count / len(train) * 100
        print(f"    {label:<22} {count:>4}  ({pct:.1f}%)")

    print("\nMacular edema risk (training):")
    for label, count in train["macular_edema_label"].value_counts().sort_index().items():
        pct = count / len(train) * 100
        print(f"    {label:<22} {count:>4}  ({pct:.1f}%)")

    if len(masked):
        print(f"\n  Lesion burden (segmentation subset, n={len(masked)}):")
        lesion_cols = [
            ("hard_exudates_area_pct",  "Hard exudates"),
            ("soft_exudates_area_pct",  "Soft exudates"),
            ("haemorrhages_area_pct",   "Haemorrhages"),
            ("microaneurysms_area_pct", "Microaneurysms"),
            ("optic_disc_area_pct",     "Optic disc"),
        ]
        for col, label in lesion_cols:
            if col in masked.columns:
                m = masked[col]
                print(f"    {label:<22} mean={m.mean():.4f}%  max={m.max():.4f}%")

    print("\n" + sep + "\n")


# -- Entry point -------------------------------------------------

def run_pipeline():
    print("=" * 52)
    print("  IDRiD Retinal Data Pipeline")
    print("=" * 52)
    extract_segmentation_zip()
    master = build_master_dataset()
    save_processed_data(master)
    print_summary(master)
    print("[complete] Pipeline finished.")
    return master


if __name__ == "__main__":
    run_pipeline()
