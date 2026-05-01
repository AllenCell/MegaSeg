#!/bin/bash
#SBATCH --job-name=lamp1_download
#SBATCH --output=/home/ashwin.samudre/logs/%j.out
#SBATCH --error=/home/ashwin.samudre/logs/%j.err
#SBATCH --partition=aics
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set --
source /allen/aics/users/ashwin.samudre/miniconda3/etc/profile.d/conda.sh
conda activate emt-acm-env

LAMP1_DIR="/allen/aics/rep_learn/MegaSeg/data/variance-100x/LAMP1"
RAW_DIR="${LAMP1_DIR}/raw"
SEG_DIR="${LAMP1_DIR}/seg"
LOCAL_DIR="${LAMP1_DIR}/local"

mkdir -p "$RAW_DIR" "$SEG_DIR" "${LOCAL_DIR}/test"

python3 - << 'PYEOF'
import csv, boto3, os, re
from pathlib import Path
from botocore import UNSIGNED
from botocore.config import Config

LAMP1_DIR = Path("/allen/aics/rep_learn/MegaSeg/data/variance-100x/LAMP1")
RAW_DIR   = LAMP1_DIR / "raw"
SEG_DIR   = LAMP1_DIR / "seg"
LOCAL_DIR = LAMP1_DIR / "local"
S3_BUCKET = "allencell"
S3_PREFIX = "aics/hipsc_single_cell_image_dataset"

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")

def s3_key_to_filename(s3_path):
    """Extract filename from S3 URI."""
    return s3_path.split("/")[-1]

def download_if_missing(s3_uri, local_path):
    if local_path.exists():
        print(f"  [skip] {local_path.name}")
        return
    # Parse s3://bucket/key
    without_prefix = s3_uri.replace("s3://", "")
    bucket, key = without_prefix.split("/", 1)
    print(f"  [download] {local_path.name}")
    s3.download_file(bucket, key, str(local_path))

splits = {
    "train": LAMP1_DIR / "train_lamp1_s3_cytodl.csv",
    "val":   LAMP1_DIR / "val_lamp1_s3_cytodl.csv",
    "test":  LAMP1_DIR / "test_lamp1_s3_cytodl.csv",
}

local_rows = {"train": [], "val": [], "test": []}

for split, csv_path in splits.items():
    print(f"\nProcessing {split} split...")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_s3 = row["raw"]
            seg_s3 = row["seg"]
            raw_fname = s3_key_to_filename(raw_s3)
            seg_fname = s3_key_to_filename(seg_s3)
            raw_local = RAW_DIR / raw_fname
            seg_local = SEG_DIR / seg_fname
            download_if_missing(raw_s3, raw_local)
            download_if_missing(seg_s3, seg_local)
            local_rows[split].append({"raw": str(raw_local), "seg": str(seg_local)})

# Write local CSVs
for split, rows in local_rows.items():
    out_path = LOCAL_DIR / f"{split}_lamp1_local_cytodl.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["raw", "seg"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")

# Write test CSV into local/test/ subdirectory (for eval configs)
test_csv_in_test_dir = LOCAL_DIR / "test" / "test_lamp1_local_cytodl.csv"
with open(test_csv_in_test_dir, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["raw", "seg"])
    writer.writeheader()
    writer.writerows(local_rows["test"])
print(f"Wrote test CSV -> {test_csv_in_test_dir}")

print("\nDone.")
PYEOF
