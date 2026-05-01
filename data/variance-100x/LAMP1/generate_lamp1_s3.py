"""
Generate train/val/test S3 CSV files for LAMP1 fine-tuning experiments.

Replicates the same approach used for LMNB1 (variance-100x/LMNB1/).
- Source: s3://allencell/aics/hipsc_single_cell_image_dataset/metadata.csv
- Filter: edge_flag == 0, outlier == 'No'
- Deduplicate at FOV level (one row per unique fov_path)
- Split: 160 train / 40 val / 67 test  (same as LMNB1)
- Output format: raw,seg  with full S3 URIs

Usage:
    python generate_lamp1_s3.py
"""

import boto3, io, csv, random
from botocore import UNSIGNED
from botocore.config import Config

S3_BUCKET = "allencell"
S3_PREFIX = "aics/hipsc_single_cell_image_dataset"
S3_BASE = f"s3://{S3_BUCKET}/{S3_PREFIX}"

N_TRAIN = 160
N_VAL = 40
N_TEST = 67
SEED = 42

print("Streaming metadata.csv from S3 ...")
s3 = boto3.client(
    "s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1"
)
resp = s3.get_object(Bucket=S3_BUCKET, Key=f"{S3_PREFIX}/metadata.csv")
body = resp["Body"]

lamp1_rows_raw = []
header = None
buf = ""

for chunk in body.iter_chunks(chunk_size=65536):
    buf += chunk.decode("utf-8")
    lines = buf.split("\n")
    buf = lines[-1]
    for line in lines[:-1]:
        if header is None:
            header = line
            continue
        if "LAMP1" in line:
            lamp1_rows_raw.append(line)

if buf.strip() and "LAMP1" in buf:
    lamp1_rows_raw.append(buf)

cols = list(csv.reader([header]))[0]
rows = [dict(zip(cols, r)) for r in csv.reader(lamp1_rows_raw)]
print(f"  Total LAMP1 cell rows: {len(rows)}")

# Filter quality
filtered = [r for r in rows if r["edge_flag"] == "0" and r["outlier"] == "No"]
print(f"  After edge_flag=0 + outlier=No: {len(filtered)} rows")

# Deduplicate at FOV level (keep first occurrence)
fov_dict = {}
for r in filtered:
    fp = r["fov_path"]
    if fp not in fov_dict:
        fov_dict[fp] = r

fov_list = list(fov_dict.values())
print(f"  Unique FOVs: {len(fov_list)}")

assert len(fov_list) >= N_TRAIN + N_VAL + N_TEST, (
    f"Not enough FOVs: need {N_TRAIN + N_VAL + N_TEST}, got {len(fov_list)}"
)

# Shuffle deterministically
rng = random.Random(SEED)
rng.shuffle(fov_list)

train_rows = fov_list[:N_TRAIN]
val_rows = fov_list[N_TRAIN : N_TRAIN + N_VAL]
test_rows = fov_list[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]

print(
    f"  Split -> train: {len(train_rows)}, val: {len(val_rows)}, test: {len(test_rows)}"
)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["raw", "seg"])
        for r in rows:
            raw = f"{S3_BASE}/{r['fov_path']}"
            seg = f"{S3_BASE}/{r['struct_seg_path']}"
            writer.writerow([raw, seg])
    print(f"  Wrote {len(rows)} rows -> {path}")


write_csv("train_lamp1_s3_cytodl.csv", train_rows)
write_csv("val_lamp1_s3_cytodl.csv", val_rows)
write_csv("test_lamp1_s3_cytodl.csv", test_rows)

print("Done.")
