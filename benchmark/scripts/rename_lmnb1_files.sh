#!/usr/bin/env bash
# Strip the 64-char hex hash prefix from downloaded LMNB1 filenames.
#
# Files were downloaded from S3 with names like:
#   0a94f13e..._3500000922_100X_20170522_1-Scene-07-P7-F05.ome.tiff
# This renames them to:
#   3500000922_100X_20170522_1-Scene-07-P7-F05.ome.tiff
#
# After renaming, it swaps in the pre-generated CSVs that reference the short names.
#
# Usage:
#   bash scripts/rename_lmnb1_files.sh           # dry run (default)
#   bash scripts/rename_lmnb1_files.sh --execute  # actually rename

set -euo pipefail

DATA_DIR="/allen/aics/rep_learn/MegaSeg/data/variance-100x/LMNB1"
DRY_RUN=true

if [[ "${1:-}" == "--execute" ]]; then
    DRY_RUN=false
fi

rename_files() {
    local dir="$1"
    local count=0
    local skipped=0

    for f in "$dir"/*; do
        basename=$(basename "$f")
        # Match 64-char hex prefix followed by underscore
        if [[ "$basename" =~ ^[a-f0-9]{64}_ ]]; then
            short_name="${basename:65}"  # skip 64 hex chars + 1 underscore
            target="$dir/$short_name"

            if [[ -e "$target" ]]; then
                echo "SKIP (target exists): $basename -> $short_name"
                skipped=$((skipped + 1))
                continue
            fi

            if $DRY_RUN; then
                echo "WOULD RENAME: $basename -> $short_name"
            else
                mv "$f" "$target"
            fi
            count=$((count + 1))
        fi
    done

    echo "  $count files to rename, $skipped skipped in $(basename "$dir")/"
}

echo "=== Renaming raw files ==="
rename_files "$DATA_DIR/raw"

echo ""
echo "=== Renaming seg files ==="
rename_files "$DATA_DIR/seg"

if ! $DRY_RUN; then
    echo ""
    echo "=== Swapping CSVs ==="
    # Back up old CSVs
    for split in train val test; do
        old="$DATA_DIR/local/${split}_lmnb1_local_cytodl.csv"
        new="$DATA_DIR/local_renamed/${split}_lmnb1_local_cytodl.csv"
        if [[ -f "$new" ]]; then
            cp "$old" "$old.bak"
            cp "$new" "$old"
            echo "Replaced $old (backup at $old.bak)"
        else
            echo "WARNING: $new not found, skipping"
        fi
    done
    echo ""
    echo "Done. Files renamed and CSVs updated."
else
    echo ""
    echo "Dry run complete. Use --execute to apply changes."
fi
