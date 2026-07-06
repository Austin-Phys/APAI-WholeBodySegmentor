"""
Extract volume and fat-fraction metrics from TotalSegmentator label maps.

ASHA / MuscleMap workflow:
    Full label map   -> volume metrics
    Eroded label map -> FF metrics

Inputs:
    --ff_map          Dixon_FF_map.nii.gz
    --label_map_full  TotalSegmentator_nonMuscleMap_labels.nii.gz
    --label_map_ff    TotalSegmentator_nonMuscleMap_labels_eroded1.nii.gz
    --label_csv       TotalSegmentator_nonMuscleMap_labels.csv

Output:
    --out_csv         TotalSegmentator_FF_volume_metrics_eroded1.csv

CSV behavior:
- Numeric values are written as clean numeric-looking text without leading apostrophes.
- NaN/invalid values are written as blank cells, not "nan".
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def read_label_csv(label_csv: Path):
    labels = {}
    with open(label_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "label_id" not in reader.fieldnames or "structure" not in reader.fieldnames:
            raise ValueError(
                f"Label CSV must contain 'label_id' and 'structure'. Found: {reader.fieldnames}"
            )
        for row in reader:
            try:
                label_id = int(row["label_id"])
            except Exception:
                continue
            labels[label_id] = row["structure"]
    return labels


def robust_ff_scale(ff_data: np.ndarray):
    finite = ff_data[np.isfinite(ff_data)]
    if finite.size == 0:
        return ff_data.astype(np.float32), "unknown_empty"

    p99 = float(np.nanpercentile(finite, 99))
    max_val = float(np.nanmax(finite))

    if p99 > 1.5 or max_val > 2.0:
        return (ff_data / 100.0).astype(np.float32), "input_appears_percent_0_100_converted_to_fraction"
    return ff_data.astype(np.float32), "input_appears_fraction_0_1"


def clean_cell(value, decimals=8):
    """
    Return values for CSV without Excel-breaking artifacts.
    - NaN/inf -> blank
    - ints stay ints
    - floats use fixed decimal formatting with trailing zeros removed
    - no leading apostrophe is ever added
    """
    if value is None:
        return ""
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not math.isfinite(value):
            return ""
        s = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
        return s
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract TotalSegmentator volume metrics from full map and FF metrics from eroded map."
    )
    parser.add_argument("--ff_map", default="Dixon_FF_map.nii.gz")
    parser.add_argument("--label_map_full", default="TotalSegmentator_nonMuscleMap_labels.nii.gz")
    parser.add_argument("--label_map_ff", default="")
    parser.add_argument("--label_csv", default="TotalSegmentator_nonMuscleMap_labels.csv")
    parser.add_argument("--out_csv", default="TotalSegmentator_FF_volume_metrics.csv")
    parser.add_argument("--subject", default="")
    parser.add_argument("--session", default="")
    parser.add_argument("--station", default="")
    parser.add_argument("--min_valid_ff", type=float, default=0.0)
    parser.add_argument("--max_valid_ff", type=float, default=1.0)
    parser.add_argument("--clip_ff", action="store_true")
    args = parser.parse_args()

    ff_path = Path(args.ff_map)
    full_path = Path(args.label_map_full)
    ff_label_path = Path(args.label_map_ff) if args.label_map_ff else full_path
    label_csv_path = Path(args.label_csv)

    for p in [ff_path, full_path, ff_label_path, label_csv_path]:
        if not p.exists():
            print(f"ERROR: required input not found: {p}", file=sys.stderr)
            return 1

    ff_img = nib.load(str(ff_path))
    full_img = nib.load(str(full_path))
    ff_label_img = nib.load(str(ff_label_path))

    if ff_img.shape != full_img.shape or ff_img.shape != ff_label_img.shape:
        print("ERROR: Input shape mismatch:", file=sys.stderr)
        print(f"  FF map:         {ff_img.shape}", file=sys.stderr)
        print(f"  Full label map: {full_img.shape}", file=sys.stderr)
        print(f"  FF label map:   {ff_label_img.shape}", file=sys.stderr)
        return 1

    if not np.allclose(ff_img.affine, full_img.affine, atol=1e-4):
        print("WARNING: FF map and full label map affines differ. Check alignment in Slicer.", file=sys.stderr)
    if not np.allclose(ff_img.affine, ff_label_img.affine, atol=1e-4):
        print("WARNING: FF map and FF label map affines differ. Check alignment in Slicer.", file=sys.stderr)

    ff_raw = ff_img.get_fdata(dtype=np.float32)
    ff_frac, scale_note = robust_ff_scale(ff_raw)

    if args.clip_ff:
        ff_for_stats = np.clip(ff_frac, args.min_valid_ff, args.max_valid_ff)
    else:
        ff_for_stats = ff_frac

    full_data = full_img.get_fdata().astype(np.int32)
    ff_label_data = ff_label_img.get_fdata().astype(np.int32)
    labels = read_label_csv(label_csv_path)

    zooms = full_img.header.get_zooms()[:3]
    voxel_volume_mm3 = float(zooms[0] * zooms[1] * zooms[2])
    voxel_volume_ml = voxel_volume_mm3 / 1000.0

    rows = []
    for label_id in sorted(labels.keys()):
        structure = labels[label_id]

        full_mask = full_data == label_id
        ff_mask = ff_label_data == label_id

        voxel_count_full = int(np.count_nonzero(full_mask))
        voxel_count_ff_mask = int(np.count_nonzero(ff_mask))

        if voxel_count_full == 0:
            continue

        volume_ml = voxel_count_full * voxel_volume_ml

        values = ff_for_stats[ff_mask]
        finite_mask = np.isfinite(values)
        if not args.clip_ff:
            finite_mask &= (values >= args.min_valid_ff) & (values <= args.max_valid_ff)
        values_valid = values[finite_mask]
        valid_ff_voxels = int(values_valid.size)

        if valid_ff_voxels > 0:
            mean_ff = float(np.mean(values_valid))
            median_ff = float(np.median(values_valid))
            sd_ff = float(np.std(values_valid, ddof=1)) if valid_ff_voxels > 1 else 0.0
            min_ff = float(np.min(values_valid))
            max_ff = float(np.max(values_valid))
            p05_ff = float(np.percentile(values_valid, 5))
            p95_ff = float(np.percentile(values_valid, 95))
            fat_volume_ml = volume_ml * mean_ff
            ff_mask_available = True
        else:
            mean_ff = median_ff = sd_ff = min_ff = max_ff = p05_ff = p95_ff = math.nan
            fat_volume_ml = math.nan
            ff_mask_available = False

        rows.append({
            "subject": args.subject,
            "session": args.session,
            "station": args.station,
            "label_id": label_id,
            "structure": structure,
            "voxel_count_full_label": voxel_count_full,
            "voxel_count_ff_label": voxel_count_ff_mask,
            "voxel_count_valid_ff": valid_ff_voxels,
            "voxel_volume_mm3": voxel_volume_mm3,
            "volume_ml_full_mask": volume_ml,
            "mean_ff_fraction_eroded_mask": mean_ff,
            "median_ff_fraction_eroded_mask": median_ff,
            "sd_ff_fraction_eroded_mask": sd_ff,
            "min_ff_fraction_eroded_mask": min_ff,
            "max_ff_fraction_eroded_mask": max_ff,
            "p05_ff_fraction_eroded_mask": p05_ff,
            "p95_ff_fraction_eroded_mask": p95_ff,
            "mean_ff_percent_eroded_mask": mean_ff * 100 if math.isfinite(mean_ff) else math.nan,
            "median_ff_percent_eroded_mask": median_ff * 100 if math.isfinite(median_ff) else math.nan,
            "fat_volume_ml_full_volume_x_eroded_mean_ff": fat_volume_ml,
            "ff_mask_source": str(ff_label_path),
            "ff_mask_available": ff_mask_available,
            "ff_scale_note": scale_note,
        })

    out_path = Path(args.out_csv)
    fieldnames = [
        "subject", "session", "station", "label_id", "structure",
        "voxel_count_full_label", "voxel_count_ff_label", "voxel_count_valid_ff",
        "voxel_volume_mm3", "volume_ml_full_mask",
        "mean_ff_fraction_eroded_mask", "median_ff_fraction_eroded_mask", "sd_ff_fraction_eroded_mask",
        "min_ff_fraction_eroded_mask", "max_ff_fraction_eroded_mask",
        "p05_ff_fraction_eroded_mask", "p95_ff_fraction_eroded_mask",
        "mean_ff_percent_eroded_mask", "median_ff_percent_eroded_mask",
        "fat_volume_ml_full_volume_x_eroded_mean_ff",
        "ff_mask_source", "ff_mask_available", "ff_scale_note",
    ]

    numeric_float_cols = {
        "voxel_volume_mm3", "volume_ml_full_mask",
        "mean_ff_fraction_eroded_mask", "median_ff_fraction_eroded_mask", "sd_ff_fraction_eroded_mask",
        "min_ff_fraction_eroded_mask", "max_ff_fraction_eroded_mask",
        "p05_ff_fraction_eroded_mask", "p95_ff_fraction_eroded_mask",
        "mean_ff_percent_eroded_mask", "median_ff_percent_eroded_mask",
        "fat_volume_ml_full_volume_x_eroded_mean_ff",
    }

    int_cols = {"label_id", "voxel_count_full_label", "voxel_count_ff_label", "voxel_count_valid_ff"}

    cleaned_rows = []
    for row in rows:
        cleaned = {}
        for key in fieldnames:
            value = row.get(key, "")
            if key in int_cols:
                cleaned[key] = clean_cell(value, decimals=0)
            elif key in numeric_float_cols:
                cleaned[key] = clean_cell(value, decimals=8)
            else:
                cleaned[key] = value
        cleaned_rows.append(cleaned)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"Saved: {out_path}")
    print(f"Structures written: {len(rows)}")
    print(f"Voxel volume: {voxel_volume_mm3:.6f} mm^3 = {voxel_volume_ml:.6f} mL")
    print(f"Volume metrics source: {full_path}")
    print(f"FF metrics source: {ff_label_path}")
    print(f"FF scale: {scale_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
