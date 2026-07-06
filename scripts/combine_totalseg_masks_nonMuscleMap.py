"""
Combine TotalSegmentator binary mask files into one multi-label NIfTI label map.

Purpose for ASHA/MuscleMap workflow:
- Keep TotalSegmentator anatomy that does NOT overlap with MuscleMap labels.
- Exclude only TotalSegmentator structures that overlap/compete with MuscleMap.
- Skip empty or tiny masks using MIN_VOXELS.

Run from the folder containing the TotalSegmentator output directory, for example:
    python combine_totalseg_masks_nonMuscleMap.py

Defaults:
    input mask folder: TS_WATER
    output label map:  TotalSegmentator_nonMuscleMap_labels.nii.gz
    output CSV:        TotalSegmentator_nonMuscleMap_labels.csv
"""

import argparse
import csv
import fnmatch
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# Default settings
DEFAULT_MASK_DIR = "TS_WATER"
DEFAULT_OUT_NII = "TotalSegmentator_nonMuscleMap_labels.nii.gz"
DEFAULT_OUT_CSV = "TotalSegmentator_nonMuscleMap_labels.csv"
DEFAULT_MIN_VOXELS = 500

# Only blacklist structures that overlap or directly compete with MuscleMap labels.
# MuscleMap owns skeletal muscle and pelvis/femur anatomy in this workflow.
MUSCLEMAP_OVERLAP_BLACKLIST_PATTERNS = [
    "autochthon_*",      # overlaps paraspinal / erector / multifidus territory
    "gluteus_*",         # overlaps MuscleMap gluteus labels
    "iliopsoas_*",       # overlaps psoas/iliacus region
    "femur_*",           # overlaps MuscleMap femur labels
    "hip_*",             # pelvic/hip bone region; overlaps ilium/femur/sacrum territory
    "sacrum",            # overlaps MuscleMap sacrum label
]


def is_blacklisted(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine TotalSegmentator masks into one non-MuscleMap-overlap label map."
    )
    parser.add_argument("--mask_dir", default=DEFAULT_MASK_DIR,
                        help=f"Folder containing TotalSegmentator .nii.gz masks. Default: {DEFAULT_MASK_DIR}")
    parser.add_argument("--out_nii", default=DEFAULT_OUT_NII,
                        help=f"Output combined label-map NIfTI. Default: {DEFAULT_OUT_NII}")
    parser.add_argument("--out_csv", default=DEFAULT_OUT_CSV,
                        help=f"Output label mapping CSV. Default: {DEFAULT_OUT_CSV}")
    parser.add_argument("--min_voxels", type=int, default=DEFAULT_MIN_VOXELS,
                        help=f"Skip masks with fewer than this many nonzero voxels. Default: {DEFAULT_MIN_VOXELS}")
    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    if not mask_dir.exists() or not mask_dir.is_dir():
        print(f"ERROR: mask_dir not found or not a directory: {mask_dir}", file=sys.stderr)
        return 1

    files = sorted([f for f in mask_dir.iterdir() if f.name.endswith(".nii.gz")])
    if not files:
        print(f"ERROR: no .nii.gz masks found in: {mask_dir}", file=sys.stderr)
        return 1

    ref_img = nib.load(str(files[0]))
    ref_shape = ref_img.shape
    label_data = np.zeros(ref_shape, dtype=np.uint16)

    label_rows = []
    skipped_rows = []
    label_id = 1

    for path in files:
        name = path.name.replace(".nii.gz", "")

        img = nib.load(str(path))
        if img.shape != ref_shape:
            skipped_rows.append([name, "shape_mismatch", img.shape, 0])
            print(f"SKIP shape mismatch: {name} shape={img.shape}, expected={ref_shape}")
            continue

        data = img.get_fdata() > 0
        voxel_count = int(np.count_nonzero(data))

        if is_blacklisted(name, MUSCLEMAP_OVERLAP_BLACKLIST_PATTERNS):
            skipped_rows.append([name, "musclemap_overlap_blacklist", "", voxel_count])
            print(f"SKIP MuscleMap-overlap: {name} ({voxel_count} voxels)")
            continue

        if voxel_count < args.min_voxels:
            skipped_rows.append([name, f"voxel_count_below_{args.min_voxels}", "", voxel_count])
            print(f"SKIP small/empty: {name} ({voxel_count} voxels)")
            continue

        # If TotalSegmentator masks overlap, keep the first label and report overlap.
        # This avoids later masks silently overwriting earlier structures.
        overlap_voxels = int(np.count_nonzero((label_data > 0) & data))
        new_voxels = data & (label_data == 0)
        label_data[new_voxels] = label_id

        label_rows.append([label_id, name, voxel_count, int(np.count_nonzero(new_voxels)), overlap_voxels])
        print(f"KEEP {label_id}: {name} total={voxel_count} new={int(np.count_nonzero(new_voxels))} overlap={overlap_voxels}")
        label_id += 1

    out_img = nib.Nifti1Image(label_data, ref_img.affine, ref_img.header)
    out_img.set_data_dtype(np.uint16)
    nib.save(out_img, args.out_nii)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "structure", "voxel_count_total", "voxel_count_written", "overlap_voxels_not_written"])
        writer.writerows(label_rows)

    skipped_csv = Path(args.out_csv).with_name(Path(args.out_csv).stem + "_skipped.csv")
    with open(skipped_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["structure", "reason", "detail", "voxel_count"])
        writer.writerows(skipped_rows)

    print("\nSaved:")
    print(args.out_nii)
    print(args.out_csv)
    print(skipped_csv)
    print(f"\nKept labels: {len(label_rows)}")
    print(f"Skipped masks: {len(skipped_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
