"""
Erode a TotalSegmentator multi-label NIfTI label map label-by-label.

Purpose for ASHA/MuscleMap workflow:
- Preserve the original TotalSegmentator non-MuscleMap label map for volume metrics.
- Create a permanently saved eroded label map for FF metrics.
- Erode each label independently so labels do not merge or overwrite each other.

Typical workflow:
    TotalSegmentator_nonMuscleMap_labels.nii.gz
        -> TotalSegmentator_nonMuscleMap_labels_eroded.nii.gz

Example:
    python erode_totalseg_labels.py ^
      --label_map TotalSegmentator_nonMuscleMap_labels_FULL.nii.gz ^
      --label_csv TotalSegmentator_nonMuscleMap_labels_FULL.csv ^
      --out_nii TotalSegmentator_nonMuscleMap_labels_FULL_eroded.nii.gz ^
      --out_csv TotalSegmentator_nonMuscleMap_labels_FULL_eroded_stats.csv ^
      --erode_voxels 1

Notes:
- This script does NOT change the original label map.
- Volume metrics should use the full/original label map.
- FF metrics should use this eroded label map.
"""

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure


def read_label_csv(label_csv: Path):
    """Return dict: label_id -> structure."""
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


def erode_mask(mask: np.ndarray, erode_voxels: int, connectivity: int) -> np.ndarray:
    """Binary erode a 3D mask."""
    if erode_voxels <= 0:
        return mask.copy()

    structure = generate_binary_structure(rank=3, connectivity=connectivity)
    return binary_erosion(mask, structure=structure, iterations=erode_voxels)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a permanently saved eroded TotalSegmentator label map."
    )
    parser.add_argument("--label_map", default="TotalSegmentator_nonMuscleMap_labels.nii.gz")
    parser.add_argument("--label_csv", default="TotalSegmentator_nonMuscleMap_labels.csv")
    parser.add_argument("--out_nii", default="TotalSegmentator_nonMuscleMap_labels_eroded.nii.gz")
    parser.add_argument("--out_csv", default="TotalSegmentator_nonMuscleMap_labels_eroded_stats.csv")
    parser.add_argument("--erode_voxels", type=int, default=1)
    parser.add_argument("--connectivity", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument(
        "--min_eroded_voxels",
        type=int,
        default=25,
        help="If erosion leaves fewer voxels than this, the label is not written."
    )
    parser.add_argument(
        "--keep_tiny_original",
        action="store_true",
        help="If erosion leaves too few voxels, write the original full mask instead of skipping."
    )

    args = parser.parse_args()

    label_path = Path(args.label_map)
    label_csv_path = Path(args.label_csv)

    for p in [label_path, label_csv_path]:
        if not p.exists():
            print(f"ERROR: required input not found: {p}", file=sys.stderr)
            return 1

    img = nib.load(str(label_path))
    label_data = img.get_fdata().astype(np.int32)
    labels = read_label_csv(label_csv_path)

    eroded_data = np.zeros(label_data.shape, dtype=np.uint16)
    rows = []

    for label_id in sorted(labels.keys()):
        structure = labels[label_id]
        full_mask = label_data == label_id
        full_voxels = int(np.count_nonzero(full_mask))

        if full_voxels == 0:
            row = {
                "label_id": label_id,
                "structure": structure,
                "full_voxels": 0,
                "eroded_voxels": 0,
                "written_voxels": 0,
                "overlap_voxels_not_written": 0,
                "percent_retained_after_erosion": "",
                "status": "skip_empty_full_mask",
                "erode_voxels": args.erode_voxels,
                "connectivity": args.connectivity,
            }
            rows.append(row)
            print(f"SKIP empty: {label_id} {structure}")
            continue

        eroded_mask = erode_mask(full_mask, args.erode_voxels, args.connectivity)
        eroded_voxels = int(np.count_nonzero(eroded_mask))

        if eroded_voxels >= args.min_eroded_voxels:
            write_mask = eroded_mask
            status = "eroded_written"
        elif args.keep_tiny_original:
            write_mask = full_mask
            status = "full_mask_written_because_eroded_too_small"
        else:
            write_mask = np.zeros_like(full_mask, dtype=bool)
            status = "skip_eroded_too_small"

        overlap_voxels = int(np.count_nonzero((eroded_data > 0) & write_mask))
        write_mask_no_overlap = write_mask & (eroded_data == 0)
        written_voxels = int(np.count_nonzero(write_mask_no_overlap))
        eroded_data[write_mask_no_overlap] = label_id

        rows.append({
            "label_id": label_id,
            "structure": structure,
            "full_voxels": full_voxels,
            "eroded_voxels": eroded_voxels,
            "written_voxels": written_voxels,
            "overlap_voxels_not_written": overlap_voxels,
            "percent_retained_after_erosion": (
                100.0 * eroded_voxels / full_voxels if full_voxels > 0 else ""
            ),
            "status": status,
            "erode_voxels": args.erode_voxels,
            "connectivity": args.connectivity,
        })

        print(
            f"{status}: {label_id} {structure} "
            f"full={full_voxels} eroded={eroded_voxels} written={written_voxels}"
        )

    out_img = nib.Nifti1Image(eroded_data, img.affine, img.header)
    out_img.set_data_dtype(np.uint16)
    nib.save(out_img, args.out_nii)

    fieldnames = [
        "label_id", "structure", "full_voxels", "eroded_voxels", "written_voxels",
        "overlap_voxels_not_written", "percent_retained_after_erosion",
        "status", "erode_voxels", "connectivity"
    ]

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:")
    print(args.out_nii)
    print(args.out_csv)
    print(f"\nLabels processed: {len(rows)}")
    print(f"Erosion: {args.erode_voxels} voxel(s), connectivity={args.connectivity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
