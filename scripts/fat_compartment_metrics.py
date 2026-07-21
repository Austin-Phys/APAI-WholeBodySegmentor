#!/usr/bin/env python
"""
fat_compartment_metrics.py

Metrics-only exporter for WholeBodySeg fat compartment QC masks.

This module never creates or modifies segmentation masks. It reads the current
NIfTI files from ``eroded_mask_for_qc`` (including masks edited and overwritten
in 3D Slicer) and regenerates all downstream CSV files.
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def summarize(vals: np.ndarray):
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }

def _empty_stats():
    """Stats placeholder for slices/compartments with zero valid voxels."""
    return {
        "mean": np.nan,
        "median": np.nan,
        "std": np.nan,
        "min": np.nan,
        "max": np.nan,
    }


def compute_binary_mask_metrics(mask_path, ff_map_path, subject, day, compartment_name, out_dir):
    """
    Compute volume and per-slice metrics for a binary compartment mask.

    Important behavior:
      - The volume CSV contains the total compartment volume/statistics.
      - The slice CSV now writes one row for EVERY Dixon slice, even when the
        compartment is absent on that slice. Empty slices get n_voxels=0,
        CSA_cm2=0, and NaN statistics.

    This makes SAT/VAT/IMAT slice CSVs directly comparable by slice_index.
    """
    mask_img = nib.load(mask_path)
    ff_img = nib.load(ff_map_path)

    mask = mask_img.get_fdata() > 0
    ff = ff_img.get_fdata()

    dx, dy, dz = nib.affines.voxel_sizes(ff_img.affine)
    pixel_area_cm2 = float(dx * dy) / 100.0
    voxel_vol_ml = float(dx * dy * dz) / 1000.0

    vals = ff[mask]
    vals = vals[np.isfinite(vals)]

    vol_rows = []
    if vals.size > 0:
        stats = summarize(vals)
        vol_rows.append({
            "compartment": compartment_name,
            "n_voxels": int(vals.size),
            "volume_ml": int(vals.size) * voxel_vol_ml,
            **stats
        })
    else:
        vol_rows.append({
            "compartment": compartment_name,
            "n_voxels": 0,
            "volume_ml": 0.0,
            **_empty_stats()
        })

    slice_rows = []
    zdim = mask.shape[2]
    for z in range(zdim):
        sl_mask = mask[:, :, z]
        vals_z = ff[:, :, z][sl_mask]
        vals_z = vals_z[np.isfinite(vals_z)]

        if vals_z.size > 0:
            stats = summarize(vals_z)
            n_vox = int(vals_z.size)
        else:
            stats = _empty_stats()
            n_vox = 0

        slice_rows.append({
            "slice_index": int(z),
            "compartment": compartment_name,
            "n_voxels": n_vox,
            "CSA_cm2": n_vox * pixel_area_cm2,
            **stats
        })

    tag = f"{subject}_{day}" if day else subject

    pd.DataFrame(vol_rows).to_csv(
        os.path.join(out_dir, f"{tag}_{compartment_name}_volume.csv"),
        index=False
    )
    pd.DataFrame(slice_rows).to_csv(
        os.path.join(out_dir, f"{tag}_{compartment_name}_slice.csv"),
        index=False
    )


def compute_fat_compartment_slice_comparison(mask_paths: dict, ff_map_path, subject, day, out_dir):
    """
    Create one wide per-slice CSV for direct compartment comparison.

    Output columns include, for each compartment:
      <Compartment>_n_voxels
      <Compartment>_CSA_cm2
      <Compartment>_mean
      <Compartment>_median
      <Compartment>_std
      <Compartment>_min
      <Compartment>_max

    This is intended for direct VAT-vs-SAT comparisons on identical Dixon slices.
    """
    ff_img = nib.load(ff_map_path)
    ff = ff_img.get_fdata()

    dx, dy, dz = nib.affines.voxel_sizes(ff_img.affine)
    pixel_area_cm2 = float(dx * dy) / 100.0

    masks = {}
    for compartment_name, mask_path in mask_paths.items():
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata() > 0
        if mask.shape != ff.shape:
            raise ValueError(
                f"Mask shape mismatch for {compartment_name}: "
                f"mask={mask.shape}, FF={ff.shape}"
            )
        masks[compartment_name] = mask

    rows = []
    for z in range(ff.shape[2]):
        row = {"slice_index": int(z)}
        for compartment_name, mask in masks.items():
            sl_mask = mask[:, :, z]
            vals_z = ff[:, :, z][sl_mask]
            vals_z = vals_z[np.isfinite(vals_z)]

            if vals_z.size > 0:
                stats = summarize(vals_z)
                n_vox = int(vals_z.size)
            else:
                stats = _empty_stats()
                n_vox = 0

            prefix = str(compartment_name)
            row[f"{prefix}_n_voxels"] = n_vox
            row[f"{prefix}_CSA_cm2"] = n_vox * pixel_area_cm2
            for stat_name, stat_value in stats.items():
                row[f"{prefix}_{stat_name}"] = stat_value

        rows.append(row)

    tag = f"{subject}_{day}" if day else subject
    out_path = os.path.join(out_dir, f"{tag}_fat_compartment_slice_comparison.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    saved: {out_path}")



def compute_fat_compartment_vat_slice_volume_comparison(mask_paths: dict, ff_map_path, subject, day, out_dir):
    """
    Create one summary CSV comparing SAT, IMAT, and VAT volumes using ONLY the
    Dixon slices where VAT is present.

    This is useful when SAT and VAT need to be compared across the exact same
    anatomical z-range. The slice range is defined by VAT_mask > 0 on a slice.
    """
    ff_img = nib.load(ff_map_path)
    ff = ff_img.get_fdata()

    dx, dy, dz = nib.affines.voxel_sizes(ff_img.affine)
    voxel_vol_ml = float(dx * dy * dz) / 1000.0

    masks = {}
    for compartment_name, mask_path in mask_paths.items():
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata() > 0
        if mask.shape != ff.shape:
            raise ValueError(
                f"Mask shape mismatch for {compartment_name}: "
                f"mask={mask.shape}, FF={ff.shape}"
            )
        masks[compartment_name] = mask

    if "VAT" not in masks:
        raise ValueError("VAT mask is required to define VAT-containing slices.")

    vat_by_slice = np.any(masks["VAT"], axis=(0, 1))
    vat_slice_indices = np.where(vat_by_slice)[0].astype(int).tolist()

    rows = []
    for compartment_name, mask in masks.items():
        restricted_mask = mask.copy()
        if vat_by_slice.size != restricted_mask.shape[2]:
            raise ValueError("VAT slice vector length does not match mask z-dimension.")

        # Keep only slices where VAT exists.
        restricted_mask[:, :, ~vat_by_slice] = False

        vals = ff[restricted_mask]
        vals = vals[np.isfinite(vals)]

        if vals.size > 0:
            stats = summarize(vals)
            n_vox = int(vals.size)
        else:
            stats = _empty_stats()
            n_vox = 0

        rows.append({
            "restriction": "slices_with_VAT",
            "compartment": compartment_name,
            "vat_slice_count": int(len(vat_slice_indices)),
            "vat_slice_indices": ";".join(str(x) for x in vat_slice_indices),
            "n_voxels": n_vox,
            "volume_ml": n_vox * voxel_vol_ml,
            **stats,
        })

    tag = f"{subject}_{day}" if day else subject
    out_path = os.path.join(out_dir, f"{tag}_fat_compartment_volume_in_VAT_slices.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    saved: {out_path}")

def compute_label_map_metrics(label_map_path, label_csv_path, ff_map_path, subject, day, out_dir):
    """Compute volume/slice metrics for every nonzero label in a multi-label map."""
    label_img = nib.load(label_map_path)
    ff_img = nib.load(ff_map_path)

    labels = np.rint(label_img.get_fdata()).astype(np.int32)
    ff = ff_img.get_fdata()

    dx, dy, dz = nib.affines.voxel_sizes(ff_img.affine)
    pixel_area_cm2 = float(dx * dy) / 100.0
    voxel_vol_ml = float(dx * dy * dz) / 1000.0

    lookup = pd.read_csv(label_csv_path)
    id_col = "LabelID" if "LabelID" in lookup.columns else lookup.columns[0]
    name_col = "Name" if "Name" in lookup.columns else lookup.columns[1]

    tag = f"{subject}_{day}" if day else subject

    for _, row in lookup.iterrows():
        label_id = int(row[id_col])
        compartment_name = str(row[name_col])
        mask = labels == label_id

        vals = ff[mask]
        vals = vals[np.isfinite(vals)]

        vol_rows = []
        if vals.size > 0:
            stats = summarize(vals)
            vol_rows.append({
                "compartment": compartment_name,
                "label_id": label_id,
                "n_voxels": int(vals.size),
                "volume_ml": int(vals.size) * voxel_vol_ml,
                **stats,
            })

        slice_rows = []
        for z in range(labels.shape[2]):
            sl_mask = mask[:, :, z]
            if not np.any(sl_mask):
                continue

            vals_z = ff[:, :, z][sl_mask]
            vals_z = vals_z[np.isfinite(vals_z)]
            if vals_z.size == 0:
                continue

            stats = summarize(vals_z)
            slice_rows.append({
                "slice_index": int(z),
                "compartment": compartment_name,
                "label_id": label_id,
                "n_voxels": int(vals_z.size),
                "CSA_cm2": int(vals_z.size) * pixel_area_cm2,
                **stats,
            })

        pd.DataFrame(vol_rows).to_csv(
            os.path.join(out_dir, f"{tag}_{compartment_name}_volume.csv"),
            index=False,
        )
        pd.DataFrame(slice_rows).to_csv(
            os.path.join(out_dir, f"{tag}_{compartment_name}_slice.csv"),
            index=False,
        )

def _require_files(paths: dict):
    missing = [f"{name}: {path}" for name, path in paths.items() if not Path(path).exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Metrics-only export cannot continue because required QC files are missing:\n"
            f"  - {joined}\n"
            "Run build_and_metrics first, or restore the missing QC mask(s)."
        )


def export_all_fat_compartment_metrics(base_dir, subject, day=""):
    """Regenerate all fat-compartment CSVs from the current QC masks.

    The function intentionally reads masks already present in
    ``eroded_mask_for_qc``. Therefore, edits saved from 3D Slicer are treated as
    the final segmentation source of truth.
    """
    base = Path(base_dir)
    qc_dir = base / "eroded_mask_for_qc"
    csv_dir = base / "fat compartment csvs"
    csv_dir.mkdir(parents=True, exist_ok=True)

    ff_path = base / "Dixon_FF_map.nii.gz"
    sat_mask_path = qc_dir / "SAT_mask.nii.gz"
    imat_mask_path = qc_dir / "IMAT_mask.nii.gz"
    vat_mask_path = qc_dir / "VAT_mask.nii.gz"
    thoracic_labels_path = qc_dir / "Thoracic_trunk_fat_labels.nii.gz"
    thoracic_labels_csv_path = qc_dir / "Thoracic_trunk_fat_labels.csv"
    pericardial_mask_path = qc_dir / "PericardialFat_mask.nii.gz"

    required = {
        "Dixon FF map": ff_path,
        "SAT mask": sat_mask_path,
        "IMAT mask": imat_mask_path,
        "VAT mask": vat_mask_path,
        "thoracic trunk fat labels": thoracic_labels_path,
        "thoracic label lookup CSV": thoracic_labels_csv_path,
        "pericardial fat mask": pericardial_mask_path,
    }
    _require_files(required)

    print("Re-exporting metrics from current QC masks")
    print(f"    QC source: {qc_dir}")
    print(f"    CSV output: {csv_dir}")

    for name, path in [
        ("SAT", sat_mask_path),
        ("IMAT", imat_mask_path),
        ("VAT", vat_mask_path),
    ]:
        print(f"Computing {name} metrics")
        compute_binary_mask_metrics(path, ff_path, subject, day, name, csv_dir)

    mask_paths = {
        "SAT": sat_mask_path,
        "IMAT": imat_mask_path,
        "VAT": vat_mask_path,
    }

    print("Computing fat compartment slice comparison")
    compute_fat_compartment_slice_comparison(
        mask_paths, ff_path, subject, day, csv_dir
    )

    print("Computing fat compartment volumes restricted to VAT-containing slices")
    compute_fat_compartment_vat_slice_volume_comparison(
        mask_paths, ff_path, subject, day, csv_dir
    )

    print("Computing thoracic trunk fat metrics")
    compute_label_map_metrics(
        thoracic_labels_path,
        thoracic_labels_csv_path,
        ff_path,
        subject,
        day,
        csv_dir,
    )

    print("Computing standalone pericardial fat metrics")
    compute_binary_mask_metrics(
        pericardial_mask_path,
        ff_path,
        subject,
        day,
        "PericardialFat",
        csv_dir,
    )

    print("Fat compartment metrics re-export DONE")
    return csv_dir

