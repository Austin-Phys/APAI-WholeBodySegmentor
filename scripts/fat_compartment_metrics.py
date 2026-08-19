#!/usr/bin/env python
"""
fat_compartment_metrics.py

Metrics-only exporter for WholeBodySeg fat compartment QC masks.

This module never creates or modifies segmentation masks. It reads the current
NIfTI files from ``fat_compartment_masks`` (including masks edited and overwritten
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



def _find_superior_iliac_crest_slice(ff_img, hip_mask_paths):
    """
    Determine the axial slice at the superior-most extent of the left/right hip
    masks. For the TotalSegmentator total_mr hip segmentation, this serves as an
    automated approximation of the top of the iliac crest.

    Selection is based on RAS+ physical superior coordinate (world Z), not raw
    NIfTI slice index, so reversed slice ordering is handled correctly.

    Returns:
      crest_slice_index, crest_world_z_mm, slice_world_z_mm
    """
    ref_shape = ff_img.shape
    affine = ff_img.affine

    hip_any = np.zeros(ref_shape, dtype=bool)
    used = []

    for hip_name, hip_path in hip_mask_paths.items():
        hip_path = Path(hip_path)
        if not hip_path.exists():
            continue

        hip_img = nib.load(str(hip_path))
        hip = hip_img.get_fdata() > 0
        if hip.shape != ref_shape:
            raise ValueError(
                f"Hip landmark mask shape mismatch for {hip_name}: "
                f"mask={hip.shape}, FF={ref_shape}"
            )

        hip_any |= hip
        used.append(str(hip_path))

    if not used or not np.any(hip_any):
        raise FileNotFoundError(
            "Cannot normalize VAT-slice metrics to the iliac crest because no "
            "usable total_mr hip_left/hip_right mask was found. Expected masks "
            "in TS_TOTAL_MR_FULL."
        )

    # Physical superior coordinate of the center of each axial voxel plane.
    zdim = ref_shape[2]
    slice_world_z_mm = np.asarray([
        nib.affines.apply_affine(affine, [0.0, 0.0, float(z)])[2]
        for z in range(zdim)
    ], dtype=float)

    hip_by_slice = np.any(hip_any, axis=(0, 1))
    hip_slices = np.where(hip_by_slice)[0]

    # "Top of iliac crest" = most physically superior slice containing hip bone.
    crest_slice_index = int(
        hip_slices[np.argmax(slice_world_z_mm[hip_slices])]
    )
    crest_world_z_mm = float(slice_world_z_mm[crest_slice_index])

    return crest_slice_index, crest_world_z_mm, slice_world_z_mm


def compute_fat_compartment_vat_slice_volume_comparison(
    mask_paths: dict,
    ff_map_path,
    subject,
    day,
    out_dir,
    hip_mask_paths=None,
):
    """
    Compare SAT, IMAT, and VAT volumes over a standardized abdominal range.

    Inferior boundary:
      superior-most slice containing the TotalSegmentator total_mr hip_left or
      hip_right mask, used as an automated approximation of the top of the
      iliac crest.

    Superior boundary:
      highest anatomically superior slice containing VAT.

    Only VAT-containing slices at or superior to the iliac-crest landmark are
    included. This removes lower pelvic fat and makes the inferior boundary less
    dependent on how far inferiorly an individual Dixon acquisition happened to
    extend.
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

    if not hip_mask_paths:
        raise ValueError(
            "hip_mask_paths is required for iliac-crest-normalized VAT-slice metrics."
        )

    crest_slice_index, crest_world_z_mm, slice_world_z_mm = _find_superior_iliac_crest_slice(
        ff_img, hip_mask_paths
    )

    vat_by_slice = np.any(masks["VAT"], axis=(0, 1))

    # Anatomically superior to (or on) the iliac-crest slice. A small numerical
    # tolerance prevents floating-point noise from excluding the landmark slice.
    superior_to_crest = slice_world_z_mm >= (crest_world_z_mm - 1e-6)

    analysis_by_slice = vat_by_slice & superior_to_crest
    analysis_slice_indices = np.where(analysis_by_slice)[0].astype(int).tolist()

    if not analysis_slice_indices:
        raise ValueError(
            "No VAT-containing slices were found at or superior to the detected "
            f"iliac-crest landmark (slice {crest_slice_index})."
        )

    analysis_world_z = slice_world_z_mm[analysis_by_slice]
    inferior_world_z_mm = float(np.min(analysis_world_z))
    superior_world_z_mm = float(np.max(analysis_world_z))

    rows = []
    for compartment_name, mask in masks.items():
        restricted_mask = mask.copy()
        if analysis_by_slice.size != restricted_mask.shape[2]:
            raise ValueError("Analysis slice vector length does not match mask z-dimension.")

        restricted_mask[:, :, ~analysis_by_slice] = False

        vals = ff[restricted_mask]
        vals = vals[np.isfinite(vals)]

        if vals.size > 0:
            stats = summarize(vals)
            n_vox = int(vals.size)
        else:
            stats = _empty_stats()
            n_vox = 0

        rows.append({
            "restriction": "VAT-containing slices from top of iliac crest superiorly",
            "inferior_landmark": "Top of iliac crest",
            "landmark_source": "TotalSegmentator total_mr hip_left/hip_right superior-most physical slice",
            "compartment": compartment_name,
            "iliac_crest_slice_index": crest_slice_index,
            "iliac_crest_world_z_mm": crest_world_z_mm,
            "analysis_inferior_world_z_mm": inferior_world_z_mm,
            "analysis_superior_world_z_mm": superior_world_z_mm,
            "analysis_slice_count": int(len(analysis_slice_indices)),
            "analysis_slice_indices": ";".join(str(x) for x in analysis_slice_indices),
            "n_voxels": n_vox,
            "volume_ml": n_vox * voxel_vol_ml,
            **stats,
        })

    tag = f"{subject}_{day}" if day else subject
    out_path = os.path.join(out_dir, f"{tag}_fat_compartment_volume_in_VAT_slices.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(
        "    Iliac-crest normalization: "
        f"crest slice={crest_slice_index}, physical Z={crest_world_z_mm:.2f} mm; "
        f"using {len(analysis_slice_indices)} VAT-containing slices at/superior to crest."
    )
    print(f"    saved: {out_path}")


def compute_fat_compartment_summary(
    sat_mask_path,
    vat_mask_path,
    imat_mask_path,
    thoracic_labels_path,
    pericardial_mask_path,
    ff_map_path,
    subject,
    day,
    out_dir,
):
    """
    Create one overall fat-compartment summary CSV from the CURRENT QC masks.

    Rows:
      SAT
      VAT
      IMAT
      ThoracicTrunkFat
      PericardialFat
      MediastinalFat
      NonSpecificTrunkFat

    Thoracic_trunk_fat_labels convention:
      1 = PericardialFat
      2 = MediastinalFat
      3 = NonSpecificTrunkFat

    ThoracicTrunkFat is retained as the combined total of all non-zero
    thoracic labels.

    Metrics:
      n_voxels
      volume_ml
      mean / median / std / min / max Dixon fat fraction

    Because this function reads the masks from fat_compartment_masks each time it is
    called, manual QC edits saved over those NIfTI files are automatically
    reflected during a metrics-only re-export.
    """
    ff_img = nib.load(ff_map_path)
    ff = ff_img.get_fdata()

    dx, dy, dz = nib.affines.voxel_sizes(ff_img.affine)
    voxel_vol_ml = float(dx * dy * dz) / 1000.0

    sat = nib.load(sat_mask_path).get_fdata() > 0
    vat = nib.load(vat_mask_path).get_fdata() > 0
    imat = nib.load(imat_mask_path).get_fdata() > 0

    thoracic_raw = np.rint(
        nib.load(thoracic_labels_path).get_fdata()
    ).astype(np.int32)

    pericardial_binary = nib.load(pericardial_mask_path).get_fdata() > 0

    for name, arr in {
        "SAT": sat,
        "VAT": vat,
        "IMAT": imat,
        "ThoracicTrunkFat": thoracic_raw,
        "PericardialFat": pericardial_binary,
    }.items():
        if arr.shape != ff.shape:
            raise ValueError(
                f"Mask shape mismatch for {name}: mask={arr.shape}, FF={ff.shape}"
            )

    masks = [
        ("SAT", sat),
        ("VAT", vat),
        ("IMAT", imat),
        ("ThoracicTrunkFat", thoracic_raw > 0),
        ("PericardialFat", thoracic_raw == 1),
        ("MediastinalFat", thoracic_raw == 2),
        ("NonSpecificTrunkFat", thoracic_raw == 3),
    ]

    rows = []
    for compartment, mask in masks:
        n_mask_voxels = int(np.count_nonzero(mask))
        vals = ff[mask]
        vals = vals[np.isfinite(vals)]

        stats = summarize(vals) if vals.size > 0 else _empty_stats()

        rows.append({
            "subject": subject,
            "day": day,
            "compartment": compartment,
            "n_voxels": n_mask_voxels,
            "volume_ml": n_mask_voxels * voxel_vol_ml,
            "n_voxels_with_valid_ff": int(vals.size),
            **stats,
        })

    tag = f"{subject}_{day}" if day else subject
    out_path = os.path.join(out_dir, f"{tag}_fat_compartment_summary.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    saved: {out_path}")
    return out_path


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
    ``fat_compartment_masks``. Therefore, edits saved from 3D Slicer are treated as
    the final segmentation source of truth.
    """
    base = Path(base_dir)
    qc_dir = base / "fat_compartment_masks"
    misc_dir = base / "misc"
    csv_dir = base / "fat compartment csvs"
    csv_dir.mkdir(parents=True, exist_ok=True)

    ff_path = base / "Dixon_FF_map.nii.gz"
    sat_mask_path = qc_dir / "SAT_mask.nii.gz"
    imat_mask_path = qc_dir / "IMAT_mask.nii.gz"
    vat_mask_path = qc_dir / "VAT_mask.nii.gz"
    thoracic_labels_path = qc_dir / "Thoracic_trunk_fat_labels.nii.gz"
    pericardial_mask_path = qc_dir / "PericardialFat_mask.nii.gz"

    # total_mr hip masks provide the standardized inferior landmark for the
    # VAT-slice volume comparison (superior-most hip = top of iliac crest).
    total_mr_dir = base / "TS_TOTAL_MR_FULL"
    hip_mask_paths = {
        "hip_left": total_mr_dir / "hip_left.nii.gz",
        "hip_right": total_mr_dir / "hip_right.nii.gz",
    }

    # Lookup metadata is diagnostic/supporting output, so it lives in misc.
    thoracic_labels_csv_path = misc_dir / "Thoracic_trunk_fat_labels.csv"

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

    if not any(Path(p).exists() for p in hip_mask_paths.values()):
        raise FileNotFoundError(
            "Iliac-crest-normalized fat metrics require at least one TotalSegmentator "
            "total_mr hip mask. Expected hip_left.nii.gz and/or hip_right.nii.gz in "
            f"{total_mr_dir}"
        )

    print("Re-exporting all fat metrics from current fat_compartment_masks QC masks")
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

    print("Computing fat compartment volumes using top of iliac crest as the inferior boundary")
    compute_fat_compartment_vat_slice_volume_comparison(
        mask_paths,
        ff_path,
        subject,
        day,
        csv_dir,
        hip_mask_paths=hip_mask_paths,
    )

    print("Computing overall fat compartment summary")
    compute_fat_compartment_summary(
        sat_mask_path=sat_mask_path,
        vat_mask_path=vat_mask_path,
        imat_mask_path=imat_mask_path,
        thoracic_labels_path=thoracic_labels_path,
        pericardial_mask_path=pericardial_mask_path,
        ff_map_path=ff_path,
        subject=subject,
        day=day,
        out_dir=csv_dir,
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

