#!/usr/bin/env python
"""
fat_compartment_pipeline.py

WholeBodySeg fat compartment module.

Runs after:
  1. musclemap_dixon_pipeline.py
     - Dixon_FF_map.nii.gz
     - Dixon_W_COMP_dseg.nii.gz
     - eroded_mask_for_qc/Dixon_W_COMP_eroded_dseg.nii.gz
  2. totalseg_pipeline.py
     - TotalSegmentator outputs for future organ/anatomy exclusions

Current version:
  - Builds SAT and IMAT masks using Dixon FF map + MuscleMap muscle segmentation.
  - Builds VAT by carving internal trunk/pelvis fat out of the initial SAT mask using TotalSegmentator anatomy.
  - Computes SAT, IMAT, and VAT metrics.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage as ndi


def summarize(vals: np.ndarray):
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def dseg_name_for(fn: str) -> str:
    low = fn.lower()
    if low.endswith(".nii.gz"):
        return fn[:-7] + "_dseg.nii.gz"
    if low.endswith(".nii"):
        return fn[:-4] + "_dseg.nii"
    return fn + "_dseg.nii.gz"


def _largest_two_components_2d(mask2d: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask2d)
    if n == 0:
        return mask2d.astype(bool)
    sizes = ndi.sum(mask2d, lab, index=np.arange(1, n + 1))
    keep_labels = np.argsort(sizes)[-2:] + 1
    return np.isin(lab, keep_labels)


def erode_mask_inplane(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool).copy()

    structure = np.ones((3, 3), dtype=bool)
    out = np.zeros_like(mask, dtype=bool)

    for z in range(mask.shape[2]):
        sl = mask[:, :, z].astype(bool)
        if np.any(sl):
            out[:, :, z] = ndi.binary_erosion(sl, structure=structure, iterations=iterations)

    return out


def erode_segmentation_per_label(seg: np.ndarray, iterations: int = 1) -> np.ndarray:
    if iterations <= 0:
        return seg.copy()

    eroded_seg = np.zeros_like(seg, dtype=seg.dtype)
    labels = np.unique(seg)
    labels = labels[labels > 0]

    for lab in labels:
        lab_mask = seg == lab
        lab_eroded = erode_mask_inplane(lab_mask, iterations=iterations)
        eroded_seg[lab_eroded] = lab

    return eroded_seg


def save_like(reference_img: nib.Nifti1Image, data: np.ndarray, out_path: str, dtype=None):
    arr = data.astype(dtype) if dtype is not None else data
    img = nib.Nifti1Image(arr, reference_img.affine, reference_img.header)
    nib.save(img, out_path)


def find_required_dixon_inputs(base: str):
    files = [f for f in os.listdir(base) if f.lower().endswith(".nii") or f.lower().endswith(".nii.gz")]

    w_files = [f for f in files if "w_comp" in f.lower() and "_dseg" not in f.lower()]
    f_files = [f for f in files if "f_comp" in f.lower() and "_dseg" not in f.lower()]

    if not w_files:
        raise FileNotFoundError(f"Could not find Dixon W image in: {base}")
    if not f_files:
        raise FileNotFoundError(f"Could not find Dixon F image in: {base}")

    w_fn = w_files[0]
    f_fn = f_files[0]

    w_path = os.path.join(base, w_fn)
    f_path = os.path.join(base, f_fn)
    ff_path = os.path.join(base, "Dixon_FF_map.nii.gz")
    seg_path = os.path.join(base, dseg_name_for(w_fn))

    if not os.path.exists(ff_path):
        raise FileNotFoundError(f"Missing FF map. Run musclemap_dixon_pipeline first: {ff_path}")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Missing Dixon W segmentation. Run musclemap_dixon_pipeline first: {seg_path}")

    return w_path, f_path, ff_path, seg_path


def load_total_segmentator_mask(base: str, reference_shape):
    """
    Load TotalSegmentator combined anatomy labels as a binary anatomy mask.

    Returns an all-False mask if TotalSeg output is missing or does not match the Dixon grid.
    """
    label_path = Path(base) / "TotalSegmentator_nonMuscleMap_labels.nii.gz"

    if not label_path.exists():
        print("    NOTE: TotalSeg combined label map not found; VAT/SAT correction will use non-TotalSeg logic only.")
        return np.zeros(reference_shape, dtype=bool)

    try:
        ts_img = nib.load(str(label_path))
        ts = np.rint(ts_img.get_fdata()).astype(np.int32)
    except Exception as e:
        print(f"    NOTE: Could not read TotalSeg label map ({e}); VAT/SAT correction will use non-TotalSeg logic only.")
        return np.zeros(reference_shape, dtype=bool)

    if ts.shape != tuple(reference_shape):
        print(
            "    NOTE: TotalSeg label map shape does not match Dixon grid; "
            f"TotalSeg logic skipped. TotalSeg={ts.shape}, Dixon={tuple(reference_shape)}"
        )
        return np.zeros(reference_shape, dtype=bool)

    return ts > 0



def load_total_segmentator_labels(base: str, reference_shape):
    """
    Load TotalSegmentator combined anatomy label image and label CSV.

    Returns:
      labels_img: int ndarray, same shape as Dixon, or zeros if unavailable
      label_name_to_id: dict name -> integer label ID
    """
    label_path = Path(base) / "TotalSegmentator_nonMuscleMap_labels.nii.gz"
    csv_path = Path(base) / "TotalSegmentator_nonMuscleMap_labels.csv"

    if not label_path.exists() or not csv_path.exists():
        return np.zeros(reference_shape, dtype=np.int32), {}

    try:
        img = nib.load(str(label_path))
        labels_img = np.rint(img.get_fdata()).astype(np.int32)
    except Exception:
        return np.zeros(reference_shape, dtype=np.int32), {}

    if labels_img.shape != tuple(reference_shape):
        return np.zeros(reference_shape, dtype=np.int32), {}

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return labels_img, {}

    # Be tolerant of column naming.
    cols = {str(c).strip().lower(): c for c in df.columns}
    name_col = None
    id_col = None
    for candidate in ["name", "label", "structure", "mask_name", "class_name"]:
        if candidate in cols:
            name_col = cols[candidate]
            break
    for candidate in ["label_id", "id", "label_value", "value"]:
        if candidate in cols:
            id_col = cols[candidate]
            break

    label_name_to_id = {}
    if name_col is not None and id_col is not None:
        for _, row in df.iterrows():
            try:
                name = str(row[name_col]).strip().lower()
                lab = int(row[id_col])
                label_name_to_id[name] = lab
            except Exception:
                pass

    return labels_img, label_name_to_id



def _load_binary_nifti_mask(mask_path: Path, reference_shape, label: str = "mask") -> np.ndarray:
    """
    Load a binary NIfTI mask only if it matches the Dixon grid.
    Returns all-False if unavailable/mismatched.
    """
    if not mask_path.exists():
        return np.zeros(reference_shape, dtype=bool)

    try:
        img = nib.load(str(mask_path))
        arr = img.get_fdata() > 0
    except Exception as e:
        print(f"    NOTE: Could not read {label} ({mask_path}): {e}")
        return np.zeros(reference_shape, dtype=bool)

    if arr.shape != tuple(reference_shape):
        print(
            f"    NOTE: {label} shape does not match Dixon grid; skipped. "
            f"{label}={arr.shape}, Dixon={tuple(reference_shape)}"
        )
        return np.zeros(reference_shape, dtype=bool)

    return arr.astype(bool)


def load_totalseg_abdominal_cavity(
    base: str,
    reference_shape,
    folder_name: str = "TS_TRUNK_CAVITIES",
) -> np.ndarray:
    """
    Prefer TotalSegmentator trunk_cavities/abdominal_cavity.nii.gz when available.

    This mask is used as the primary internal cavity for VAT/SAT carving.
    """
    base = Path(base)
    candidates = [
        base / folder_name / "abdominal_cavity.nii.gz",
        base / folder_name / "abdominal_cavity.nii",
    ]

    for p in candidates:
        mask = _load_binary_nifti_mask(p, reference_shape, label="TotalSeg abdominal_cavity")
        if np.any(mask):
            print(f"    Using TotalSeg abdominal cavity mask: {p}")
            return mask

    print(f"    NOTE: TotalSeg abdominal cavity mask not found in {base / folder_name}; using fallback cavity logic.")
    return np.zeros(reference_shape, dtype=bool)


def load_totalseg_torso_fat(
    base: str,
    reference_shape,
    folder_name: str = "TS_TISSUE_TYPES_MR",
) -> np.ndarray:
    """
    Load TotalSegmentator tissue_types_mr/torso_fat.nii.gz when available.

    This is used preferentially as the VAT mask because it appears to separate
    internal trunk fat better than the heuristic SAT-carving method.
    """
    base = Path(base)
    folder = base / folder_name
    candidates = [
        folder / "torso_fat.nii.gz",
        folder / "torso_fat.nii",
    ]

    for p in candidates:
        mask = _load_binary_nifti_mask(p, reference_shape, label="TotalSeg tissue_types_mr torso_fat")
        if np.any(mask):
            print(f"    Using TotalSeg tissue_types_mr torso_fat mask for VAT: {p}")
            return mask

    print(f"    NOTE: TotalSeg torso_fat mask not found in {folder}; using abdominal-cavity VAT fallback.")
    return np.zeros(reference_shape, dtype=bool)


def load_totalseg_subcutaneous_fat(
    base: str,
    reference_shape,
    folder_name: str = "TS_TISSUE_TYPES_MR",
) -> np.ndarray:
    """
    Load TotalSegmentator tissue_types_mr/subcutaneous_fat.nii.gz when available.

    This is used preferentially as the SAT mask because tissue_types_mr
    separates the superficial fat shell better than the heuristic fascia-based
    SAT method in whole-body Dixon stations.
    """
    base = Path(base)
    folder = base / folder_name
    candidates = [
        folder / "subcutaneous_fat.nii.gz",
        folder / "subcutaneous_fat.nii",
        folder / "body_fat.nii.gz",
        folder / "body_fat.nii",
    ]

    for p in candidates:
        mask = _load_binary_nifti_mask(p, reference_shape, label="TotalSeg tissue_types_mr subcutaneous_fat")
        if np.any(mask):
            print(f"    Using TotalSeg tissue_types_mr subcutaneous fat mask for SAT: {p}")
            return mask

    print(f"    NOTE: TotalSeg subcutaneous_fat mask not found in {folder}; using computed SAT fallback.")
    return np.zeros(reference_shape, dtype=bool)


def load_totalseg_abdominal_muscles(
    base: str,
    reference_shape,
    folder_name: str = "TS_ABDOMINAL_MUSCLES",
) -> np.ndarray:
    """
    Load and combine all TotalSegmentator abdominal_muscles subtask masks.

    These are used as a trunk/body-wall muscle boundary for SAT/VAT separation.
    They are also excluded from fat candidates so muscle voxels are not counted
    as SAT/VAT/IMAT.
    """
    folder = Path(base) / folder_name
    if not folder.exists():
        print(f"    NOTE: TotalSeg abdominal muscles folder not found: {folder}")
        return np.zeros(reference_shape, dtype=bool)

    mask_files = sorted(list(folder.glob("*.nii.gz")) + list(folder.glob("*.nii")))
    if not mask_files:
        print(f"    NOTE: No abdominal muscle NIfTI masks found in: {folder}")
        return np.zeros(reference_shape, dtype=bool)

    combined = np.zeros(reference_shape, dtype=bool)
    used = []
    for p in mask_files:
        m = _load_binary_nifti_mask(p, reference_shape, label=f"abdominal muscle mask {p.name}")
        if np.any(m):
            combined |= m
            used.append(p.name)

    if used:
        print(f"    Using TotalSeg abdominal muscle masks from {folder}: {', '.join(used)}")
    else:
        print(f"    NOTE: Abdominal muscle masks were present but none matched/contained voxels: {folder}")

    return combined

def build_lung_block_mask(base: str, reference_shape, thoracic_extension_slices: int = 15):
    """
    Build a slice-wise mask that blocks abdominal cavity/VAT only after allowing
    the abdominal cavity mask to extend superiorly into the lower thorax.

    Uses TotalSegmentator lung_left/lung_right labels when available. The previous
    behavior stopped the abdominal cavity near the most-inferior lung slice, which
    caused SAT to bleed into the diaphragm/lower thoracic region. This version
    permits the cavity/VAT correction to continue upward by
    ``thoracic_extension_slices`` before blocking it.
    """
    labels_img, name_to_id = load_total_segmentator_labels(base, reference_shape)
    if not name_to_id or labels_img.shape != tuple(reference_shape):
        return np.zeros(reference_shape, dtype=bool)

    lung_ids = []
    for key in ["lung_left", "lung_right"]:
        if key in name_to_id:
            lung_ids.append(name_to_id[key])

    if not lung_ids:
        return np.zeros(reference_shape, dtype=bool)

    lung_mask = np.isin(labels_img, lung_ids)
    lung_by_z = np.any(lung_mask, axis=(0, 1))
    lung_z = np.where(lung_by_z)[0]

    if lung_z.size == 0:
        return np.zeros(reference_shape, dtype=bool)

    # In this ASHA/WholeBodySeg orientation, higher axial z indices correspond to
    # more superior slices for the displayed files. The block begins above the
    # inferior lung slice only after allowing a configurable lower-thorax extension.
    inferior_lung_z = int(np.min(lung_z))
    block_start_z = inferior_lung_z + int(thoracic_extension_slices)

    block = np.zeros(reference_shape, dtype=bool)
    if block_start_z < reference_shape[2]:
        block[:, :, max(0, block_start_z):] = True
    return block


def build_internal_cavity_from_totalseg_slice(
    ts_slice: np.ndarray,
    body_slice: np.ndarray,
    muscle_slice: np.ndarray,
    surface_erode_iter: int = 12,
) -> np.ndarray:
    """
    Estimate the abdominal/thoracic internal cavity wall on one axial slice.

    This is intentionally different from an organ-shaped mask:
      1. Use the signal-derived body mask as the outer boundary.
      2. Erode inward to remove the superficial SAT shell/body wall.
      3. Use TotalSeg anatomy only as a landmark to confirm this is a trunk/pelvis slice.
      4. Keep the connected internal component that contains the TotalSeg anatomy.
      5. Remove muscle from that internal cavity.

    Result:
      - In trunk/pelvis slices, returns a broad internal cavity region.
      - In limb-only slices, returns empty and preserves existing limb SAT/IMAT logic.
    """
    if not np.any(body_slice) or not np.any(ts_slice):
        return np.zeros_like(body_slice, dtype=bool)

    if int(np.count_nonzero(ts_slice)) < 250:
        return np.zeros_like(body_slice, dtype=bool)

    body = ndi.binary_fill_holes(body_slice.astype(bool))
    body = _largest_two_components_2d(body)

    inner = ndi.binary_erosion(
        body,
        structure=np.ones((3, 3), dtype=bool),
        iterations=int(surface_erode_iter),
        border_value=0,
    )

    if not np.any(inner):
        return np.zeros_like(body_slice, dtype=bool)

    inner = ndi.binary_closing(inner, structure=np.ones((9, 9), dtype=bool))
    inner = inner & (~muscle_slice.astype(bool))

    anchor = ndi.binary_dilation(ts_slice.astype(bool), structure=np.ones((7, 7), dtype=bool), iterations=2)
    anchor = anchor & inner

    if not np.any(anchor):
        return np.zeros_like(body_slice, dtype=bool)

    lab, n = ndi.label(inner)
    if n == 0:
        return np.zeros_like(body_slice, dtype=bool)

    anchor_labels = np.unique(lab[anchor])
    anchor_labels = anchor_labels[anchor_labels != 0]
    if anchor_labels.size == 0:
        return np.zeros_like(body_slice, dtype=bool)

    cavity = np.isin(lab, anchor_labels)
    cavity = ndi.binary_closing(cavity, structure=np.ones((11, 11), dtype=bool))
    cavity = ndi.binary_fill_holes(cavity)
    cavity = cavity & body & (~muscle_slice.astype(bool))

    return cavity




def smooth_and_extend_cavity_mask_3d(
    cavity_mask: np.ndarray,
    signal: np.ndarray,
    muscle_mask: np.ndarray,
    lung_block: np.ndarray,
    max_superior_extend_slices: int = 15,
    min_seed_area_voxels: int = 500,
) -> np.ndarray:
    """
    Stabilize the slice-wise abdominal cavity mask in 3D.

    The per-slice cavity estimate can collapse near the diaphragm, which lets SAT
    flood into internal trunk fat. This function smooths the detected cavity and,
    when needed, propagates the last good cavity superiorly for a configurable
    number of slices while respecting the body signal, muscle, and lung-block mask.
    """
    cavity = cavity_mask.astype(bool).copy()

    # First, smooth existing detections across adjacent slices without allowing
    # growth outside the body or into muscle.
    if np.any(cavity):
        cavity = ndi.binary_closing(cavity, structure=np.ones((5, 5, 3), dtype=bool))
        cavity = cavity & signal.astype(bool) & (~muscle_mask.astype(bool)) & (~lung_block.astype(bool))

    areas = np.count_nonzero(cavity, axis=(0, 1))
    good_z = np.where(areas >= int(min_seed_area_voxels))[0]
    if good_z.size == 0:
        return cavity

    zdim = cavity.shape[2]
    max_ext = max(0, int(max_superior_extend_slices))

    # Determine likely superior direction from the lung block. In the current
    # WholeBodySeg orientation, superior is increasing z. This keeps the logic
    # tied to the same orientation assumption already used for the lung block.
    superior_step = 1
    z = int(good_z.max())
    last = cavity[:, :, z].copy()

    for i in range(1, max_ext + 1):
        zn = z + superior_step * i
        if zn < 0 or zn >= zdim:
            break
        if lung_block[:, :, zn].any():
            break

        current_area = int(np.count_nonzero(cavity[:, :, zn]))
        last_area = int(np.count_nonzero(last))
        if last_area == 0:
            break

        # Only fill weak/collapsed slices. If the slice already has a reasonable
        # cavity, use it as the new template.
        if current_area >= 0.7 * last_area:
            last = cavity[:, :, zn].copy()
            continue

        body2d = ndi.binary_fill_holes(signal[:, :, zn].astype(bool))
        body2d = _largest_two_components_2d(body2d)
        allowed = body2d & (~muscle_mask[:, :, zn].astype(bool)) & (~lung_block[:, :, zn].astype(bool))

        propagated = ndi.binary_closing(last, structure=np.ones((7, 7), dtype=bool))
        propagated = ndi.binary_fill_holes(propagated)
        propagated = propagated & allowed

        if np.count_nonzero(propagated) >= 0.3 * last_area:
            cavity[:, :, zn] = propagated
            last = propagated

    return cavity

def build_sat_imat_masks(
    water_path: str,
    fat_path: str,
    ff_path: str,
    muscle_seg_path: str,
    sat_mask_out: str,
    imat_mask_out: str,
    vat_mask_out: str = None,
    cavity_mask_out: str = None,
    signal_threshold: float = 50.0,
    fat_ff_threshold_imat: float = 0.2,
    fat_ff_threshold_sat: float = 0.3,
    fat_ff_threshold_vat: float = 0.3,
    fascia_dilate_size: int = 3,
    fascia_close_size: int = 15,
    erode_voxels: int = 1,
    eroded_seg_out: str = None,
    use_totalseg_exclusions: bool = True,
    base_dir: str = "",
    abdominal_cavity_thoracic_extension_slices: int = 15,
    cavity_smooth_3d: bool = True,
    use_totalseg_abdominal_cavity: bool = True,
    totalseg_trunk_cavities_folder: str = "TS_TRUNK_CAVITIES",
    use_totalseg_abdominal_muscles: bool = True,
    totalseg_abdominal_muscles_folder: str = "TS_ABDOMINAL_MUSCLES",
    abdominal_wall_dilate_size: int = 5,
    abdominal_wall_close_size: int = 7,
    abdominal_muscles_mask_out: str = None,
    use_totalseg_torso_fat_for_vat: bool = True,
    use_totalseg_subcutaneous_fat_for_sat: bool = True,
    totalseg_tissue_types_mr_folder: str = "TS_TISSUE_TYPES_MR",
    torso_fat_mask_out: str = None,
    subcutaneous_fat_mask_out: str = None
):
    w_img = nib.load(water_path)
    f_img = nib.load(fat_path)
    ff_img = nib.load(ff_path)
    seg_img = nib.load(muscle_seg_path)

    w = w_img.get_fdata().astype(np.float32)
    f = f_img.get_fdata().astype(np.float32)
    ff = ff_img.get_fdata().astype(np.float32)
    seg = np.rint(seg_img.get_fdata()).astype(np.int32)

    signal = (w + f) > signal_threshold
    seg_eroded = erode_segmentation_per_label(seg, iterations=erode_voxels)
    muscle_mask_eroded = seg_eroded > 0

    # TotalSeg anatomy is used in two ways:
    #   1. Exclude labeled organs/anatomy from fat candidates.
    #   2. Estimate trunk/pelvis internal cavity so VAT can be carved out of SAT.
    ts_anatomy = np.zeros_like(signal, dtype=bool)
    if use_totalseg_exclusions:
        ts_anatomy = load_total_segmentator_mask(base_dir or os.path.dirname(ff_path), signal.shape)

    exclusion = ts_anatomy

    # Block cavity/VAT at and above the lungs so the abdominal cavity mask does
    # not extend into the thorax. This does not affect SAT/IMAT directly.
    lung_block = np.zeros_like(signal, dtype=bool)
    if use_totalseg_exclusions:
        lung_block = build_lung_block_mask(
            base_dir or os.path.dirname(ff_path),
            signal.shape,
            thoracic_extension_slices=abdominal_cavity_thoracic_extension_slices,
        )

    direct_abdominal_cavity = np.zeros_like(signal, dtype=bool)
    if use_totalseg_exclusions and use_totalseg_abdominal_cavity:
        direct_abdominal_cavity = load_totalseg_abdominal_cavity(
            base_dir or os.path.dirname(ff_path),
            signal.shape,
            folder_name=totalseg_trunk_cavities_folder,
        )

    abdominal_muscle_mask = np.zeros_like(signal, dtype=bool)
    if use_totalseg_exclusions and use_totalseg_abdominal_muscles:
        abdominal_muscle_mask = load_totalseg_abdominal_muscles(
            base_dir or os.path.dirname(ff_path),
            signal.shape,
            folder_name=totalseg_abdominal_muscles_folder,
        )

    torso_fat_direct = np.zeros_like(signal, dtype=bool)
    if use_totalseg_exclusions and use_totalseg_torso_fat_for_vat:
        torso_fat_direct = load_totalseg_torso_fat(
            base_dir or os.path.dirname(ff_path),
            signal.shape,
            folder_name=totalseg_tissue_types_mr_folder,
        )

    subcutaneous_fat_direct = np.zeros_like(signal, dtype=bool)
    if use_totalseg_exclusions and use_totalseg_subcutaneous_fat_for_sat:
        subcutaneous_fat_direct = load_totalseg_subcutaneous_fat(
            base_dir or os.path.dirname(ff_path),
            signal.shape,
            folder_name=totalseg_tissue_types_mr_folder,
        )

    abdominal_muscle_exclusion = abdominal_muscle_mask.astype(bool)

    fat_candidate_imat = signal & np.isfinite(ff) & (ff >= fat_ff_threshold_imat) & (~exclusion) & (~abdominal_muscle_exclusion)
    fat_candidate_sat = signal & np.isfinite(ff) & (ff >= fat_ff_threshold_sat) & (~exclusion) & (~abdominal_muscle_exclusion)
    fat_candidate_vat = signal & np.isfinite(ff) & (ff >= fat_ff_threshold_vat) & (~exclusion) & (~abdominal_muscle_exclusion)

    sat_mask = np.zeros_like(fat_candidate_sat, dtype=bool)
    imat_mask = np.zeros_like(fat_candidate_imat, dtype=bool)
    vat_mask = np.zeros_like(fat_candidate_vat, dtype=bool)
    cavity_mask = np.zeros_like(fat_candidate_vat, dtype=bool)

    # Save the pre-VAT/carve versions so a smoothed 3D cavity mask can be used
    # after the slice-wise pass. This prevents SAT leakage when the cavity
    # estimate briefly collapses near the diaphragm.
    sat_pre_cavity_carve = np.zeros_like(fat_candidate_sat, dtype=bool)
    imat_pre_cavity_carve = np.zeros_like(fat_candidate_imat, dtype=bool)

    zdim = ff.shape[2]
    dilate_size = max(3, int(fascia_dilate_size))
    close_size = max(3, int(fascia_close_size))
    dilate_structure = np.ones((dilate_size, dilate_size), dtype=bool)
    close_structure = np.ones((close_size, close_size), dtype=bool)

    for z in range(zdim):
        sig = signal[:, :, z]
        fc_imat = fat_candidate_imat[:, :, z]
        fc_sat = fat_candidate_sat[:, :, z]
        fc_vat = fat_candidate_vat[:, :, z]
        ts_slice = ts_anatomy[:, :, z]
        muscle_union = seg[:, :, z] > 0
        muscle_union_eroded = muscle_mask_eroded[:, :, z]
        abdominal_wall_slice = abdominal_muscle_mask[:, :, z]
        muscle_union_for_fascia = muscle_union | abdominal_wall_slice
        muscle_union_eroded_for_fat = muscle_union_eroded | abdominal_wall_slice

        if not np.any(sig):
            continue

        body = ndi.binary_fill_holes(sig)
        body = _largest_two_components_2d(body)

        # Fat candidates exclude TotalSeg anatomy, but the cavity wall is estimated
        # from the full body mask so organs do not create holes in the wall estimate.
        body_for_fat = body & (~exclusion[:, :, z])

        fc_imat = fc_imat & body_for_fat
        fc_sat = fc_sat & body_for_fat
        fc_vat = fc_vat & body_for_fat

        if not np.any(fc_imat) and not np.any(fc_sat) and not np.any(fc_vat):
            continue

        if np.any(muscle_union_for_fascia):
            fascia_seed = muscle_union_for_fascia

            # When TotalSeg abdominal muscles are present, use a slightly stronger
            # trunk/body-wall closing to keep the abdominal wall continuous.
            if np.any(abdominal_wall_slice):
                aw_dilate = max(3, int(abdominal_wall_dilate_size))
                aw_close = max(3, int(abdominal_wall_close_size))
                fascia = ndi.binary_dilation(fascia_seed, structure=np.ones((aw_dilate, aw_dilate), dtype=bool))
                fascia = ndi.binary_closing(fascia, structure=np.ones((aw_close, aw_close), dtype=bool))
            else:
                fascia = ndi.binary_dilation(fascia_seed, structure=dilate_structure)
                fascia = ndi.binary_closing(fascia, structure=close_structure)

            fascia = ndi.binary_fill_holes(fascia)
            fascia = fascia & body_for_fat
        else:
            fascia = np.zeros_like(body, dtype=bool)

        imat_slice = fc_imat & fascia & (~muscle_union_eroded_for_fat)
        sat_slice = fc_sat & body & (~fascia)

        sat_pre_cavity_carve[:, :, z] = sat_slice
        imat_pre_cavity_carve[:, :, z] = imat_slice

        # Trunk/pelvis correction:
        # The limb-style SAT definition can overcall internal abdominal/pelvic fat as SAT.
        # Use TotalSeg anatomy to estimate the internal cavity, then carve that region
        # out of SAT and reassign high-FF voxels to VAT.
        internal_cavity = build_internal_cavity_from_totalseg_slice(
            ts_slice=ts_slice,
            body_slice=body,
            muscle_slice=muscle_union_eroded,
            surface_erode_iter=12,
        )

        # Do not allow abdominal cavity/VAT at or above lung-containing slices.
        if lung_block[:, :, z].any():
            internal_cavity = np.zeros_like(internal_cavity, dtype=bool)

        # VAT is carved from the over-inclusive SAT region only inside the estimated
        # abdominal/thoracic cavity wall.
        vat_slice = sat_slice & internal_cavity & fc_vat & (~muscle_union_eroded)

        # Correct SAT after carving out VAT. Limb-only slices have empty internal_cavity,
        # so limb SAT/IMAT behavior is preserved.
        sat_slice = sat_slice & (~vat_slice)

        # Keep IMAT mostly unchanged, but prevent overlap if VAT was carved from a trunk slice.
        imat_slice = imat_slice & (~vat_slice)

        sat_mask[:, :, z] = sat_slice
        imat_mask[:, :, z] = imat_slice
        vat_mask[:, :, z] = vat_slice
        cavity_mask[:, :, z] = internal_cavity

    if use_totalseg_exclusions and np.any(direct_abdominal_cavity):
        # Prefer the explicit TotalSegmentator trunk_cavities abdominal_cavity mask.
        # Restrict to body signal and exclude MuscleMap/abdominal muscle voxels.
        cavity_mask = direct_abdominal_cavity & signal & (~muscle_mask_eroded) & (~abdominal_muscle_mask)
        if cavity_smooth_3d:
            cavity_mask = ndi.binary_closing(cavity_mask, structure=np.ones((5, 5, 3), dtype=bool))
            cavity_mask = cavity_mask & signal & (~muscle_mask_eroded) & (~abdominal_muscle_mask)
    elif cavity_smooth_3d and use_totalseg_exclusions:
        cavity_mask = smooth_and_extend_cavity_mask_3d(
            cavity_mask=cavity_mask,
            signal=signal,
            muscle_mask=(muscle_mask_eroded | abdominal_muscle_mask),
            lung_block=lung_block,
            max_superior_extend_slices=abdominal_cavity_thoracic_extension_slices,
        )

    if use_totalseg_exclusions and use_totalseg_torso_fat_for_vat and np.any(torso_fat_direct):
        # Prefer TotalSegmentator tissue_types_mr/torso_fat as VAT.
        # Restrict to signal and fat-fraction threshold to avoid non-fat tissue leakage.
        muscle_exclusion_for_fat = muscle_mask_eroded | abdominal_muscle_mask
        vat_mask = torso_fat_direct & signal & fat_candidate_vat & (~muscle_exclusion_for_fat)

        # Remove any directly identified torso fat from SAT/IMAT. This replaces the
        # older abdominal-cavity carving approach when torso_fat is available.
        sat_mask = sat_pre_cavity_carve & (~vat_mask)
        imat_mask = imat_pre_cavity_carve & (~vat_mask)

    elif use_totalseg_exclusions and (np.any(cavity_mask) or np.any(direct_abdominal_cavity)):
        # Re-carve VAT from the pre-cavity SAT mask using the best available cavity.
        # This removes SAT bleed-over in internal trunk slices.
        muscle_exclusion_for_fat = muscle_mask_eroded | abdominal_muscle_mask
        vat_mask = sat_pre_cavity_carve & cavity_mask & fat_candidate_vat & (~muscle_exclusion_for_fat)
        sat_mask = sat_pre_cavity_carve & (~vat_mask)
        imat_mask = imat_pre_cavity_carve & (~vat_mask)

    if use_totalseg_exclusions and use_totalseg_subcutaneous_fat_for_sat and np.any(subcutaneous_fat_direct):
        # Prefer TotalSegmentator tissue_types_mr/subcutaneous_fat as SAT.
        # Keep the FF threshold and muscle/VAT exclusions so the saved SAT mask is
        # conservative and non-overlapping with VAT/IMAT.
        muscle_exclusion_for_fat = muscle_mask_eroded | abdominal_muscle_mask
        sat_mask = subcutaneous_fat_direct & signal & fat_candidate_sat & (~muscle_exclusion_for_fat) & (~vat_mask)
        imat_mask = imat_mask & (~sat_mask) & (~vat_mask)

    save_like(ff_img, sat_mask, sat_mask_out, dtype=np.uint8)
    save_like(ff_img, imat_mask, imat_mask_out, dtype=np.uint8)

    if vat_mask_out:
        save_like(ff_img, vat_mask, vat_mask_out, dtype=np.uint8)

    if cavity_mask_out:
        save_like(ff_img, cavity_mask, cavity_mask_out, dtype=np.uint8)

    if abdominal_muscles_mask_out:
        save_like(ff_img, abdominal_muscle_mask, abdominal_muscles_mask_out, dtype=np.uint8)

    if torso_fat_mask_out:
        save_like(ff_img, torso_fat_direct, torso_fat_mask_out, dtype=np.uint8)

    if subcutaneous_fat_mask_out:
        save_like(ff_img, subcutaneous_fat_direct, subcutaneous_fat_mask_out, dtype=np.uint8)

    if eroded_seg_out:
        save_like(seg_img, seg_eroded, eroded_seg_out, dtype=np.int16)


def compute_binary_mask_metrics(mask_path, ff_map_path, subject, day, compartment_name, out_dir):
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

    slice_rows = []
    zdim = mask.shape[2]
    for z in range(zdim):
        sl_mask = mask[:, :, z]
        if not np.any(sl_mask):
            continue

        vals = ff[:, :, z][sl_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        stats = summarize(vals)
        slice_rows.append({
            "slice_index": int(z),
            "compartment": compartment_name,
            "n_voxels": int(vals.size),
            "CSA_cm2": int(vals.size) * pixel_area_cm2,
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


def main():
    ap = argparse.ArgumentParser(description="WholeBodySeg SAT/IMAT fat compartment module.")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--day", default="")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--signal_threshold", type=float, default=50.0)
    ap.add_argument("--fat_ff_threshold_imat", type=float, default=0.2)
    ap.add_argument("--fat_ff_threshold_sat", type=float, default=0.3)
    ap.add_argument("--fat_ff_threshold_vat", type=float, default=0.3)
    ap.add_argument("--fascia_dilate_size", type=int, default=3)
    ap.add_argument("--fascia_close_size", type=int, default=15)
    ap.add_argument("--erode_voxels", type=int, default=1)
    ap.add_argument("--abdominal_cavity_thoracic_extension_slices", type=int, default=15)
    ap.add_argument("--totalseg_trunk_cavities_folder", default="TS_TRUNK_CAVITIES")
    ap.add_argument("--totalseg_abdominal_muscles_folder", default="TS_ABDOMINAL_MUSCLES")
    ap.add_argument("--totalseg_tissue_types_mr_folder", default="TS_TISSUE_TYPES_MR")
    ap.add_argument("--disable_totalseg_torso_fat_for_vat", action="store_true")
    ap.add_argument("--disable_totalseg_subcutaneous_fat_for_sat", action="store_true")
    ap.add_argument("--abdominal_wall_dilate_size", type=int, default=5)
    ap.add_argument("--abdominal_wall_close_size", type=int, default=7)
    ap.add_argument("--disable_totalseg_abdominal_cavity", action="store_true")
    ap.add_argument("--disable_totalseg_abdominal_muscles", action="store_true")
    ap.add_argument("--disable_cavity_smooth_3d", action="store_true")
    ap.add_argument("--disable_totalseg_exclusions", action="store_true")
    args = ap.parse_args()

    base = args.dir

    w_path, f_path, ff_path, seg_path = find_required_dixon_inputs(base)

    qc_dir = os.path.join(base, "eroded_mask_for_qc")
    os.makedirs(qc_dir, exist_ok=True)

    eroded_seg_path = os.path.join(qc_dir, "Dixon_W_COMP_eroded_dseg.nii.gz")
    sat_mask_path = os.path.join(qc_dir, "SAT_mask.nii.gz")
    imat_mask_path = os.path.join(qc_dir, "IMAT_mask.nii.gz")
    vat_mask_path = os.path.join(qc_dir, "VAT_mask.nii.gz")
    cavity_mask_path = os.path.join(qc_dir, "Abdominal_cavity_mask.nii.gz")
    abdominal_muscles_mask_path = os.path.join(qc_dir, "Abdominal_muscles_mask.nii.gz")
    torso_fat_mask_path = os.path.join(qc_dir, "TS_torso_fat_for_VAT_mask.nii.gz")
    subcutaneous_fat_mask_path = os.path.join(qc_dir, "TS_subcutaneous_fat_for_SAT_mask.nii.gz")

    print("Building SAT, IMAT, VAT, and abdominal cavity QC masks")
    build_sat_imat_masks(
        water_path=w_path,
        fat_path=f_path,
        ff_path=ff_path,
        muscle_seg_path=seg_path,
        sat_mask_out=sat_mask_path,
        imat_mask_out=imat_mask_path,
        vat_mask_out=vat_mask_path,
        cavity_mask_out=cavity_mask_path,
        signal_threshold=args.signal_threshold,
        fat_ff_threshold_imat=args.fat_ff_threshold_imat,
        fat_ff_threshold_sat=args.fat_ff_threshold_sat,
        fat_ff_threshold_vat=args.fat_ff_threshold_vat,
        fascia_dilate_size=args.fascia_dilate_size,
        fascia_close_size=args.fascia_close_size,
        erode_voxels=args.erode_voxels,
        eroded_seg_out=eroded_seg_path,
        use_totalseg_exclusions=not args.disable_totalseg_exclusions,
        base_dir=base,
        abdominal_cavity_thoracic_extension_slices=args.abdominal_cavity_thoracic_extension_slices,
        cavity_smooth_3d=not args.disable_cavity_smooth_3d,
        use_totalseg_abdominal_cavity=not args.disable_totalseg_abdominal_cavity,
        totalseg_trunk_cavities_folder=args.totalseg_trunk_cavities_folder,
        use_totalseg_abdominal_muscles=not args.disable_totalseg_abdominal_muscles,
        totalseg_abdominal_muscles_folder=args.totalseg_abdominal_muscles_folder,
        abdominal_wall_dilate_size=args.abdominal_wall_dilate_size,
        abdominal_wall_close_size=args.abdominal_wall_close_size,
        abdominal_muscles_mask_out=abdominal_muscles_mask_path,
        use_totalseg_torso_fat_for_vat=not args.disable_totalseg_torso_fat_for_vat,
        use_totalseg_subcutaneous_fat_for_sat=not args.disable_totalseg_subcutaneous_fat_for_sat,
        totalseg_tissue_types_mr_folder=args.totalseg_tissue_types_mr_folder,
        torso_fat_mask_out=torso_fat_mask_path,
        subcutaneous_fat_mask_out=subcutaneous_fat_mask_path
    )

    print("Computing SAT metrics")
    compute_binary_mask_metrics(sat_mask_path, ff_path, args.subject, args.day, "SAT", base)

    print("Computing IMAT metrics")
    compute_binary_mask_metrics(imat_mask_path, ff_path, args.subject, args.day, "IMAT", base)

    print("Computing VAT metrics")
    compute_binary_mask_metrics(vat_mask_path, ff_path, args.subject, args.day, "VAT", base)

    print("Fat compartment module DONE")


def run_fat_compartments(station_dir, cfg):
    station_dir = Path(station_dir)
    subject = cfg.get("current_subject", "")
    session = cfg.get("current_session", "")

    argv_old = sys.argv[:]

    sys.argv = [
        "fat_compartment_pipeline.py",
        "--subject", subject,
        "--day", session,
        "--dir", str(station_dir),
        "--signal_threshold", str(cfg.get("signal_threshold", 50.0)),
        "--fat_ff_threshold_imat", str(cfg.get("fat_ff_threshold_imat", 0.2)),
        "--fat_ff_threshold_sat", str(cfg.get("fat_ff_threshold_sat", 0.3)),
        "--fat_ff_threshold_vat", str(cfg.get("fat_ff_threshold_vat", 0.3)),
        "--fascia_dilate_size", str(cfg.get("fascia_dilate_size", 3)),
        "--fascia_close_size", str(cfg.get("fascia_close_size", 15)),
        "--erode_voxels", str(cfg.get("erode_voxels", 1)),
        "--abdominal_cavity_thoracic_extension_slices", str(cfg.get("abdominal_cavity_thoracic_extension_slices", 15)),
        "--totalseg_trunk_cavities_folder", str(cfg.get("totalseg_trunk_cavities_folder", "TS_TRUNK_CAVITIES")),
        "--totalseg_abdominal_muscles_folder", str(cfg.get("totalseg_abdominal_muscles_folder", "TS_ABDOMINAL_MUSCLES")),
        "--totalseg_tissue_types_mr_folder", str(cfg.get("totalseg_tissue_types_mr_folder", "TS_TISSUE_TYPES_MR")),
        "--abdominal_wall_dilate_size", str(cfg.get("abdominal_wall_dilate_size", 5)),
        "--abdominal_wall_close_size", str(cfg.get("abdominal_wall_close_size", 7)),
    ]

    if not cfg.get("use_totalseg_abdominal_cavity", True):
        sys.argv += ["--disable_totalseg_abdominal_cavity"]

    if not cfg.get("use_totalseg_abdominal_muscles", True):
        sys.argv += ["--disable_totalseg_abdominal_muscles"]

    if not cfg.get("use_totalseg_torso_fat_for_vat", True):
        sys.argv += ["--disable_totalseg_torso_fat_for_vat"]

    if not cfg.get("use_totalseg_subcutaneous_fat_for_sat", True):
        sys.argv += ["--disable_totalseg_subcutaneous_fat_for_sat"]

    if not cfg.get("cavity_smooth_3d", True):
        sys.argv += ["--disable_cavity_smooth_3d"]

    if not cfg.get("use_totalseg_fat_exclusions", True):
        sys.argv += ["--disable_totalseg_exclusions"]

    try:
        main()
    finally:
        sys.argv = argv_old


if __name__ == "__main__":
    main()
