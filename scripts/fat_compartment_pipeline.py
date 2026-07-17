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



def load_totalseg_trunk_cavity_exclusions(
    base: str,
    reference_shape,
    folder_name: str = "TS_TRUNK_CAVITIES",
    exclude_names=None,
) -> np.ndarray:
    """
    Load TotalSegmentator trunk_cavities masks that should be excluded from fat
    compartments, especially SAT.

    Important:
      - abdominal_cavity is intentionally NOT included by default because it is
        used for VAT/SAT separation.
      - thoracic_cavity, pericardium/pericardial cavity, and mediastinum are
        excluded so superficial/trunk SAT does not leak into these cavities.
    """
    base = Path(base)
    folder = base / folder_name

    if exclude_names is None:
        exclude_names = [
            "thoracic_cavity",
            "pericardium",
            "pericardial_cavity",
            "mediastinum",
            "pleural_cavity",
            "pleural_cavity_left",
            "pleural_cavity_right",
        ]

    if not folder.exists():
        print(f"    NOTE: TotalSeg trunk cavities folder not found for cavity exclusions: {folder}")
        return np.zeros(reference_shape, dtype=bool)

    combined = np.zeros(reference_shape, dtype=bool)
    used = []

    for name in exclude_names:
        candidates = [
            folder / f"{name}.nii.gz",
            folder / f"{name}.nii",
        ]
        for candidate in candidates:
            m = _load_binary_nifti_mask(candidate, reference_shape, label=f"TotalSeg trunk cavity exclusion {name}")
            if np.any(m):
                combined |= m
                used.append(candidate.name)
                break

    if used:
        print(f"    Excluding TotalSeg trunk cavity masks from fat compartments: {', '.join(used)}")
    else:
        print(f"    NOTE: No matching TotalSeg trunk cavity exclusion masks found in {folder}")

    return combined




def load_totalseg_trunk_cavity_masks(
    base: str,
    reference_shape,
    folder_name: str = "TS_TRUNK_CAVITIES",
) -> dict:
    """
    Load the individual TotalSegmentator trunk_cavities masks needed to build
    thoracic trunk fat compartments.

    Expected masks from trunk_cavities:
      - thoracic_cavity.nii.gz
      - pericardium.nii.gz
      - mediastinum.nii.gz

    Returns a dict of name -> binary mask. Missing/mismatched masks are returned
    as all-False masks so downstream logic can run safely.
    """
    base = Path(base)
    folder = base / folder_name
    names = ["thoracic_cavity", "pericardium", "mediastinum"]
    masks = {}

    if not folder.exists():
        print(f"    NOTE: TotalSeg trunk cavities folder not found for thoracic fat labels: {folder}")
        return {name: np.zeros(reference_shape, dtype=bool) for name in names}

    for name in names:
        mask = np.zeros(reference_shape, dtype=bool)
        for candidate in [folder / f"{name}.nii.gz", folder / f"{name}.nii"]:
            m = _load_binary_nifti_mask(candidate, reference_shape, label=f"TotalSeg trunk cavity {name}")
            if np.any(m):
                mask = m
                print(f"    Using TotalSeg trunk cavity mask for thoracic fat labels: {candidate}")
                break
        masks[name] = mask

    return masks


def _mask_name_matches(name: str, include_terms) -> bool:
    """Return True when a TotalSegmentator mask name matches any include term."""
    low = str(name).lower()
    return any(str(term).lower() in low for term in include_terms)


def load_totalseg_named_anatomy_exclusion(
    base: str,
    reference_shape,
    folder_names=None,
    include_terms=None,
    eroded_label_filename: str = "TotalSegmentator_nonMuscleMap_labels_eroded1.nii.gz",
) -> np.ndarray:
    """
    Load organ/bone/anatomy masks that should be excluded from thoracic fat labels.

    Preferred behavior:
      - Use the precomputed eroded TotalSegmentator label map located directly
        in the station folder: TotalSegmentator_nonMuscleMap_labels_eroded1.nii.gz
      - Use TotalSegmentator_nonMuscleMap_labels.csv as the lookup table.

    This prevents thoracic trunk fat from counting lung, heart, spine/vertebrae,
    ribs/sternum, aorta, esophagus, trachea/bronchi, and similar non-fat anatomy,
    while avoiding over-conservative exclusion from full organ boundaries.

    Fallback behavior:
      - If the eroded label map is missing, fall back to the full combined label map.
      - If the combined label map/CSV are missing, fall back to individual masks
        in TotalSegmentator output folders.
    """
    base = Path(base)
    if folder_names is None:
        folder_names = ["TS_TOTAL_MR_FULL", "TS_EXTRA_FOR_LABELS"]
    if include_terms is None:
        include_terms = [
            "lung", "heart", "vertebra", "spinal_cord", "rib", "sternum",
            "aorta", "esophagus", "trachea", "bronch",
        ]

    # 1) Preferred: use precomputed eroded combined label map in the station folder.
    # The CSV label table remains the same as the full combined label map.
    csv_path = base / "TotalSegmentator_nonMuscleMap_labels.csv"
    preferred_label_paths = [
        base / eroded_label_filename,
        base / "TotalSegmentator_nonMuscleMap_labels.nii.gz",
    ]

    for label_path in preferred_label_paths:
        if not label_path.exists() or not csv_path.exists():
            continue
        try:
            img = nib.load(str(label_path))
            labels_img = np.rint(img.get_fdata()).astype(np.int32)
        except Exception as e:
            print(f"    NOTE: Could not read thoracic anatomy label map {label_path.name}: {e}")
            continue

        if labels_img.shape != tuple(reference_shape):
            print(
                f"    NOTE: Thoracic anatomy label map shape mismatch; skipped. "
                f"{label_path.name}={labels_img.shape}, Dixon={tuple(reference_shape)}"
            )
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"    NOTE: Could not read TotalSegmentator label CSV for thoracic exclusions: {e}")
            continue

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

        if name_col is None or id_col is None:
            print("    NOTE: TotalSegmentator label CSV missing name/id columns for thoracic exclusions.")
            continue

        ids = []
        matched_names = []
        for _, row in df.iterrows():
            try:
                name = str(row[name_col]).strip().lower()
                lab = int(row[id_col])
            except Exception:
                continue
            if _mask_name_matches(name, include_terms):
                ids.append(lab)
                matched_names.append(name)

        if ids:
            combined = np.isin(labels_img, ids)
            if np.any(combined):
                source_label = "eroded" if label_path.name == eroded_label_filename else "full fallback"
                preview = ", ".join(matched_names[:12])
                if len(matched_names) > 12:
                    preview += f", ... (+{len(matched_names)-12} more)"
                print(
                    f"    Excluding thoracic organ/bone anatomy from trunk fat labels "
                    f"using {source_label} label map {label_path.name}: {preview}"
                )
                return combined

    # 2) Last-resort fallback: individual full-resolution masks.
    combined = np.zeros(reference_shape, dtype=bool)
    used = []

    for folder_name in folder_names:
        folder = base / folder_name
        if not folder.exists():
            continue
        for p in sorted(list(folder.glob("*.nii.gz")) + list(folder.glob("*.nii"))):
            stem = p.name
            if stem.lower().endswith(".nii.gz"):
                stem = stem[:-7]
            elif stem.lower().endswith(".nii"):
                stem = stem[:-4]
            if not _mask_name_matches(stem, include_terms):
                continue
            m = _load_binary_nifti_mask(p, reference_shape, label=f"thoracic anatomy exclusion {stem}")
            if np.any(m):
                combined |= m
                used.append(f"{folder_name}/{p.name}")

    if used:
        preview = ", ".join(used[:12])
        if len(used) > 12:
            preview += f", ... (+{len(used)-12} more)"
        print(f"    Excluding thoracic organ/bone anatomy from trunk fat labels using individual full-mask fallback: {preview}")
    else:
        print("    NOTE: No thoracic organ/bone anatomy exclusions found for trunk fat labels.")
    return combined


def save_slicer_color_table(out_ctbl: str):
    """Save a Slicer-compatible color table so label IDs display with names."""
    lines = [
        "# Color table file generated by WholeBodySeg",
        "# Label Name R G B A",
        "0 Background 0 0 0 0",
        "1 PericardialFat 95 190 95 255",
        "2 MediastinalFat 245 210 120 255",
        "3 NonSpecificTrunkFat 230 150 120 255",
    ]
    with open(out_ctbl, "w", newline="") as f:
        f.write("\n".join(lines) + "\n")


def build_thoracic_trunk_fat_labels(
    signal: np.ndarray,
    ff: np.ndarray,
    thoracic_cavity: np.ndarray,
    pericardium: np.ndarray,
    mediastinum: np.ndarray,
    muscle_exclusion: np.ndarray,
    anatomy_exclusion: np.ndarray = None,
    fat_ff_threshold: float = 0.3,
) -> np.ndarray:
    """
    Build one mutually exclusive multi-label thoracic trunk fat segmentation.

    Label values:
      0 = background
      1 = PericardialFat
      2 = MediastinalFat
      3 = NonSpecificTrunkFat

    NonSpecificTrunkFat is fat inside thoracic_cavity that is not already
    assigned to pericardial or mediastinal fat.
    """
    if anatomy_exclusion is None:
        anatomy_exclusion = np.zeros(signal.shape, dtype=bool)

    candidate = (
        signal.astype(bool)
        & np.isfinite(ff)
        & (ff >= float(fat_ff_threshold))
        & (~muscle_exclusion.astype(bool))
        & (~anatomy_exclusion.astype(bool))
    )

    thoracic_cavity = thoracic_cavity.astype(bool)
    pericardium = pericardium.astype(bool)
    mediastinum = mediastinum.astype(bool)

    # Hierarchical, mutually exclusive assignment.
    pericardial_fat = candidate & pericardium
    mediastinal_fat = candidate & mediastinum & (~pericardial_fat)
    nonspecific_trunk_fat = candidate & thoracic_cavity & (~pericardium) & (~mediastinum)

    labels = np.zeros(signal.shape, dtype=np.uint8)
    labels[pericardial_fat] = 1
    labels[mediastinal_fat] = 2
    labels[nonspecific_trunk_fat] = 3
    return labels


def save_thoracic_trunk_fat_label_csv(out_csv: str):
    rows = [
        {"LabelID": 1, "Name": "PericardialFat", "Description": "Fat-fraction-positive voxels inside TotalSegmentator pericardium, excluding eroded organ/bone anatomy", "R": 95, "G": 190, "B": 95, "A": 255},
        {"LabelID": 2, "Name": "MediastinalFat", "Description": "Fat-fraction-positive voxels inside TotalSegmentator mediastinum, excluding eroded organ/bone anatomy", "R": 245, "G": 210, "B": 120, "A": 255},
        {"LabelID": 3, "Name": "NonSpecificTrunkFat", "Description": "Remaining fat-fraction-positive voxels inside thoracic cavity outside pericardium and mediastinum, excluding eroded organ/bone anatomy", "R": 230, "G": 150, "B": 120, "A": 255},
    ]
    pd.DataFrame(rows).to_csv(out_csv, index=False)

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


def grow_seeded_mask_inplane(
    seed_mask: np.ndarray,
    allowed_mask: np.ndarray,
    max_iterations: int = 5,
) -> np.ndarray:
    """
    Recover nearby allowed voxels from a trusted seed without permitting unlimited
    connected-component growth.

    Growth is restricted to the axial plane (3x3x1), which prevents a small seed
    from propagating long distances superiorly/inferiorly through loosely connected
    fat. The returned mask is always a subset of ``allowed_mask``.
    """
    seed = seed_mask.astype(bool) & allowed_mask.astype(bool)
    allowed = allowed_mask.astype(bool)

    if not np.any(seed):
        return np.zeros_like(allowed, dtype=bool)

    grown = seed.copy()
    structure = np.ones((3, 3, 1), dtype=bool)

    for _ in range(max(0, int(max_iterations))):
        expanded = ndi.binary_dilation(grown, structure=structure)
        updated = expanded & allowed
        if np.array_equal(updated, grown):
            break
        grown = updated

    return grown

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
    subcutaneous_fat_mask_out: str = None,
    trunk_cavity_exclusion_mask_out: str = None,
    thoracic_trunk_fat_labels_out: str = None,
    thoracic_trunk_fat_labels_csv_out: str = None,
    thoracic_trunk_fat_labels_ctbl_out: str = None,
    pericardial_fat_mask_out: str = None,
    station_name: str = "",
    upper_imat_dilate_voxels: int = 1
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
    trunk_cavity_exclusion = np.zeros_like(signal, dtype=bool)
    if use_totalseg_exclusions and use_totalseg_abdominal_cavity:
        direct_abdominal_cavity = load_totalseg_abdominal_cavity(
            base_dir or os.path.dirname(ff_path),
            signal.shape,
            folder_name=totalseg_trunk_cavities_folder,
        )
        trunk_cavity_exclusion = load_totalseg_trunk_cavity_exclusions(
            base_dir or os.path.dirname(ff_path),
            signal.shape,
            folder_name=totalseg_trunk_cavities_folder,
        )

    trunk_cavity_masks = {
        "thoracic_cavity": np.zeros_like(signal, dtype=bool),
        "pericardium": np.zeros_like(signal, dtype=bool),
        "mediastinum": np.zeros_like(signal, dtype=bool),
    }
    thoracic_anatomy_exclusion = np.zeros_like(signal, dtype=bool)
    if use_totalseg_exclusions:
        trunk_cavity_masks = load_totalseg_trunk_cavity_masks(
            base_dir or os.path.dirname(ff_path),
            signal.shape,
            folder_name=totalseg_trunk_cavities_folder,
        )
        thoracic_anatomy_exclusion = load_totalseg_named_anatomy_exclusion(
            base_dir or os.path.dirname(ff_path),
            signal.shape,
        )

    thoracic_trunk_fat_labels = np.zeros_like(signal, dtype=np.uint8)

    # Exclude standard TotalSeg anatomy plus selected trunk cavity masks from
    # fat candidates. The abdominal cavity itself remains available for VAT logic;
    # thoracic/pericardial/mediastinal spaces are blocked from SAT/IMAT/VAT.
    exclusion = ts_anatomy | trunk_cavity_exclusion

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

    # -------------------------------------------------------------------------
    # Compartment-first classification with seeded VAT recovery
    # -------------------------------------------------------------------------
    # The TotalSegmentator torso_fat mask is used as a trusted VAT seed rather than
    # as the final VAT result. Nearby Dixon-positive fat may be recovered only when
    # it lies inside the abdominal cavity and can be reached from that seed within a
    # limited number of in-plane growth steps while preserving the trusted torso_fat core. This avoids accepting every fatty
    # voxel in an imperfect abdominal-cavity mask (for example posterior spinal or
    # paraspinal extensions), while recovering small VAT regions missed by
    # tissue_types_mr.
    station_is_upper = str(station_name).strip().lower() == "upper"
    muscle_exclusion_for_fat = muscle_mask_eroded | abdominal_muscle_mask

    # Preserve the trusted torso_fat result as the VAT core. The abdominal cavity
    # is used only to recover additional nearby VAT; it is not allowed to delete
    # core torso_fat voxels when the cavity mask is locally incomplete.
    vat_core = (
        torso_fat_direct
        & fat_candidate_vat
        & (~muscle_exclusion_for_fat)
    )

    vat_recovery_allowed = (
        fat_candidate_vat
        & cavity_mask
        & (~muscle_exclusion_for_fat)
    )

    if (
        use_totalseg_exclusions
        and use_totalseg_torso_fat_for_vat
        and np.any(vat_core)
    ):
        # Include vat_core in the allowed domain so trusted seed voxels are retained
        # even when they fall just outside an imperfect abdominal-cavity mask.
        vat_growth_domain = vat_recovery_allowed | vat_core

        vat_recovered = grow_seeded_mask_inplane(
            seed_mask=vat_core,
            allowed_mask=vat_growth_domain,
            max_iterations=10,
        )
        vat_mask = vat_core | vat_recovered

        print(
            "    Upper/trunk VAT logic: preserved torso_fat core plus cavity-constrained "
            "Dixon VAT recovery (maximum 10 in-plane voxels)."
        )
    elif use_totalseg_exclusions and np.any(vat_recovery_allowed):
        # Conservative fallback when torso_fat is unavailable: retain the earlier
        # cavity-carved VAT rather than classifying all cavity fat as VAT.
        vat_mask = vat_mask & vat_recovery_allowed
        print(
            "    NOTE: torso_fat VAT core unavailable; using conservative "
            "cavity-carved VAT fallback."
        )
    else:
        vat_mask = np.zeros_like(fat_candidate_vat, dtype=bool)

    # SAT remains driven by the direct subcutaneous-fat mask when available;
    # otherwise retain the existing fascia/body-shell result. VAT has priority for
    # any accidental overlap inside the abdominal cavity.
    if use_totalseg_exclusions and use_totalseg_subcutaneous_fat_for_sat and np.any(subcutaneous_fat_direct):
        sat_mask = (
            subcutaneous_fat_direct
            & signal
            & fat_candidate_sat
            & (~muscle_exclusion_for_fat)
            & (~vat_mask)
        )
    else:
        sat_mask = sat_pre_cavity_carve & (~vat_mask)

    if station_is_upper:
        # Upper-station IMAT is restricted to a tight skeletal-muscle neighborhood.
        # This prevents residual intra-abdominal fat from being classified as IMAT.
        # The full (non-eroded) muscle labels provide the anatomical support, while
        # the eroded muscle core is excluded so the mask remains a fat compartment.
        upper_muscle_support = (seg > 0) | abdominal_muscle_mask
        dilate_iter = max(0, int(upper_imat_dilate_voxels))
        if dilate_iter > 0:
            upper_muscle_support = ndi.binary_dilation(
                upper_muscle_support,
                structure=np.ones((3, 3, 1), dtype=bool),
                iterations=dilate_iter,
            )

        imat_mask = (
            fat_candidate_imat
            & upper_muscle_support
            & (~muscle_mask_eroded)
            & (~vat_mask)
            & (~sat_mask)
        )
        print(
            f"    Upper-station compartment-first logic: cavity-defined VAT; "
            f"IMAT restricted to muscle support (dilation={dilate_iter} voxel(s))."
        )
    else:
        # Preserve the validated lower-body fascia-based IMAT behavior.
        imat_mask = imat_pre_cavity_carve & (~vat_mask) & (~sat_mask)
        print("    Lower-station logic: preserving fascia-based IMAT classification.")

    # Build a mutually exclusive thoracic trunk fat label map from the explicit
    # TotalSegmentator trunk_cavities outputs. These labels are intentionally
    # separate from SAT/VAT/IMAT.
    muscle_exclusion_for_thoracic_fat = muscle_mask_eroded | abdominal_muscle_mask
    if use_totalseg_exclusions and np.any(trunk_cavity_masks.get("thoracic_cavity", False)):
        thoracic_trunk_fat_labels = build_thoracic_trunk_fat_labels(
            signal=signal,
            ff=ff,
            thoracic_cavity=trunk_cavity_masks.get("thoracic_cavity"),
            pericardium=trunk_cavity_masks.get("pericardium"),
            mediastinum=trunk_cavity_masks.get("mediastinum"),
            muscle_exclusion=muscle_exclusion_for_thoracic_fat,
            anatomy_exclusion=thoracic_anatomy_exclusion,
            fat_ff_threshold=fat_ff_threshold_sat,
        )

    thoracic_trunk_fat_any = thoracic_trunk_fat_labels > 0

    # Final safety: never allow SAT/IMAT/VAT to include selected trunk cavity masks
    # or the newly assigned thoracic trunk fat labels. This is especially important
    # when using direct TotalSeg tissue_types_mr subcutaneous_fat as SAT, because it
    # can occasionally include thoracic or mediastinal cavity voxels.
    if use_totalseg_exclusions and np.any(trunk_cavity_exclusion):
        sat_mask = sat_mask & (~trunk_cavity_exclusion)
        imat_mask = imat_mask & (~trunk_cavity_exclusion)
        vat_mask = vat_mask & (~trunk_cavity_exclusion)

    if use_totalseg_exclusions and np.any(thoracic_trunk_fat_any):
        sat_mask = sat_mask & (~thoracic_trunk_fat_any)
        imat_mask = imat_mask & (~thoracic_trunk_fat_any)
        vat_mask = vat_mask & (~thoracic_trunk_fat_any)

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

    if trunk_cavity_exclusion_mask_out:
        save_like(ff_img, trunk_cavity_exclusion, trunk_cavity_exclusion_mask_out, dtype=np.uint8)

    if thoracic_trunk_fat_labels_out:
        save_like(ff_img, thoracic_trunk_fat_labels, thoracic_trunk_fat_labels_out, dtype=np.uint8)

    if thoracic_trunk_fat_labels_csv_out:
        save_thoracic_trunk_fat_label_csv(thoracic_trunk_fat_labels_csv_out)

    if thoracic_trunk_fat_labels_ctbl_out:
        save_slicer_color_table(thoracic_trunk_fat_labels_ctbl_out)

    # Export pericardial fat as a standalone binary compartment in addition to the
    # combined thoracic multi-label image. In Thoracic_trunk_fat_labels.nii.gz,
    # label value 1 is PericardialFat.
    if pericardial_fat_mask_out:
        pericardial_fat_mask = thoracic_trunk_fat_labels == 1
        save_like(ff_img, pericardial_fat_mask, pericardial_fat_mask_out, dtype=np.uint8)

    if eroded_seg_out:
        save_like(seg_img, seg_eroded, eroded_seg_out, dtype=np.int16)


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
    ap.add_argument("--station", default="", help="Station name, typically Upper or Lower")
    ap.add_argument("--upper_imat_dilate_voxels", type=int, default=1)
    args = ap.parse_args()

    base = args.dir

    w_path, f_path, ff_path, seg_path = find_required_dixon_inputs(base)

    qc_dir = os.path.join(base, "eroded_mask_for_qc")
    os.makedirs(qc_dir, exist_ok=True)

    # Keep all CSV outputs from this module together so QC masks and tabular
    # analysis outputs are not mixed in the station folder.
    csv_dir = os.path.join(base, "fat compartment csvs")
    os.makedirs(csv_dir, exist_ok=True)

    eroded_seg_path = os.path.join(qc_dir, "Dixon_W_COMP_eroded_dseg.nii.gz")
    sat_mask_path = os.path.join(qc_dir, "SAT_mask.nii.gz")
    imat_mask_path = os.path.join(qc_dir, "IMAT_mask.nii.gz")
    vat_mask_path = os.path.join(qc_dir, "VAT_mask.nii.gz")
    cavity_mask_path = os.path.join(qc_dir, "Abdominal_cavity_mask.nii.gz")
    abdominal_muscles_mask_path = os.path.join(qc_dir, "Abdominal_muscles_mask.nii.gz")
    torso_fat_mask_path = os.path.join(qc_dir, "TS_torso_fat_for_VAT_mask.nii.gz")
    subcutaneous_fat_mask_path = os.path.join(qc_dir, "TS_subcutaneous_fat_for_SAT_mask.nii.gz")
    trunk_cavity_exclusion_mask_path = os.path.join(qc_dir, "TS_trunk_cavity_exclusion_mask.nii.gz")
    thoracic_trunk_fat_labels_path = os.path.join(qc_dir, "Thoracic_trunk_fat_labels.nii.gz")
    thoracic_trunk_fat_labels_csv_path = os.path.join(qc_dir, "Thoracic_trunk_fat_labels.csv")
    thoracic_trunk_fat_labels_ctbl_path = os.path.join(qc_dir, "Thoracic_trunk_fat_labels.ctbl")
    pericardial_fat_mask_path = os.path.join(qc_dir, "PericardialFat_mask.nii.gz")

    print("Building SAT, IMAT, VAT, abdominal cavity, and thoracic trunk fat QC masks")
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
        subcutaneous_fat_mask_out=subcutaneous_fat_mask_path,
        trunk_cavity_exclusion_mask_out=trunk_cavity_exclusion_mask_path,
        thoracic_trunk_fat_labels_out=thoracic_trunk_fat_labels_path,
        thoracic_trunk_fat_labels_csv_out=thoracic_trunk_fat_labels_csv_path,
        thoracic_trunk_fat_labels_ctbl_out=thoracic_trunk_fat_labels_ctbl_path,
        pericardial_fat_mask_out=pericardial_fat_mask_path,
        station_name=args.station,
        upper_imat_dilate_voxels=args.upper_imat_dilate_voxels
    )

    print("Computing SAT metrics")
    compute_binary_mask_metrics(sat_mask_path, ff_path, args.subject, args.day, "SAT", csv_dir)

    print("Computing IMAT metrics")
    compute_binary_mask_metrics(imat_mask_path, ff_path, args.subject, args.day, "IMAT", csv_dir)

    print("Computing VAT metrics")
    compute_binary_mask_metrics(vat_mask_path, ff_path, args.subject, args.day, "VAT", csv_dir)

    print("Computing fat compartment slice comparison")
    compute_fat_compartment_slice_comparison(
        {
            "SAT": sat_mask_path,
            "IMAT": imat_mask_path,
            "VAT": vat_mask_path,
        },
        ff_path,
        args.subject,
        args.day,
        csv_dir,
    )

    print("Computing fat compartment volumes restricted to VAT-containing slices")
    compute_fat_compartment_vat_slice_volume_comparison(
        {
            "SAT": sat_mask_path,
            "IMAT": imat_mask_path,
            "VAT": vat_mask_path,
        },
        ff_path,
        args.subject,
        args.day,
        csv_dir,
    )

    print("Computing thoracic trunk fat metrics")
    compute_label_map_metrics(
        thoracic_trunk_fat_labels_path,
        thoracic_trunk_fat_labels_csv_path,
        ff_path,
        args.subject,
        args.day,
        csv_dir,
    )

    print("Computing standalone pericardial fat metrics")
    compute_binary_mask_metrics(
        pericardial_fat_mask_path,
        ff_path,
        args.subject,
        args.day,
        "PericardialFat",
        csv_dir,
    )

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
        "--station", station_dir.name,
        "--upper_imat_dilate_voxels", str(cfg.get("upper_imat_dilate_voxels", 1)),
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
