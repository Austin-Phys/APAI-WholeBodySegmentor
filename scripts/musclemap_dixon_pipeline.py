#!/usr/bin/env python
import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage as ndi

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MUSCLEMAP_REPO = str(SCRIPT_DIR.parent / "MuscleMap")
MM_SEGMENT_REL = os.path.join("scripts", "mm_segment.py")


def load_label_map(csv_path: str) -> Dict[int, Tuple[str, str, str]]:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"label_id", "region", "anatomy", "side"}
    if not required.issubset(df.columns):
        raise ValueError(f"labels CSV must contain {sorted(required)}")
    df = df.dropna(subset=["label_id"]).copy()
    df["label_id"] = df["label_id"].astype(int)
    for c in ["region", "anatomy", "side"]:
        df[c] = df[c].astype(str).str.strip()
    return {int(r["label_id"]): (r["region"], r["anatomy"], r["side"]) for _, r in df.iterrows()}


def label_to_name(label_id: int, label_map: Dict[int, Tuple[str, str, str]]):
    return label_map.get(int(label_id), ("unknown", f"label_{label_id}", "unknown"))


def dseg_name_for(fn: str) -> str:
    low = fn.lower()
    if low.endswith(".nii.gz"):
        return fn[:-7] + "_dseg.nii.gz"
    if low.endswith(".nii"):
        return fn[:-4] + "_dseg.nii"
    return fn + "_dseg.nii.gz"


def run_mm_segment(mm_segment_path: str, input_path: str):
    """
    Run MuscleMap segmentation with the working directory set to the folder
    containing the input image.

    Newer MuscleMap/mm_segment versions save Dixon_W_COMP_dseg.nii.gz into the
    current working directory rather than beside the input file unless cwd is set.
    Setting cwd here keeps the segmentation output inside the station/session
    folder, e.g. Musclemap Data/Upper or Musclemap Data/Lower.
    """
    input_path_obj = Path(input_path)
    run_dir = input_path_obj.parent

    cmd = [sys.executable, mm_segment_path, "-i", str(input_path_obj)]
    env = os.environ.copy()
    # Anaconda's MKL numpy and pip-installed torch both ship libiomp5md.dll,
    # which aborts the OpenMP runtime on import ("OMP: Error #15") unless
    # duplicate loading is explicitly allowed.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    subprocess.run(cmd, check=True, cwd=str(run_dir), env=env)

    # Defensive fallback: if mm_segment still writes into the caller/code directory,
    # move the expected dseg file back beside the input.
    expected = run_dir / dseg_name_for(input_path_obj.name)
    if not expected.exists():
        stray = Path.cwd() / dseg_name_for(input_path_obj.name)
        if stray.exists():
            stray.replace(expected)


def summarize(vals: np.ndarray):
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def compute_metrics(map_path, seg_path, label_map, subject, day, map_name, out_dir):
    m_img = nib.load(map_path)
    s_img = nib.load(seg_path)
    m = m_img.get_fdata()
    s = np.rint(s_img.get_fdata()).astype(int)

    dx, dy, dz = nib.affines.voxel_sizes(m_img.affine)
    pixel_area_cm2 = float(dx * dy) / 100.0
    voxel_vol_ml = float(dx * dy * dz) / 1000.0

    vol_rows = []
    for lab in np.unique(s):
        if lab == 0:
            continue
        vals = m[s == lab]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        r, a, side = label_to_name(lab, label_map)
        stats = summarize(vals)
        vol_rows.append({
            "label_id": int(lab),
            "region": r,
            "anatomy": a,
            "side": side,
            "n_voxels": int(vals.size),
            "volume_ml": int(vals.size) * voxel_vol_ml,
            **stats
        })

    slice_rows = []
    zdim = s.shape[2]
    for z in range(zdim):
        sl_seg = s[:, :, z]
        labs = np.unique(sl_seg)
        labs = labs[labs != 0]
        if labs.size == 0:
            continue
        for lab in labs:
            vals = m[:, :, z][sl_seg == lab]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            r, a, side = label_to_name(lab, label_map)
            stats = summarize(vals)
            slice_rows.append({
                "slice_index": int(z),
                "label_id": int(lab),
                "region": r,
                "anatomy": a,
                "side": side,
                "n_voxels": int(vals.size),
                "CSA_cm2": int(vals.size) * pixel_area_cm2,
                **stats
            })

    tag = f"{subject}_{day}" if day else subject
    pd.DataFrame(vol_rows).to_csv(os.path.join(out_dir, f"{tag}_{map_name}_volume.csv"), index=False)
    pd.DataFrame(slice_rows).to_csv(os.path.join(out_dir, f"{tag}_{map_name}_slice.csv"), index=False)


def create_ff(w_path, f_path, out_path, signal_threshold=50.0):
    w = sitk.ReadImage(w_path, sitk.sitkFloat32)
    f = sitk.ReadImage(f_path, sitk.sitkFloat32)
    denom = sitk.Add(w, f)
    mask = sitk.Greater(denom, signal_threshold)
    ff = sitk.Divide(f, sitk.Add(denom, 1e-6))
    ff = sitk.Mask(ff, mask)
    ff = sitk.Clamp(ff, lowerBound=0.0, upperBound=1.0)
    sitk.WriteImage(ff, out_path)


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


def build_sat_imat_masks(
    water_path: str,
    fat_path: str,
    ff_path: str,
    muscle_seg_path: str,
    sat_mask_out: str,
    imat_mask_out: str,
    signal_threshold: float = 50.0,
    fat_ff_threshold_imat: float = 0.2,
    fat_ff_threshold_sat: float = 0.3,
    fascia_dilate_size: int = 3,
    fascia_close_size: int = 15,
    erode_voxels: int = 0,
    eroded_seg_out: str = None
):
    """
    Approximate fascia from the union of all muscle labels on each slice.
    SAT  = fat outside fascia but inside body
    IMAT = fat inside fascia but outside labeled muscle

    Uses separate FF thresholds for IMAT and SAT so IMAT can capture
    lower-fat partial-volume voxels while SAT remains more conservative.

    Compared with the earlier version, this uses:
    - modest dilation
    - binary closing
    - hole filling

    This usually gives a more anatomically plausible fascia-like envelope.
    """
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
    fat_candidate_imat = signal & np.isfinite(ff) & (ff >= fat_ff_threshold_imat)
    fat_candidate_sat = signal & np.isfinite(ff) & (ff >= fat_ff_threshold_sat)

    sat_mask = np.zeros_like(fat_candidate_sat, dtype=bool)
    imat_mask = np.zeros_like(fat_candidate_imat, dtype=bool)

    zdim = ff.shape[2]
    dilate_size = max(3, int(fascia_dilate_size))
    close_size = max(3, int(fascia_close_size))
    dilate_structure = np.ones((dilate_size, dilate_size), dtype=bool)
    close_structure = np.ones((close_size, close_size), dtype=bool)

    for z in range(zdim):
        sig = signal[:, :, z]
        fc_imat = fat_candidate_imat[:, :, z]
        fc_sat = fat_candidate_sat[:, :, z]
        muscle_union = seg[:, :, z] > 0
        muscle_union_eroded = muscle_mask_eroded[:, :, z]

        if not np.any(sig):
            continue

        body = ndi.binary_fill_holes(sig)
        body = _largest_two_components_2d(body)

        fc_imat = fc_imat & body
        fc_sat = fc_sat & body
        if not np.any(fc_imat) and not np.any(fc_sat):
            continue

        if np.any(muscle_union):
            fascia = ndi.binary_dilation(muscle_union, structure=dilate_structure)
            fascia = ndi.binary_closing(fascia, structure=close_structure)
            fascia = ndi.binary_fill_holes(fascia)
            fascia = fascia & body
        else:
            fascia = np.zeros_like(body, dtype=bool)

        imat_slice = fc_imat & fascia & (~muscle_union_eroded)
        sat_slice = fc_sat & body & (~fascia)

        sat_mask[:, :, z] = sat_slice
        imat_mask[:, :, z] = imat_slice

    save_like(ff_img, sat_mask, sat_mask_out, dtype=np.uint8)
    save_like(ff_img, imat_mask, imat_mask_out, dtype=np.uint8)
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
    pd.DataFrame(vol_rows).to_csv(os.path.join(out_dir, f"{tag}_{compartment_name}_volume.csv"), index=False)
    pd.DataFrame(slice_rows).to_csv(os.path.join(out_dir, f"{tag}_{compartment_name}_slice.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description="WholeBodySeg Dixon MuscleMap module: create FF map, segment Dixon W, and compute muscle FF metrics.")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--day", default="")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--musclemap_repo", default=DEFAULT_MUSCLEMAP_REPO, help="Path to MuscleMap repo")
    ap.add_argument("--skip_seg", action="store_true")
    ap.add_argument("--signal_threshold", type=float, default=50.0)
    ap.add_argument("--erode_voxels", type=int, default=1)
    args = ap.parse_args()

    base = args.dir
    label_map = load_label_map(os.path.join(args.code_dir, "musclemap_labels.csv"))
    mm_segment = os.path.join(args.musclemap_repo, MM_SEGMENT_REL)
    if not os.path.isfile(mm_segment):
        raise FileNotFoundError(f"mm_segment.py not found at: {mm_segment}")

    files = [f for f in os.listdir(base) if f.lower().endswith(".nii") or f.lower().endswith(".nii.gz")]
    w_files = [f for f in files if "w_comp" in f.lower() and "_dseg" not in f.lower()]
    f_files = [f for f in files if "f_comp" in f.lower() and "_dseg" not in f.lower()]
    if not w_files or not f_files:
        raise FileNotFoundError("Could not find Dixon W/F NIfTI files in the target folder.")

    w_fn = w_files[0]
    f_fn = f_files[0]
    w_path = os.path.join(base, w_fn)
    f_path = os.path.join(base, f_fn)
    seg_path = os.path.join(base, dseg_name_for(w_fn))

    if not args.skip_seg and not os.path.exists(seg_path):
        print("Running segmentation on Dixon W")
        run_mm_segment(mm_segment, w_path)

    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Missing segmentation: {seg_path}")

    ff_path = os.path.join(base, "Dixon_FF_map.nii.gz")
    print("Creating FF map")
    create_ff(w_path, f_path, ff_path, signal_threshold=args.signal_threshold)

    qc_dir = os.path.join(base, "eroded_mask_for_qc")
    os.makedirs(qc_dir, exist_ok=True)

    eroded_seg_path = os.path.join(qc_dir, "Dixon_W_COMP_eroded_dseg.nii.gz")

    print("Building eroded muscle segmentation")
    seg_img = nib.load(seg_path)
    seg = np.rint(seg_img.get_fdata()).astype(np.int32)
    seg_eroded = erode_segmentation_per_label(seg, iterations=args.erode_voxels)
    save_like(seg_img, seg_eroded, eroded_seg_path, dtype=np.int16)

    print("Computing muscle FF metrics from eroded segmentation")
    compute_metrics(ff_path, eroded_seg_path, label_map, args.subject, args.day, "Dixon_FF", base)

    print("Dixon MuscleMap module DONE")

def run_musclemap_dixon(station_dir, cfg):
    from pathlib import Path
    import sys

    station_dir = Path(station_dir)

    subject = cfg.get("current_subject", "")
    session = cfg.get("current_session", "")

    argv_old = sys.argv[:]

    sys.argv = [
        "musclemap_dixon_pipeline.py",
        "--subject", subject,
        "--day", session,
        "--dir", str(station_dir),
        "--code_dir", cfg.get("code_dir", str(Path.cwd())),
        "--musclemap_repo", cfg.get("musclemap_repo", str(Path.cwd() / "MuscleMap")),
        "--signal_threshold", str(cfg.get("signal_threshold", 50.0)),
        "--erode_voxels", str(cfg.get("erode_voxels", 1)),
    ]

    if cfg.get("skip_musclemap_dixon_seg", False):
        sys.argv += ["--skip_seg"]

    try:
        main()
    finally:
        sys.argv = argv_old

if __name__ == "__main__":
    main()
