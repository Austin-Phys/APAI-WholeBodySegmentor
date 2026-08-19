#!/usr/bin/env python
"""
musclemap_t2_448_pipeline.py

WholeBodySeg T2-448 MuscleMap module.

This is a T2-only version derived from the prior combined W/F + Grappa pipeline.
It processes only T2 448 thigh/calf images and does not process Dixon W/F images.

For one station folder, it:
1. Detects or uses configured T2-448 thigh/calf images
2. Runs MuscleMap segmentation with mm_segment.py
3. Computes native per-label volume and slice metrics

Expected typical files:
    T2_448_Thigh.nii.gz
    T2_448_Calf.nii.gz

But detection is intentionally flexible because some data include GRAPPA in the name
and some do not.
"""

import os
import sys
import argparse
import subprocess
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import nibabel as nib


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MUSCLEMAP_REPO = str(SCRIPT_DIR.parent / "MuscleMap")
MM_SEGMENT_REL = os.path.join("scripts", "mm_segment.py")


# ----------------------------
# LABELS
# ----------------------------
def load_label_map(csv_path: str) -> Dict[int, Tuple[str, str, str]]:
    """Load label_id -> (region, anatomy, side) from musclemap_labels.csv."""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"label_id", "region", "anatomy", "side"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"musclemap_labels.csv header must include: {sorted(required)}\n"
            f"Found: {list(df.columns)}\n"
            f"Fix the first line to: label_id,region,anatomy,side"
        )

    df = df.dropna(subset=["label_id"]).copy()
    df["label_id"] = df["label_id"].astype(int)

    for c in ["region", "anatomy", "side"]:
        df[c] = df[c].astype(str).str.strip()

    return {int(r["label_id"]): (r["region"], r["anatomy"], r["side"]) for _, r in df.iterrows()}


def label_to_name(label_id: int, label_map: Dict[int, Tuple[str, str, str]]) -> Tuple[str, str, str]:
    return label_map.get(int(label_id), ("unknown", f"label_{int(label_id)}", "unknown"))


# ----------------------------
# FILE DETECTION
# ----------------------------
def list_nii_files(base_dir: str) -> List[str]:
    files = []
    for fn in os.listdir(base_dir):
        low = fn.lower()
        if (low.endswith(".nii") or low.endswith(".nii.gz")) and ("_dseg" not in low):
            files.append(fn)
    return files


def score_match(filename: str, must: List[str], must_not: List[str], prefer: Optional[List[str]] = None) -> int:
    low = filename.lower()

    for m in must:
        if m.lower() not in low:
            return -10_000

    for b in must_not:
        if b.lower() in low:
            return -10_000

    score = 0
    prefer = prefer or ["448", "grappa", "tse", "ax", "t2", "filt", "comp"]
    for p in prefer:
        if p.lower() in low:
            score += 2

    # Slight penalty for very long names, often duplicates or converted intermediates.
    score -= len(filename) // 80
    return score


def find_best_t2_448(base_dir: str, region: str) -> Optional[str]:
    """
    Flexible T2 448 detector.

    We require T2 and the anatomical region name, but we do not require 'grappa'
    because some studies have the 448 T2 acquisition without GRAPPA in the filename.
    """
    candidates = list_nii_files(base_dir)

    must = ["t2", region]
    must_not = ["dixon", "w_comp", "f_comp", "ff", "fat", "fatfraction", "t2map", "t2s", "t2*", "moco"]
    prefer = ["448", "grappa", "tse", "ax", region]

    scored = [(score_match(fn, must, must_not, prefer=prefer), fn) for fn in candidates]
    scored = [x for x in scored if x[0] > -1000]
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1] if scored else None


def find_combined_t2_448(base_dir: str) -> Optional[str]:
    """
    Fallback detector for protocols that acquire thigh+calf in a single combined
    T2 448 series (e.g. ProtocolName "t2_tse_tra_448", no separate thigh/calf series
    and no "grappa" token at all). Used only when no region-specific thigh/calf file
    is found, so studies with real split series keep using those.
    """
    candidates = list_nii_files(base_dir)

    must = ["t2"]
    must_not = ["dixon", "w_comp", "f_comp", "ff", "fat", "fatfraction", "t2map", "t2s", "t2*", "moco", "sag", "spine"]
    prefer = ["448", "combined", "tse", "tra", "grappa"]

    scored = [(score_match(fn, must, must_not, prefer=prefer), fn) for fn in candidates]
    scored = [x for x in scored if x[0] > -1000]
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1] if scored else None


def resolve_t2_image(base_dir: str, region: str, explicit_name: str = "") -> Optional[str]:
    """
    Resolve a T2 image for the current station folder.

    If an explicit config filename is provided, only use it when it actually
    exists in this station folder. If it does not exist, fall back to flexible
    auto-detection. This prevents Upper from failing just because the config
    names a Lower/Thigh T2 file.
    """
    explicit_name = (explicit_name or "").strip()

    if explicit_name:
        explicit_path = Path(base_dir) / explicit_name
        if explicit_path.is_file():
            return explicit_name

        print(
            f"    NOTE: configured {region} T2 file not found in this station, "
            f"will try auto-detection: {explicit_name}"
        )

    return find_best_t2_448(base_dir, region)


def dseg_name_for(map_filename: str) -> str:
    low = map_filename.lower()
    if low.endswith(".nii.gz"):
        return map_filename[:-7] + "_dseg.nii.gz"
    if low.endswith(".nii"):
        return map_filename[:-4] + "_dseg.nii"
    return map_filename + "_dseg.nii.gz"


def musclemap_mask_name_for(map_filename: str) -> str:
    """Return stable MuscleMap mask filename while preserving the T2 image basename."""
    low = map_filename.lower()
    if low.endswith(".nii.gz"):
        return map_filename[:-7] + "_MuscleMap_Mask.nii.gz"
    if low.endswith(".nii"):
        return map_filename[:-4] + "_MuscleMap_Mask.nii"
    return map_filename + "_MuscleMap_Mask.nii.gz"


def should_reverse_slice_index(map_name: str) -> bool:
    """
    Reverse reported slice numbering for older GRAPPA-style outputs so CSV slice labels
    match the expected anatomical ordering. This changes only the reported slice index,
    not metric extraction.
    """
    low = map_name.lower()
    return "grappa" in low


# ----------------------------
# MUSCLEMAP SEGMENT RUNNER
# ----------------------------
def run_mm_segment(mm_segment_path: str, input_path: str, model_version: str = "1.4"):
    """Run MuscleMap segmentation in the folder containing the input image."""
    input_path_obj = Path(input_path)
    run_dir = input_path_obj.parent
    input_name = input_path_obj.name

    cmd_str = (
        f'cd /d "{run_dir}" && '
        f'"{sys.executable}" "{mm_segment_path}" -i "{input_name}" '
        f'--model_version "{model_version}"'
    )

    print("    Running:", cmd_str)
    env = os.environ.copy()
    # Anaconda's MKL numpy and pip-installed torch both ship libiomp5md.dll,
    # which aborts the OpenMP runtime on import ("OMP: Error #15") unless
    # duplicate loading is explicitly allowed.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    subprocess.run(cmd_str, check=True, shell=True, env=env)

    # MuscleMap writes <input>_dseg.nii.gz. Normalize that output to the
    # stable WholeBodySeg naming convention.
    raw_expected = run_dir / dseg_name_for(input_name)
    if not raw_expected.exists():
        stray = Path.cwd() / dseg_name_for(input_name)
        if stray.exists():
            stray.replace(raw_expected)

    final_mask = run_dir / musclemap_mask_name_for(input_name)
    if raw_expected.exists() and raw_expected != final_mask:
        if final_mask.exists():
            final_mask.unlink()
        raw_expected.replace(final_mask)


# ----------------------------
# CSV HELPERS
# ----------------------------
def _excel_safe_to_csv(df: pd.DataFrame, out_path: str):
    for c in df.columns:
        if c in {"region", "anatomy", "side", "map", "map_file", "seg_file", "subject", "day"}:
            continue
        if df[c].dtype == object:
            s = df[c].astype(str).str.replace("'", "", regex=False)
            df[c] = pd.to_numeric(s, errors="ignore")

    df.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.6f")


def _safe_col(text: str) -> str:
    """Create a stable wide-CSV column token."""
    s = str(text).strip().lower()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def build_t2_448_station_summary(base_dir: str, subject: str, day: str):
    """
    Build consolidated T2-448 outputs for this station.

    Reads per-map native volume metrics created by compute_metrics() and writes:
      1. <subject>_T2_448_native_volume_metrics_ALL.csv
      2. <subject>_T2_448_native_volume_metrics_WIDE.csv

    The WIDE file is intended for easy downstream merging with Dixon/fat/TotalSeg
    summary outputs. T2 mean columns are named mean_T2, not mean_FF.
    """
    tag = f"{subject}_{day}" if day else subject
    base = Path(base_dir)

    pattern = f"{tag}_T2_448_*_native_volume_metrics.csv"
    files = sorted(base.glob(pattern))

    # Avoid recursively reading our own combined output if the script is rerun.
    files = [
        p for p in files
        if not p.name.endswith("_ALL.csv") and not p.name.endswith("_WIDE.csv")
    ]

    if not files:
        print("    NOTE: no T2-448 native volume metric CSVs found to combine.")
        return

    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
            df["source_csv"] = p.name
            frames.append(df)
        except Exception as e:
            print(f"    WARNING: could not read T2 metrics CSV {p.name}: {e}")

    if not frames:
        print("    NOTE: no readable T2-448 native volume metric CSVs found to combine.")
        return

    long_df = pd.concat(frames, ignore_index=True)

    # Explicitly label the meaning of generic stats for T2 maps.
    rename = {}
    if "mean" in long_df.columns:
        rename["mean"] = "mean_T2"
    if "median" in long_df.columns:
        rename["median"] = "median_T2"
    if "std" in long_df.columns:
        rename["std"] = "std_T2"
    if "p05" in long_df.columns:
        rename["p05"] = "p05_T2"
    if "p95" in long_df.columns:
        rename["p95"] = "p95_T2"
    if "min" in long_df.columns:
        rename["min"] = "min_T2"
    if "max" in long_df.columns:
        rename["max"] = "max_T2"

    long_df = long_df.rename(columns=rename)

    out_long = base / f"{tag}_T2_448_native_volume_metrics_ALL.csv"
    _excel_safe_to_csv(long_df, str(out_long))
    print("    saved:", out_long.name)

    # Build one-row wide output.
    id_cols = {
        "subject": subject,
        "day": day,
    }
    wide = dict(id_cols)

    metric_cols = [
        "n_voxels",
        "volume_ml",
        "mean_T2",
        "median_T2",
        "std_T2",
        "p05_T2",
        "p95_T2",
        "min_T2",
        "max_T2",
    ]

    for _, row in long_df.iterrows():
        map_token = _safe_col(row.get("map", "T2_448"))
        region_token = _safe_col(row.get("region", "unknown"))
        anatomy_token = _safe_col(row.get("anatomy", f"label_{row.get('label_id', 'unknown')}"))
        side_token = _safe_col(row.get("side", "unknown"))

        # Skip useless side token if it is blank/unknown/none.
        parts = ["MM_T2", map_token, region_token, anatomy_token]
        if side_token not in {"", "unknown", "none", "nan"}:
            parts.append(side_token)
        base_col = "_".join(parts)

        for metric in metric_cols:
            if metric in row.index and pd.notna(row[metric]):
                wide[f"{base_col}_{metric}"] = row[metric]

    out_wide = base / f"{tag}_T2_448_native_volume_metrics_WIDE.csv"
    wide_df = pd.DataFrame([wide])
    _excel_safe_to_csv(wide_df, str(out_wide))
    print("    saved:", out_wide.name)


# ----------------------------
# METRICS
# ----------------------------
def summarize(vals: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p05": float(np.quantile(vals, 0.05)),
        "p95": float(np.quantile(vals, 0.95)),
        "std": float(np.std(vals, ddof=0)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def compute_metrics(
    map_path: str,
    seg_path: str,
    label_map: Dict[int, Tuple[str, str, str]],
    subject: str,
    day: str,
    map_name: str,
    out_dir: str,
):
    m_img = nib.load(map_path)
    s_img = nib.load(seg_path)

    if m_img.shape != s_img.shape:
        raise ValueError(
            f"Grid mismatch in native T2-448 pipeline:\n"
            f"  MAP: {os.path.basename(map_path)} shape={m_img.shape}\n"
            f"  SEG: {os.path.basename(seg_path)} shape={s_img.shape}\n"
            f"Fix by re-running segmentation on the same map."
        )

    m = m_img.get_fdata(dtype=np.float32)
    seg = np.rint(s_img.get_fdata()).astype(np.int32)

    dx, dy, dz = nib.affines.voxel_sizes(m_img.affine)
    pixel_area_cm2 = float(dx * dy) / 100.0
    voxel_vol_ml = float(dx * dy * dz) / 1000.0

    labels = np.unique(seg)
    labels = labels[labels != 0]
    if labels.size == 0:
        print("    WARNING: segmentation is empty (all zeros). Skipping metrics.")
        return

    reverse_slice_index = should_reverse_slice_index(map_name)

    vol_rows = []
    for lab in labels:
        lab = int(lab)
        mask = seg == lab
        vals = m[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        region, anatomy, side = label_to_name(lab, label_map)
        stats = summarize(vals)
        nvox = int(vals.size)

        vol_rows.append({
            "subject": subject,
            "day": day,
            "map": map_name,
            "label_id": lab,
            "region": region,
            "anatomy": anatomy,
            "side": side,
            "n_voxels": nvox,
            "volume_ml": nvox * voxel_vol_ml,
            **stats,
            "map_file": os.path.basename(map_path),
            "seg_file": os.path.basename(seg_path),
        })

    df_vol = pd.DataFrame(vol_rows)
    if df_vol.empty:
        print("    WARNING: no valid labeled voxels. Skipping metrics.")
        return
    df_vol = df_vol.sort_values(["region", "anatomy", "side", "label_id"])

    slice_rows = []
    zdim = seg.shape[2]
    for z in range(zdim):
        sl_seg = seg[:, :, z]
        sl_m = m[:, :, z]

        labs = np.unique(sl_seg)
        labs = labs[labs != 0]
        if labs.size == 0:
            continue

        reported_z = zdim - 1 - z if reverse_slice_index else z

        for lab in labs:
            lab = int(lab)
            mask = sl_seg == lab
            vals = sl_m[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            region, anatomy, side = label_to_name(lab, label_map)
            stats = summarize(vals)
            nvox = int(vals.size)

            slice_rows.append({
                "subject": subject,
                "day": day,
                "map": map_name,
                "slice_index": int(reported_z),
                "label_id": lab,
                "region": region,
                "anatomy": anatomy,
                "side": side,
                "n_voxels": nvox,
                "CSA_cm2": nvox * pixel_area_cm2,
                **stats,
                "map_file": os.path.basename(map_path),
                "seg_file": os.path.basename(seg_path),
            })

    df_slice = pd.DataFrame(slice_rows)
    if not df_slice.empty:
        df_slice = df_slice.sort_values(["slice_index", "region", "anatomy", "side", "label_id"])

    tag = f"{subject}_{day}" if day else subject
    out_vol = os.path.join(out_dir, f"{tag}_{map_name}_native_volume_metrics.csv")
    out_slice = os.path.join(out_dir, f"{tag}_{map_name}_native_slice_metrics.csv")

    _excel_safe_to_csv(df_vol, out_vol)
    print("    saved:", os.path.basename(out_vol))

    if not df_slice.empty:
        _excel_safe_to_csv(df_slice, out_slice)
        print("    saved:", os.path.basename(out_slice))
    else:
        print("    NOTE: slice-wise metrics empty.")

    unknown = df_vol[df_vol["region"] == "unknown"]["label_id"].unique().tolist()
    if unknown:
        print(f"    NOTE: unknown IDs present (missing from CSV): {unknown}")


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="WholeBodySeg T2-448 MuscleMap module: segment thigh/calf T2 448 images and compute native metrics."
    )
    ap.add_argument("--subject", required=True, help="Subject ID, e.g. P008")
    ap.add_argument("--day", default="", help="Optional session/day/visit")
    ap.add_argument("--dir", required=True, help="Station directory containing T2 images")
    ap.add_argument("--code_dir", required=True, help="Directory containing musclemap_labels.csv")
    ap.add_argument("--musclemap_repo", default=DEFAULT_MUSCLEMAP_REPO, help="Path to MuscleMap repo")
    ap.add_argument("--skip_seg", action="store_true", help="Skip segmentation step")
    ap.add_argument("--skip_metrics", action="store_true", help="Skip native metrics step")
    ap.add_argument("--model_version", default="1.4", help="MuscleMap whole-body model version (default: 1.4)")
    ap.add_argument("--thigh", default="", help="Optional explicit thigh T2 image filename")
    ap.add_argument("--calf", default="", help="Optional explicit calf T2 image filename")
    args = ap.parse_args()

    base_dir = args.dir
    subject = args.subject
    day = args.day.strip()

    labels_csv = os.path.join(args.code_dir, "musclemap_labels.csv")
    if not os.path.isfile(labels_csv):
        raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")
    label_map = load_label_map(labels_csv)

    mm_segment_path = os.path.join(args.musclemap_repo, MM_SEGMENT_REL)
    if not os.path.isfile(mm_segment_path):
        raise FileNotFoundError(f"mm_segment.py not found at: {mm_segment_path}")

    if not os.path.isdir(base_dir):
        raise NotADirectoryError(base_dir)

    thigh = resolve_t2_image(base_dir, "thigh", args.thigh)
    calf = resolve_t2_image(base_dir, "calf", args.calf)

    detected = {
        "T2_448_Thigh": thigh,
        "T2_448_Calf": calf,
    }

    # Some protocols acquire thigh+calf as a single combined series instead of
    # separate thigh/calf series (no region-specific token in the filename at all).
    # Only fall back to it when neither region-specific file was found, so studies
    # with real split series are unaffected.
    if not thigh and not calf:
        combined = find_combined_t2_448(base_dir)
        if combined:
            detected = {"T2_448_Combined": combined}

    print("\n=== WholeBodySeg MuscleMap T2-448 Pipeline ===")
    print("Subject:", subject, " Day:", (day if day else "(none)"))
    print("Dir:", base_dir)
    print("Python:", sys.executable)
    print("mm_segment:", mm_segment_path)
    print("labels:", labels_csv)
    print("\nDetected files:")
    for k, v in detected.items():
        print(f"  {k}: {v if v else 'NOT FOUND'}")

    if not any(detected.values()):
        print("\nNOTE: No T2-448 thigh/calf images detected in this station folder. Nothing to do.")
        return

    if not args.skip_seg:
        print("\n--- STEP 1: T2-448 Segmentation ---")
        for map_name, fn in detected.items():
            if not fn:
                continue
            map_path = os.path.join(base_dir, fn)
            seg_fn = musclemap_mask_name_for(fn)
            seg_path = os.path.join(base_dir, seg_fn)

            if os.path.isfile(seg_path):
                print(f"[SKIP] {map_name}: MuscleMap mask exists ({seg_fn})")
                continue

            print(f"[SEG ] {map_name}: {fn}")
            run_mm_segment(mm_segment_path, map_path, model_version=args.model_version)

    if not args.skip_metrics:
        print("\n--- STEP 2: T2-448 Native Metrics ---")
        for map_name, fn in detected.items():
            if not fn:
                continue
            map_path = os.path.join(base_dir, fn)
            seg_fn = musclemap_mask_name_for(fn)
            seg_path = os.path.join(base_dir, seg_fn)

            if not os.path.isfile(seg_path):
                print(f"!! Missing MuscleMap mask for {map_name}: {seg_fn} (segmentation step failed or was skipped)")
                continue

            print(f"[MET ] {map_name}: {fn}")
            compute_metrics(
                map_path=map_path,
                seg_path=seg_path,
                label_map=label_map,
                subject=subject,
                day=day,
                map_name=map_name,
                out_dir=base_dir,
            )

        print("\n--- STEP 3: T2-448 Combined Summary CSVs ---")
        build_t2_448_station_summary(base_dir, subject, day)

    print("\n=== T2-448 DONE ===")


def run_musclemap_t2_448(station_dir, cfg):
    """WholeBodySeg wrapper entry point.

    T2-448/GRAPPA files are session-level outputs in the ASHA WholeBodySeg layout,
    not station-level outputs. The master pipeline calls this function once per
    station, so this wrapper redirects processing to the parent session folder by
    default and prevents duplicate reruns from Upper + Lower calls.
    """
    station_dir = Path(station_dir)
    subject = cfg.get("current_subject", "")
    session = cfg.get("current_session", "")

    # T2 files live one level above Upper/Lower:
    #   .../P008/Musclemap Data/T2_448_Thigh.nii.gz
    # not:
    #   .../P008/Musclemap Data/Upper/
    #   .../P008/Musclemap Data/Lower/
    t2_dir = station_dir.parent if station_dir.name.lower() in {"upper", "lower"} else station_dir

    # Avoid running the same session-level T2 step twice when the master loops
    # over both Upper and Lower stations.
    cache_key = f"_t2_448_done__{str(t2_dir.resolve())}"
    if cfg.get(cache_key, False):
        print(f"Skipping T2-448; already processed session folder: {t2_dir}")
        return
    cfg[cache_key] = True

    t2_images = cfg.get("t2_images", {}) or {}
    thigh_name = t2_images.get("Thigh", "")
    calf_name = t2_images.get("Calf", "")

    argv_old = sys.argv[:]
    sys.argv = [
        "musclemap_t2_448_pipeline.py",
        "--subject", subject,
        "--day", session,
        "--dir", str(t2_dir),
        "--code_dir", cfg.get("code_dir", str(Path.cwd())),
        "--musclemap_repo", cfg.get("musclemap_repo", str(Path.cwd() / "MuscleMap")),
        "--model_version", str(cfg.get("musclemap_model_version", "1.4")),
    ]

    if thigh_name:
        sys.argv += ["--thigh", thigh_name]

    if calf_name:
        sys.argv += ["--calf", calf_name]

    if cfg.get("skip_musclemap_t2_448_seg", False):
        sys.argv += ["--skip_seg"]

    if cfg.get("skip_musclemap_t2_448_metrics", False):
        sys.argv += ["--skip_metrics"]

    try:
        main()
    finally:
        sys.argv = argv_old


if __name__ == "__main__":
    main()
