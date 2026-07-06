"""
dicom_to_nifti_pipeline.py

Robust DICOM -> NIfTI (.nii.gz) conversion pipeline using dcm2niix on Windows.

Key behaviors (v18):
- Uses DICOM header text (ProtocolName/SeriesDescription) to select the correct SeriesInstanceUID.
- Converts ONLY selected series (RULES) into canonical filenames under each session's "Musclemap Data" folder.
- Separates Dixon water/fat outputs by acquisition station using DICOM headers:
  * ProtocolName containing TrunkRegion -> Upper
  * ProtocolName containing LowerExtrm -> Lower
- Converts each selected series into its own temp output folder to prevent mis-renaming across multiple outputs.
- Automatically collapses 4D NIfTI -> 3D (needed for MuscleMap segmentation) WITHOUT creating *_ORIG backups (as requested).
- Adds hard filters for Dixon_W_COMP / Dixon_F_COMP to avoid accidentally selecting T1Map DIXON sequences.
  * Excludes tokens like "t1map", "moco", "ff", "fatfraction", "grappa", "cartesian", etc.
  * Requires water-ish / fat-ish tokens.

How to test HC002 D1 only:
- Set TEST_SESSION to that session path and FORCE_RECONVERT=True.

If Dixon W/F still don't match, set PRINT_SERIES_INVENTORY=True for one run and tune MUST_CONTAIN_ANY
to the exact scanner wording for the water/fat recon series.
"""

import os
import re
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def collect_series_files(dicom_root: Path, target_uid: str) -> List[str]:
    """
    Collect all DICOM files under dicom_root whose SeriesInstanceUID matches target_uid.
    This is used as a fallback when the initial index stores only a representative file.
    """
    files = []
    for p in dicom_root.rglob("*"):
        if not p.is_file():
            continue
        # quick extension filter: accept .dcm or no extension; ignore obvious non-dicom
        if p.suffix.lower() in (".nii", ".gz", ".json", ".csv", ".txt", ".png", ".jpg", ".jpeg"):
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception:
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
        if uid == target_uid:
            files.append(str(p))
    return files

import pandas as pd
import pydicom
import nibabel as nib
import numpy as np


# =========================
# PORTABLE CONFIG HELPERS
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR

SUBJECT_FILTER = None
SESSION_FILTER = None
HAS_VISITS = True


def load_config(config_path: str) -> dict:
    """Load a study config JSON. Relative paths are resolved from the project folder."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def resolve_project_path(path_value: str) -> Path:
    """Resolve absolute paths as-is; relative paths are relative to the portable project folder."""
    path_value = Path(path_value)
    if path_value.is_absolute():
        return path_value
    return PROJECT_DIR / path_value


# =========================
# DEFAULT CONFIG
# =========================

# Defaults preserve your original PAD workflow if --config is not supplied.
DATA_ROOT = Path(r"C:\Users\kadav\OneDrive\Desktop\Local UTA Folder\PAD Project\Data")
DCM2NIIX_EXE = Path(r"C:\dcm2niix_win\dcm2niix.exe")
OUTPUT_FOLDER_NAME = "Musclemap Data"

# DICOM staging folder pattern (your current exports look like "... - DICOM")
DICOM_STAGING_PATTERN = re.compile(r".*\s-\sDICOM$", re.IGNORECASE)

# dcm2niix output naming inside per-series temp output folders (we'll rename to canonical)
DCM2NIIX_NAME_TEMPLATE = r"%t_%p_%s"

# 4D -> 3D strategy for NIfTI (required for mm_segment when Dixon comes out 4D)
# - "first": take volume 0
# - "mean": average across volume axis
COLLAPSE_4D_STRATEGY = "first"  # "first" or "mean"

# Apply 4D->3D collapse to which outputs?
# - "all": collapse any 4D output
# - "seg_only": only collapse Dixon_W_COMP and Dixon_F_COMP
COLLAPSE_SCOPE = "all"

# IMPORTANT: v3 does NOT create *_ORIG backups when collapsing 4D -> 3D (as requested).
KEEP_ORIG_BACKUP = False

# No preferred quantitative map series are used in this portable version.
# This script now extracts only:
#   - Dixon_W_COMP_Upper
#   - Dixon_F_COMP_Upper
#   - Dixon_W_COMP_Lower
#   - Dixon_F_COMP_Lower
#   - T2_Grappa_Thigh
#   - T2_Grappa_Calf
PREFERRED_SERIES_NUMBERS = {}

# Keyword rules: output_name -> list of keyword-groups (ANY group matches; ALL keywords in group must appear)
# NOTE: Dixon rules intentionally pair station + component wording so upper/lower W/F are written separately.
# Station is determined from ProtocolName:
#   - TrunkRegion  -> Upper
#   - LowerExtrm   -> Lower
# Component is determined primarily from SeriesDescription:
#   - W_COMP / water -> W
#   - F_COMP / fat   -> F
RULES: Dict[str, List[List[str]]] = {
    "Dixon_W_COMP_Upper": [
        ["trunkregion", "w comp"],
        ["trunkregion", "composed w comp"],
        ["trunkregion", "water"],
    ],
    "Dixon_F_COMP_Upper": [
        ["trunkregion", "f comp"],
        ["trunkregion", "composed f comp"],
        ["trunkregion", "fat"],
    ],
    "Dixon_W_COMP_Lower": [
        ["lowerextrm", "w comp"],
        ["lowerextrm", "composed w comp"],
        ["lowerextrm", "water"],
    ],
    "Dixon_F_COMP_Lower": [
        ["lowerextrm", "f comp"],
        ["lowerextrm", "composed f comp"],
        ["lowerextrm", "fat"],
    ],
    "T2_Grappa_Thigh": [
        ["t2", "grappa", "thigh"],
        ["grappa", "thigh"],
        ["t2", "grappa", "upper"],
        ["grappa", "upper"],
        ["t2", "448"],
    ],
}

# Hard filters for Dixon to avoid selecting T1/T2 map DIXON-derived sequences.
MUST_NOT_CONTAIN = {
    "Dixon_W_COMP_Upper": [],
    "Dixon_F_COMP_Upper": [],
    "Dixon_W_COMP_Lower": [],
    "Dixon_F_COMP_Lower": [],
    "T2_Grappa_Thigh": [],
}

# Require at least one of these strong tokens for W/F.
# If your scanner uses different wording, update these lists to match inventory output.
MUST_CONTAIN_ANY = {
    "Dixon_W_COMP_Upper": ["w comp", "water"],
    "Dixon_F_COMP_Upper": ["f comp", "fat"],
    "Dixon_W_COMP_Lower": ["w comp", "water"],
    "Dixon_F_COMP_Lower": ["f comp", "fat"],
    "T2_Grappa_Thigh": [],
}

# Debug options
PRINT_SERIES_INVENTORY = False  # print all series headers found per session (use to tune RULES)
FORCE_RECONVERT = False          # reconvert even if canonical outputs exist
TEST_SESSION = None

# =========================
# TEXT NORMALIZATION / MATCHING
# =========================

def normalize_header_text(protocol_name: Optional[str], series_desc: Optional[str], patient_position: Optional[str] = None) -> str:
    """
    Normalize header text for matching:
    - lowercases
    - converts '*' -> 'star'
    - strips to alnum tokens
    """
    s = f"{protocol_name or ''} {series_desc or ''} {patient_position or ''}".lower()
    s = s.replace("*", "star")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =========================
# DICOM HEADER READING
# =========================

def read_header(path: Path):
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


# =========================
# NIFTI HELPERS
# =========================

def nifti_shape_zooms(path: Path) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    img = nib.load(str(path))
    return tuple(int(x) for x in img.shape), tuple(float(z) for z in img.header.get_zooms())


def collapse_4d_to_3d_inplace(path: Path, strategy: str) -> Tuple[bool, str, str]:
    """
    If NIfTI is 4D, overwrite with a 3D volume (t=0 or mean), without backups unless KEEP_ORIG_BACKUP=True.

    Returns (changed, original_shape_str, new_shape_str)

    Notes:
    - Downstream SimpleITK registration/resampling is 3D; 4D outputs will break it.
    - We write float32 to keep things consistent and avoid int16/4D limitations.
    """
    img = nib.load(str(path))
    data = img.get_fdata()
    orig_shape = str(tuple(int(x) for x in data.shape))

    if data.ndim != 4:
        return False, orig_shape, orig_shape

    if strategy.lower() == "mean":
        data3 = np.mean(data, axis=3)
    else:
        data3 = data[..., 0]

    new_shape = str(tuple(int(x) for x in data3.shape))

    if KEEP_ORIG_BACKUP:
        bak = path.with_name(path.name.replace(".nii.gz", "_ORIG.nii.gz"))
        if not bak.exists():
            shutil.copy2(path, bak)

    data3 = np.asarray(data3, dtype=np.float32)
    out = nib.Nifti1Image(data3, img.affine, img.header)
    out.header.set_data_dtype(np.float32)

    # Ensure zooms are 3D (some 4D headers can confuse downstream tools)
    try:
        z = img.header.get_zooms()
        if len(z) >= 3:
            out.header.set_zooms(z[:3])
    except Exception:
        pass

    nib.save(out, str(path))
    return True, orig_shape, new_shape


# =========================
# SERIES INVENTORY / SELECTION
# =========================

def build_series_index(staging_dir: Path):
    series = defaultdict(lambda: {
        "SeriesNumber": None,
        "SeriesDescription": None,
        "ProtocolName": None,
        "ImageType": None,
        "PulseSequenceName": None,
        "PatientPosition": None,
        "ImageType": None,
        "Modality": None,
        "count": 0,
        "example": None,
        "norm_text": None,
    })

    for root, _, files in os.walk(staging_dir):
        for fn in files:
            p = Path(root) / fn
            ds = read_header(p)
            if ds is None:
                continue

            uid = getattr(ds, "SeriesInstanceUID", None)
            if not uid:
                continue

            rec = series[uid]
            rec["count"] += 1
            rec["SeriesNumber"] = rec["SeriesNumber"] or getattr(ds, "SeriesNumber", None)
            rec["SeriesDescription"] = rec["SeriesDescription"] or getattr(ds, "SeriesDescription", None)
            rec["ImageType"] = rec.get("ImageType") or getattr(ds, "ImageType", None)
            rec["PulseSequenceName"] = rec.get("PulseSequenceName") or getattr(ds, "PulseSequenceName", None)
            rec["ProtocolName"] = rec["ProtocolName"] or getattr(ds, "ProtocolName", None)
            rec["PatientPosition"] = rec["PatientPosition"] or getattr(ds, "PatientPosition", None)
            rec["ImageType"] = rec.get("ImageType") or getattr(ds, "ImageType", None)
            rec["PulseSequenceName"] = rec.get("PulseSequenceName") or getattr(ds, "PulseSequenceName", None)
            rec["ImageType"] = rec["ImageType"] or getattr(ds, "ImageType", None)
            rec["Modality"] = rec["Modality"] or getattr(ds, "Modality", None)
            rec["example"] = rec["example"] or p

    for uid, rec in series.items():
        rec["norm_text"] = normalize_header_text(rec.get("ProtocolName"), rec.get("SeriesDescription"), rec.get("PatientPosition"))

    return series



def dixon_output_name_from_headers(rec) -> Optional[str]:
    """Return canonical upper/lower Dixon output name from DICOM headers, or None.

    Strong rule:
      ProtocolName with TrunkRegion -> Upper
      ProtocolName with LowerExtrm  -> Lower
      SeriesDescription with W_COMP/water -> W
      SeriesDescription with F_COMP/fat   -> F

    PatientPosition is only a fallback because ProtocolName is more specific for this dataset.
    """
    proto = normalize_header_text(rec.get("ProtocolName"), None, rec.get("PatientPosition"))
    desc = normalize_header_text(None, rec.get("SeriesDescription"), None)
    text = rec.get("norm_text") or ""

    station = None
    if "trunkregion" in proto:
        station = "Upper"
    elif "lowerextrm" in proto or "lower extrem" in proto:
        station = "Lower"
    elif "hfs" in proto:
        station = "Upper"
    elif "ffs" in proto:
        station = "Lower"

    component = None
    if any(tok in desc for tok in ("w comp", "wcomp", "composed w comp", "water")):
        component = "W"
    elif any(tok in desc for tok in ("f comp", "fcomp", "composed f comp", "fat")):
        component = "F"

    # Fallback component detection from combined normalized text.
    if component is None:
        if any(tok in text for tok in ("w comp", "wcomp", "composed w comp", "water")):
            component = "W"
        elif any(tok in text for tok in ("f comp", "fcomp", "composed f comp", "fat")):
            component = "F"

    if station and component:
        return f"Dixon_{component}_COMP_{station}"
    return None

def pick_series(series_dict):
    selected = {}
    global MUST_NOT_CONTAIN, MUST_CONTAIN_ANY, RULES


    # try preferred series numbers first (if configured)
    preferred_hits = set()
    if isinstance(PREFERRED_SERIES_NUMBERS, dict) and PREFERRED_SERIES_NUMBERS:
        # Build reverse lookup: SeriesNumber -> (uid, rec)
        by_sn = {}
        for uid, rec in series_dict.items():
            sn = rec.get('SeriesNumber')
            if not isinstance(sn, int):
                continue
            prev = by_sn.get(sn)
            if prev is None:
                by_sn[sn] = (uid, rec)
            else:
                _, prev_rec = prev
                if int(rec.get('count') or 0) > int(prev_rec.get('count') or 0):
                    by_sn[sn] = (uid, rec)
        for outname, sn in PREFERRED_SERIES_NUMBERS.items():
            if outname in selected:
                continue
            if isinstance(sn, int) and sn in by_sn:
                selected[outname] = by_sn[sn]
                preferred_hits.add(outname)

    for outname, keyword_groups in RULES.items():
        if outname in selected:
            continue
        # SPECIAL-CASE MOCO: choose by ImageType markers + exact ProtocolName station, not by fuzzy keywords.
        # This avoids confusion when multiple series share similar SeriesDescription/ProtocolName (e.g., "..._MOCO" vs derived "..._MOCO_T1/T2").
        if outname in ("T1_MOCO_Thigh","T1_MOCO_Calf","T2_MOCO_Thigh","T2_MOCO_Calf"):
            want_protocol = None
            want_img_tokens = None
            if outname == "T1_MOCO_Thigh":
                want_protocol = "t1map_long_t1_1ktr_thigh"
                want_img_tokens = ("myomaps_t1","t1_map")
            elif outname == "T1_MOCO_Calf":
                want_protocol = "t1map_long_t1_1ktr_calf"
                want_img_tokens = ("myomaps_t1","t1_map")
            elif outname == "T2_MOCO_Thigh":
                want_protocol = "t2map_flash_1ktr_thigh"
                want_img_tokens = ("myomaps_t2","t2_map")
            elif outname == "T2_MOCO_Calf":
                want_protocol = "t2map_flash_1ktr_calf"
                want_img_tokens = ("myomaps_t2","t2_map")

            best_uid = None
            best_rec = None

            for uid, rec in series_dict.items():
                proto = str(rec.get("ProtocolName") or "")
                proto_l = proto.lower()
                if want_protocol and want_protocol not in proto_l:
                    continue

                it = rec.get("ImageType", None)
                it_str = " ".join([str(x) for x in it]) if isinstance(it, (list, tuple)) else str(it or "")
                it_low = it_str.lower()
                if want_img_tokens and not any(tok in it_low for tok in want_img_tokens):
                    # If ImageType missing/stripped, fall back to SeriesDescription suffix or PulseSequenceName.
                    sd = str(rec.get('SeriesDescription') or '').lower()
                    ps = str(rec.get('PulseSequenceName') or '').lower()
                    if outname.startswith('T1_MOCO'):
                        ok = ('moco_t1' in sd) or ('t1_map' in it_low) or ('myomaps_t1' in it_low) or ('tfi2d1' in ps)
                    else:
                        ok = ('moco_t2' in sd) or ('t2_map' in it_low) or ('myomaps_t2' in it_low) or ('tfl2d1' in ps)
                    if not ok:
                        continue

                # Prefer explicit suffix in SeriesDescription if available, else prefer most complete + highest SeriesNumber.
                sd = str(rec.get("SeriesDescription") or "").lower()
                suffix_bonus = 1 if (("moco_t1" in sd) or ("moco_t2" in sd)) else 0

                count = int(rec.get("count") or 0)
                sn = rec.get("SeriesNumber")
                sn_rank = sn if isinstance(sn, int) else -1

                if best_rec is None:
                    best_uid, best_rec = uid, rec
                else:
                    best_count = int(best_rec.get("count") or 0)
                    best_sn = best_rec.get("SeriesNumber")
                    best_sn_rank = best_sn if isinstance(best_sn, int) else -1
                    best_sd = str(best_rec.get("SeriesDescription") or "").lower()
                    best_bonus = 1 if (("moco_t1" in best_sd) or ("moco_t2" in best_sd)) else 0

                    if (suffix_bonus, count, sn_rank) > (best_bonus, best_count, best_sn_rank):
                        best_uid, best_rec = uid, rec

            if best_uid is not None:
                selected[outname] = (best_uid, best_rec)
            continue
        best_uid = None
        best_rec = None

        for uid, rec in series_dict.items():
            text = rec.get("norm_text") or ""
            # Strong Dixon station/component auto-detection from DICOM headers.
            # This prevents upper/lower stations from overwriting each other and does not rely on filenames.
            dixon_name = dixon_output_name_from_headers(rec)
            if dixon_name in RULES:
                # If multiple candidates map to the same output, keep the most complete/highest SeriesNumber series.
                prev = selected.get(dixon_name)
                if prev is None:
                    selected[dixon_name] = (uid, rec)
                else:
                    _, prev_rec = prev
                    count = int(rec.get("count") or 0)
                    prev_count = int(prev_rec.get("count") or 0)
                    sn = rec.get("SeriesNumber") if isinstance(rec.get("SeriesNumber"), int) else -1
                    prev_sn = prev_rec.get("SeriesNumber") if isinstance(prev_rec.get("SeriesNumber"), int) else -1
                    if (count, sn) > (prev_count, prev_sn):
                        selected[dixon_name] = (uid, rec)
                continue

            # STRICT disambiguation for FF vs T2s in qDixon series (they share protocol base)
            if outname in ("FF_Thigh","FF_Calf","T2s_Thigh","T2s_Calf"):
                sd = str(rec.get("SeriesDescription") or "")
                sd_l = sd.lower()
                if outname.startswith("FF_"):
                    if ("_ff" not in sd_l) or ("t2s" in sd_l):
                        continue
                else:  # T2s_*
                    if ("t2s" not in sd_l) or ("_ff" in sd_l):
                        continue


            # MOCO selection should be driven by ImageType (robust against missing "_T1/_T2" suffix in SeriesDescription).
            if outname in ("T1_MOCO_Thigh", "T1_MOCO_Calf", "T2_MOCO_Thigh", "T2_MOCO_Calf"):
                it = rec.get("ImageType", None)
                it_str = " ".join([str(x) for x in it]) if isinstance(it, (list, tuple)) else str(it or "")
                it_low = it_str.lower()

                if outname.startswith("T1_MOCO"):
                    # require T1 map markers
                    if ("myomaps_t1" not in it_low) and ("t1_map" not in it_low):
                        continue
                else:
                    # require T2 map markers
                    if ("myomaps_t2" not in it_low) and ("t2_map" not in it_low):
                        continue


            # (1) Hard exclude tokens for this output
            bad_tokens = MUST_NOT_CONTAIN.get(outname, [])
            if any(bad in text for bad in bad_tokens):
                continue

            # (2) Must-contain-any tokens for this output (optional)
            must_any = MUST_CONTAIN_ANY.get(outname, [])
            if must_any and not any(tok in text for tok in must_any):
                continue

            # (3) Keyword-group match (any group)
            matched = any(all(k.lower() in text for k in group) for group in keyword_groups)
            if not matched:
                continue

            # ranking: prefer most complete series
            count = int(rec.get("count") or 0)
            sn = rec.get("SeriesNumber")
            sn_rank = sn if isinstance(sn, int) else -1

            if best_rec is None:
                best_uid, best_rec = uid, rec
            else:
                best_count = int(best_rec.get("count") or 0)
                best_sn = best_rec.get("SeriesNumber")
                best_sn_rank = best_sn if isinstance(best_sn, int) else -1
                if (count, sn_rank) > (best_count, best_sn_rank):
                    best_uid, best_rec = uid, rec

        if best_uid is not None:
            selected[outname] = (best_uid, best_rec)

    return selected


# =========================
# CONVERSION (per-series temp output)
# =========================


def copy_series_dicoms(session_dir: Path, series_uid: str, tmp_in_dir: Path) -> int:
    copied = 0
    tmp_root = tmp_in_dir.parent.parent  # _dicom_filtered_tmp
    for root, _, files in os.walk(session_dir):
        root_path = Path(root)
        # Skip anything inside the temp staging folder
        try:
            root_path.relative_to(tmp_root)
            continue  # this path is inside tmp_root, skip it
        except ValueError:
            pass  # not inside tmp_root, proceed normally

        for fn in files:
            src = root_path / fn
            ds = read_header(src)
            if ds is None:
                continue
            if getattr(ds, "SeriesInstanceUID", None) == series_uid:
                dst = tmp_in_dir / fn
                if src == dst:
                    continue
                shutil.copy2(src, dst)
                copied += 1
    return copied

def run_dcm2niix(input_dir: Path, out_dir: Path, four_d: str = "n") -> Tuple[int, str, str]:
    """Run dcm2niix on input_dir.

    four_d:
      - 'n' : attempt to avoid 4D NIfTIs (good for most structural scans)
      - 'y' : allow/emit 4D when input is multi-frame or otherwise requires it (needed for some derived maps)
    """
    cmd = [
        str(DCM2NIIX_EXE),
        "-z", "y",
        "-ba", "n",
        "-4", four_d,
        "-f", DCM2NIIX_NAME_TEMPLATE,
        "-o", str(out_dir),
        str(input_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


def choose_primary_nifti(nifti_paths: List[Path]) -> Optional[Path]:
    if not nifti_paths:
        return None
    return max(nifti_paths, key=lambda p: p.stat().st_size)


def already_converted(out_dir: Path) -> bool:
    for outname in RULES.keys():
        if not (out_dir / f"{outname}.nii.gz").exists():
            return False
    return True


# =========================
# ASHA STATION FOLDER ORGANIZATION
# =========================

def copy_if_exists(src: Path, dst: Path) -> bool:
    """Copy a file if it exists, overwriting the destination."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    return True


def organize_upper_lower_dixon_outputs(out_dir: Path) -> List[Dict[str, str]]:
    """Create station-specific MuscleMap input folders from ASHA upper/lower Dixon outputs.

    Keeps the original converted files in the main Musclemap Data folder for QC:
        Dixon_W_COMP_Upper.nii.gz
        Dixon_F_COMP_Upper.nii.gz
        Dixon_W_COMP_Lower.nii.gz
        Dixon_F_COMP_Lower.nii.gz

    Also creates MuscleMap-ready station folders:
        Upper/Dixon_W_COMP.nii.gz
        Upper/Dixon_F_COMP.nii.gz
        Lower/Dixon_W_COMP.nii.gz
        Lower/Dixon_F_COMP.nii.gz
    """
    station_map = {
        "Upper": {
            "Dixon_W_COMP": out_dir / "Dixon_W_COMP_Upper.nii.gz",
            "Dixon_F_COMP": out_dir / "Dixon_F_COMP_Upper.nii.gz",
        },
        "Lower": {
            "Dixon_W_COMP": out_dir / "Dixon_W_COMP_Lower.nii.gz",
            "Dixon_F_COMP": out_dir / "Dixon_F_COMP_Lower.nii.gz",
        },
    }

    rows: List[Dict[str, str]] = []

    for station, files in station_map.items():
        station_dir = out_dir / station
        station_dir.mkdir(parents=True, exist_ok=True)

        for canonical_name, src in files.items():
            dst = station_dir / f"{canonical_name}.nii.gz"
            copied = copy_if_exists(src, dst)

            # Copy matching sidecar JSON if dcm2niix generated one.
            src_json = src.with_suffix("").with_suffix(".json")
            dst_json = dst.with_suffix("").with_suffix(".json")
            json_copied = copy_if_exists(src_json, dst_json)

            shape = ""
            zooms = ""
            if copied:
                try:
                    sh, zz = nifti_shape_zooms(dst)
                    shape, zooms = str(sh), str(zz)
                except Exception as e:
                    shape = f"ERROR_READING_NIFTI: {e}"

            rows.append({
                "Station": station,
                "CanonicalOutput": canonical_name,
                "Source": str(src),
                "Destination": str(dst),
                "Copied": str(bool(copied)),
                "JsonCopied": str(bool(json_copied)),
                "Shape": shape,
                "VoxelSize": zooms,
            })

    manifest_path = out_dir / "station_folder_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Station folders organized: {out_dir / 'Upper'} and {out_dir / 'Lower'}")
    print(f"Station manifest written: {manifest_path}")
    return rows


# =========================
# SESSION DISCOVERY
# =========================

def find_sessions() -> List[Path]:
    """
    Discover sessions to process.

    Portable config behavior:
    - If subjects + sessions/visits are supplied, paths are DATA_ROOT / subject / session.
    - If subjects are supplied without sessions, paths are DATA_ROOT / subject.
    - Without a config/filter, preserves the old PAD behavior: DATA_ROOT / HC* / D*.
    """
    sessions = []

    subjects = SUBJECT_FILTER
    session_names = SESSION_FILTER

    if subjects:
        for subject in subjects:
            subject_dir = DATA_ROOT / subject

            if HAS_VISITS and session_names:
                for session_name in session_names:
                    candidate = subject_dir / session_name
                    if candidate.is_dir():
                        sessions.append(candidate)
                    else:
                        print(f"⚠️ Session folder not found — skipping: {candidate}")
            else:
                if subject_dir.is_dir():
                    sessions.append(subject_dir)
                else:
                    print(f"⚠️ Subject folder not found — skipping: {subject_dir}")

        return sessions

    # Original PAD-style fallback.
    for hc in sorted(DATA_ROOT.glob("HC*")):
        if not hc.is_dir():
            continue
        for d in sorted(hc.glob("D*")):
            if d.is_dir():
                sessions.append(d)
    return sessions

def find_staging_dir(session_dir: Path) -> Optional[Path]:
    matches = []
    for child in session_dir.iterdir():
        if child.is_dir() and DICOM_STAGING_PATTERN.match(child.name):
            matches.append(child)
    if matches:
        return max(matches, key=lambda p: sum(len(f) for _, _, f in os.walk(p)))

    dicomish = []
    for child in session_dir.iterdir():
        if child.is_dir() and "dicom" in child.name.lower():
            dicomish.append(child)
    if dicomish:
        return max(dicomish, key=lambda p: sum(len(f) for _, _, f in os.walk(p)))

    musclemap_dir = session_dir / OUTPUT_FOLDER_NAME
    if musclemap_dir.is_dir():
        candidates = [c for c in musclemap_dir.iterdir() if c.is_dir()]
        if candidates:
            return max(candidates, key=lambda p: sum(len(f) for _, _, f in os.walk(p)))

    return None


# =========================
# MAIN PROCESSING
# =========================

def process_session(session_dir: Path) -> Dict:
    staging = find_staging_dir(session_dir)
    out_dir = session_dir / OUTPUT_FOLDER_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {"session_dir": str(session_dir), "staging_dir": str(staging) if staging else None, "status": "", "notes": ""}

    if staging is None:
        result["status"] = "SKIP_NO_STAGING"
        return result

    if (not FORCE_RECONVERT) and already_converted(out_dir):
        result["status"] = "SKIP_ALREADY_DONE"
        return result

    series_dict = build_series_index(session_dir)
        
    
    if PRINT_SERIES_INVENTORY:
        print(f"\n=== SERIES INVENTORY: {session_dir} ===")
        rows = []
        for uid, rec in series_dict.items():
            rows.append((rec.get("SeriesNumber"), rec.get("count"), rec.get("Modality"), rec.get("ProtocolName"), rec.get("SeriesDescription")))
        rows.sort(key=lambda r: (r[0] is None, r[0]))
        for sn, cnt, mod, pn, sd in rows:
            print(f"{str(sn).rjust(4)} | {str(cnt).rjust(6)} | {mod} | {pn} | {sd}")

    selected = pick_series(series_dict)

      

   

    missing = [k for k in RULES.keys() if k not in selected]
    tmp_root = out_dir / "_dicom_filtered_tmp"
    # === LINE 26+: Remaining Existing Code Continues ===



    missing = [k for k in RULES.keys() if k not in selected]

    tmp_root = out_dir / "_dicom_filtered_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for outname, (uid, rec) in selected.items():
        tmp_in = tmp_root / f"in_{outname}_Series{rec.get('SeriesNumber')}"
        tmp_out = tmp_root / f"out_{outname}_Series{rec.get('SeriesNumber')}"
        tmp_in.mkdir(parents=True, exist_ok=True)
        tmp_out.mkdir(parents=True, exist_ok=True)

       # copied = copy_series_dicoms(staging, uid, tmp_in)
        copied = copy_series_dicoms(session_dir, uid, tmp_in)
        four_d = "y" if outname in ("T1_MOCO_Thigh","T1_MOCO_Calf","T2_MOCO_Thigh","T2_MOCO_Calf") else "n"
        rc, so, se = run_dcm2niix(tmp_in, tmp_out, four_d=four_d)

        produced = list(tmp_out.glob("*.nii.gz"))
        primary = choose_primary_nifti(produced)

        if primary is None and produced:
            primary = produced[0]
            


        final_path = out_dir / f"{outname}.nii.gz"
        renamed_from = ""

        if rc == 0 and primary is not None:
            try:
                renamed_from = str(primary)
                if final_path.exists():
                    final_path.unlink()
                shutil.move(str(primary), str(final_path))
            except Exception:
                pass

        # move matching .json if present
        if final_path.exists() and renamed_from:
            json_src = Path(renamed_from).with_suffix("").with_suffix(".json")
            json_dst = final_path.with_suffix("").with_suffix(".json")
            if json_src.exists():
                try:
                    if json_dst.exists():
                        json_dst.unlink()
                    shutil.move(str(json_src), str(json_dst))
                except Exception:
                    pass

        # collapse 4D->3D if requested
        changed_4d = False
        orig_shape = ""
        new_shape = ""

        shape = ""
        zooms = ""

        if final_path.exists():
            try:
                do_collapse = (COLLAPSE_SCOPE.lower() == "all") or (
                    COLLAPSE_SCOPE.lower() == "seg_only" and outname.startswith("Dixon_")
                )
                if do_collapse:
                    changed_4d, orig_shape, new_shape = collapse_4d_to_3d_inplace(final_path, COLLAPSE_4D_STRATEGY)

                sh, zz = nifti_shape_zooms(final_path)
                shape, zooms = str(sh), str(zz)
            except Exception:
                pass

        manifest_rows.append({
            "Output": outname,
            "DICOMs": copied,
            "PickedSeriesNumber": rec.get("SeriesNumber"),
            "ProtocolName": rec.get("ProtocolName"),
            "SeriesDescription": rec.get("SeriesDescription"),
            "PatientPosition": rec.get("PatientPosition"),
            "dcm2niix_return_code": rc,
            "dcm2niix_error": (se or "").strip()[:2000],
            "RenamedFrom": renamed_from,
            "FinalNifti": str(final_path) if final_path.exists() else "",
            "Shape": shape,
            "VoxelSize": zooms,
            "Collapsed4D": changed_4d,
            "OrigShape": orig_shape,
            "NewShape": new_shape,
        })

    # Create station-specific MuscleMap-ready folders after conversion.
    # Original upper/lower converted files remain in the main Musclemap Data folder for QC.
    station_rows = organize_upper_lower_dixon_outputs(out_dir)

    manifest_path = out_dir / "conversion_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    shutil.rmtree(tmp_root, ignore_errors=True)

    if not selected:
        result["status"] = "FAILED_NO_MATCHES"
        result["notes"] = "No series matched RULES. Turn PRINT_SERIES_INVENTORY=True and tune MUST_CONTAIN_ANY."
        return result

    if missing:
        result["status"] = "DONE_PARTIAL"
        result["notes"] = f"Missing matches for: {missing}. Tune MUST_CONTAIN_ANY/RULES. Manifest: {manifest_path}"
        return result

    result["status"] = "DONE"
    result["notes"] = f"Manifest: {manifest_path}"
    return result


def main():
    parser = argparse.ArgumentParser(description="Portable DICOM to NIfTI conversion pipeline")
    parser.add_argument("--config", required=False, help="Path to study config JSON")
    parser.add_argument("--subjects", nargs="+", required=False, help="Optional subject override")
    parser.add_argument("--sessions", "--days", dest="sessions", nargs="+", required=False, help="Optional session/day override")
    parser.add_argument("--force_reconvert", action="store_true", help="Reconvert even if canonical outputs already exist")
    parser.add_argument("--print_series_inventory", action="store_true", help="Print DICOM series inventory for troubleshooting")
    args = parser.parse_args()

    global DATA_ROOT, DCM2NIIX_EXE, OUTPUT_FOLDER_NAME
    global SUBJECT_FILTER, SESSION_FILTER, HAS_VISITS
    global FORCE_RECONVERT, PRINT_SERIES_INVENTORY

    if args.config:
        config = load_config(args.config)

        DATA_ROOT = resolve_project_path(config["data_root"])
        DCM2NIIX_EXE = resolve_project_path(config.get("dcm2niix_path", "tools/dcm2niix.exe"))
        OUTPUT_FOLDER_NAME = config.get("session_folder_name", "Musclemap Data")

        SUBJECT_FILTER = config.get("subjects", None)

        # Support either "sessions" or "days" in config.
        SESSION_FILTER = config.get("sessions", config.get("days", None))

        # If no sessions/days are provided, assume a single-session study:
        # DATA_ROOT / subject / Musclemap Data
        HAS_VISITS = bool(SESSION_FILTER)

        print("\nUsing portable config:")
        print(f"  DATA_ROOT: {DATA_ROOT}")
        print(f"  DCM2NIIX_EXE: {DCM2NIIX_EXE}")
        print(f"  OUTPUT_FOLDER_NAME: {OUTPUT_FOLDER_NAME}")
        print(f"  SUBJECTS: {SUBJECT_FILTER}")
        print(f"  SESSIONS/DAYS: {SESSION_FILTER}")
        print(f"  HAS_VISITS: {HAS_VISITS}")
    else:
        SUBJECT_FILTER = None
        SESSION_FILTER = None
        HAS_VISITS = True

    # Command-line overrides are useful for testing one subject/session without editing config.
    if args.subjects:
        SUBJECT_FILTER = args.subjects
    if args.sessions:
        SESSION_FILTER = args.sessions
        HAS_VISITS = True

    if args.force_reconvert:
        FORCE_RECONVERT = True
    if args.print_series_inventory:
        PRINT_SERIES_INVENTORY = True

    if not DCM2NIIX_EXE.exists():
        raise FileNotFoundError(f"dcm2niix.exe not found at: {DCM2NIIX_EXE}")

    sessions = [TEST_SESSION] if TEST_SESSION is not None else find_sessions()
    print("Found sessions:", len(sessions))

    all_results = []
    for sess in sessions:
        r = process_session(sess)
        all_results.append(r)
        if r["status"] == "SKIP_NO_STAGING":
            print("No DICOM folder:", sess)
        else:
            print(f"{r['status']}: {sess}")

    summary_path = DATA_ROOT / "dicom_conversion_pipeline_summary.csv"
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    print("Summary written:", summary_path)

def run_dicom_to_nifti(cfg):
    import sys

    argv_old = sys.argv[:]

    sys.argv = [
        "dicom_to_nifti_pipeline.py",
        "--config", cfg["config_path"],
    ]

    if cfg.get("dicom_force_reconvert", False):
        sys.argv += ["--force_reconvert"]

    if cfg.get("dicom_print_series_inventory", False):
        sys.argv += ["--print_series_inventory"]

    try:
        main()
    finally:
        sys.argv = argv_old

if __name__ == "__main__":
    main()