#!/usr/bin/env python
"""
build_wholebodyseg_summary.py

Build vertical/long WholeBodySeg summary outputs.

Primary output:
  WholeBodySeg_metrics_long.csv
    One row per metric.

Secondary output:
  WholeBodySeg_rollups.csv
    Lean, analysis-ready regional rollups.

This replaces the earlier ultra-wide station/subject summary approach.

Expected inputs, when available:
  Station-level:
    <subject>_Dixon_FF_volume.csv
    fat compartment csvs/<subject>_SAT_volume.csv
    fat compartment csvs/<subject>_IMAT_volume.csv
    fat compartment csvs/<subject>_VAT_volume.csv
    TotalSegmentator_FF_volume_metrics_eroded1.csv

  Session-level:
    <subject>_T2_448_Thigh_native_volume_metrics.csv
    <subject>_T2_448_Calf_native_volume_metrics.csv

Metric conventions:
  Dixon-derived generic "mean" -> mean_FF
  Fat compartment generic "mean" -> mean_FF
  TotalSegmentator FF metrics -> mean_FF, volume_ml, etc.
  T2 summary output -> volume_ml and n_voxels only

Regional rollups use volume-weighted means within each station/session only:
  sum(mean * volume_ml) / sum(volume_ml)

WholeBody station-combined rollups are intentionally disabled because Upper/Lower stations may overlap anatomically.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Utility
# ----------------------------
def clean_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s


def safe_token(x) -> str:
    s = clean_text(x).lower()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def read_csv_optional(path: Path) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"WARNING: could not read {path}: {e}")
        return None


def first_existing(folder: Path, patterns: Iterable[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def numeric(x):
    try:
        if x is None:
            return None
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def excel_safe_to_csv(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.6f")


def add_metric(
    rows: List[Dict[str, object]],
    subject: str,
    session: str,
    station: str,
    source: str,
    region: str,
    anatomy: str,
    side: str,
    metric: str,
    value,
    units: str = "",
    source_file: str = "",
    map_name: str = "",
    label_id: object = "",
    notes: str = "",
):
    v = numeric(value)
    if v is None:
        return

    rows.append({
        "subject": subject,
        "session": clean_text(session),
        "station": clean_text(station),
        "source": clean_text(source),
        "region": clean_text(region),
        "anatomy": clean_text(anatomy),
        "side": clean_text(side),
        "metric": clean_text(metric),
        "value": v,
        "units": clean_text(units),
        "map": clean_text(map_name),
        "label_id": clean_text(label_id),
        "source_file": clean_text(source_file),
        "notes": clean_text(notes),
    })


def station_dir_for(data_root: Path, subject: str, session, session_folder_name: str, station: str) -> Path:
    if session in [None, "", "null"]:
        base = data_root / subject
    else:
        base = data_root / subject / str(session)

    if session_folder_name:
        base = base / session_folder_name

    return base / station


def session_dir_for(data_root: Path, subject: str, session, session_folder_name: str) -> Path:
    if session in [None, "", "null"]:
        base = data_root / subject
    else:
        base = data_root / subject / str(session)

    if session_folder_name:
        base = base / session_folder_name

    return base


# ----------------------------
# Muscle grouping rules
# ----------------------------
MUSCLE_GROUPS = {
    "Quad": {
        "vastus lateralis",
        "vastus intermedius",
        "vastus medialis",
        "rectus femoris",
        "quadriceps femoris",
        "quadriceps_femoris",
    },
    "Hamstring": {
        "semimembranosus",
        "semitendinosus",
        "biceps femoris long head",
        "biceps femoris short head",
        "thigh posterior compartment",
        "thigh_posterior_compartment",
    },
    "Adductor": {
        "adductor magnus",
        "adductor longus",
        "adductor brevis",
        "gracilis",
        "pectineus",
        "thigh medial compartment",
        "thigh_medial_compartment",
    },
    "HipPelvis": {
        "gluteus maximus",
        "gluteus medius",
        "gluteus minimus",
        "tensor fascia latae",
        "piriformis",
        "obturator internus",
        "obturator externus",
        "gemelli and quadratus femoris",
        "iliacus",
        "psoas major",
        "iliopsoas",
    },
    "CalfPosterior": {
        "soleus",
        "gastrocnemius",
        "deep posterior compartment",
    },
    "CalfAnteriorLateral": {
        "anterior compartment",
        "lateral compartment",
    },
}

COMPOSITE_GROUPS = {
    "Thigh": ["Quad", "Hamstring", "Adductor"],
    "Calf": ["CalfPosterior", "CalfAnteriorLateral"],
    "LowerLimbPrimary": ["Quad", "Hamstring", "Adductor", "HipPelvis", "CalfPosterior", "CalfAnteriorLateral"],
}


def normalize_anatomy(x: str) -> str:
    return clean_text(x).lower().replace("_", " ")


def group_rows(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    if df is None or df.empty or "anatomy" not in df.columns:
        return pd.DataFrame()

    if group_name in MUSCLE_GROUPS:
        allowed = {normalize_anatomy(a) for a in MUSCLE_GROUPS[group_name]}
        return df[df["anatomy"].map(normalize_anatomy).isin(allowed)].copy()

    if group_name in COMPOSITE_GROUPS:
        parts = [group_rows(df, g) for g in COMPOSITE_GROUPS[group_name]]
        parts = [p for p in parts if not p.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    return pd.DataFrame()


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str = "volume_ml") -> Optional[float]:
    if df is None or df.empty or value_col not in df.columns or weight_col not in df.columns:
        return None

    vals = pd.to_numeric(df[value_col], errors="coerce")
    w = pd.to_numeric(df[weight_col], errors="coerce")
    ok = vals.notna() & w.notna() & (w > 0)

    if not ok.any():
        return None

    return float((vals[ok] * w[ok]).sum() / w[ok].sum())


def sum_value(df: pd.DataFrame, col: str) -> Optional[float]:
    if df is None or df.empty or col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce")
    vals = vals[vals.notna()]
    if vals.empty:
        return None
    return float(vals.sum())


# ----------------------------
# Long table collectors
# ----------------------------
def collect_musclemap_dixon(rows: List[Dict[str, object]], station_dir: Path, subject: str, session: str, station: str):
    p = first_existing(station_dir, [f"{subject}*_Dixon_FF_volume.csv", "*_Dixon_FF_volume.csv"])
    df = read_csv_optional(p)
    if df is None or df.empty:
        return

    for _, r in df.iterrows():
        region = r.get("region", "")
        anatomy = r.get("anatomy", "")
        side = r.get("side", "")
        label_id = r.get("label_id", "")

        metric_map = {
            "n_voxels": ("n_voxels", "voxels"),
            "volume_ml": ("volume_ml", "mL"),
            "mean": ("mean_FF", "fraction"),
            "median": ("median_FF", "fraction"),
            "std": ("std_FF", "fraction"),
            "min": ("min_FF", "fraction"),
            "max": ("max_FF", "fraction"),
        }

        for col, (metric, units) in metric_map.items():
            if col in df.columns:
                add_metric(rows, subject, session, station, "MM_Dixon", region, anatomy, side,
                           metric, r.get(col), units, p.name, label_id=label_id)


def collect_fat_compartments(rows: List[Dict[str, object]], station_dir: Path, subject: str, session: str, station: str):
    # Current fat-compartment metrics are written into a dedicated station-level
    # subfolder rather than directly into the Upper/Lower station directory.
    fat_csv_dir = station_dir / "fat compartment csvs"

    # Prefer the current dedicated CSV folder, but keep a station-root fallback
    # for compatibility with older already-processed datasets.
    search_dirs = [fat_csv_dir, station_dir]

    for comp in ["SAT", "IMAT", "VAT"]:
        p = None
        for folder in search_dirs:
            p = first_existing(folder, [f"{subject}*_{comp}_volume.csv", f"*_{comp}_volume.csv"])
            if p is not None:
                break

        df = read_csv_optional(p)
        if df is None or df.empty:
            continue

        if "compartment" in df.columns:
            cdf = df[df["compartment"].astype(str).str.upper() == comp]
            if not cdf.empty:
                df = cdf

        r = df.iloc[0]
        metric_map = {
            "n_voxels": ("n_voxels", "voxels"),
            "volume_ml": ("volume_ml", "mL"),
            "mean": ("mean_FF", "fraction"),
            "median": ("median_FF", "fraction"),
            "std": ("std_FF", "fraction"),
            "min": ("min_FF", "fraction"),
            "max": ("max_FF", "fraction"),
        }

        for col, (metric, units) in metric_map.items():
            if col in df.columns:
                add_metric(rows, subject, session, station, "Fat", "", comp, "",
                           metric, r.get(col), units, p.name)


def collect_totalseg(rows: List[Dict[str, object]], station_dir: Path, subject: str, session: str, station: str):
    p = station_dir / "TotalSegmentator_FF_volume_metrics_eroded1.csv"
    df = read_csv_optional(p)
    if df is None or df.empty:
        return

    for _, r in df.iterrows():
        anatomy = r.get("structure", f"label_{r.get('label_id', '')}")
        label_id = r.get("label_id", "")

        metric_map = {
            "voxel_count_full_label": ("n_voxels_full", "voxels"),
            "voxel_count_ff_label": ("n_voxels_eroded", "voxels"),
            "volume_ml_full_mask": ("volume_ml", "mL"),
            "mean_ff_fraction_eroded_mask": ("mean_FF", "fraction"),
            "median_ff_fraction_eroded_mask": ("median_FF", "fraction"),
            "sd_ff_fraction_eroded_mask": ("std_FF", "fraction"),
            "fat_volume_ml_full_volume_x_eroded_mean_ff": ("fat_volume_ml", "mL"),
        }

        for col, (metric, units) in metric_map.items():
            if col in df.columns:
                add_metric(rows, subject, session, station, "TS", "", anatomy, "",
                           metric, r.get(col), units, p.name, label_id=label_id)


def collect_t2(rows: List[Dict[str, object]], session_dir: Path, subject: str, session: str):
    files = sorted(session_dir.glob(f"{subject}*_T2_448_*_native_volume_metrics.csv"))
    files = [
        p for p in files
        if not p.name.endswith("_ALL.csv") and not p.name.endswith("_WIDE.csv")
    ]

    for p in files:
        df = read_csv_optional(p)
        if df is None or df.empty:
            continue

        for _, r in df.iterrows():
            region = r.get("region", "")
            anatomy = r.get("anatomy", "")
            side = r.get("side", "")
            map_name = r.get("map", "")
            label_id = r.get("label_id", "")

            # Use map-derived station/region because T2 is session-level.
            # Keep station blank to avoid falsely assigning it to Upper/Lower.
            metric_map = {
                "n_voxels": ("n_voxels", "voxels"),
                "volume_ml": ("volume_ml", "mL"),
            }

            for col, (metric, units) in metric_map.items():
                if col in df.columns:
                    add_metric(rows, subject, session, "", "MM_T2", region, anatomy, side,
                               metric, r.get(col), units, p.name, map_name=map_name, label_id=label_id)


# ----------------------------
# Rollup builder
# ----------------------------
def metrics_to_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long metric table to one row per measured item with metric columns.

    This is internal; final primary output remains long.
    """
    if long_df.empty:
        return pd.DataFrame()

    index_cols = ["subject", "session", "station", "source", "region", "anatomy", "side", "map", "label_id"]
    tmp = long_df.copy()
    pivot = tmp.pivot_table(
        index=index_cols,
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def append_rollup_metric(
    rows: List[Dict[str, object]],
    subject: str,
    session: str,
    station: str,
    source: str,
    group: str,
    metric: str,
    value,
    units: str,
    notes: str = "",
):
    add_metric(rows, subject, session, station, f"{source}_rollup", "", group, "",
               metric, value, units, source_file="", notes=notes)


def build_rollups(long_df: pd.DataFrame) -> pd.DataFrame:
    matrix = metrics_to_matrix(long_df)
    rows: List[Dict[str, object]] = []

    if matrix.empty:
        return pd.DataFrame(rows)

    # Fat rollups: keep direct SAT/IMAT/VAT by station.
    fat = matrix[matrix["source"] == "Fat"].copy()
    for _, r in fat.iterrows():
        for metric, units in [
            ("volume_ml", "mL"),
            ("mean_FF", "fraction"),
            ("median_FF", "fraction"),
            ("std_FF", "fraction"),
            ("n_voxels", "voxels"),
        ]:
            if metric in fat.columns and pd.notna(r.get(metric)):
                append_rollup_metric(
                    rows,
                    r["subject"], r["session"], r["station"],
                    "Fat", r["anatomy"], metric, r[metric], units,
                    notes="direct compartment metric"
                )

    # MuscleMap Dixon rollups by station/source.
    dixon = matrix[matrix["source"] == "MM_Dixon"].copy()
    for (subject, session, station), sdf in dixon.groupby(["subject", "session", "station"], dropna=False):
        for group in list(MUSCLE_GROUPS.keys()) + list(COMPOSITE_GROUPS.keys()):
            gdf = group_rows(sdf, group)
            if gdf.empty:
                continue

            if "side" not in gdf.columns:
                gdf["side"] = ""

            for side, ssdf in gdf.groupby("side", dropna=False):
                group_name = group if clean_text(side) in {"", "unknown"} else f"{group}_{clean_text(side)}"
                vol = sum_value(ssdf, "volume_ml")
                mean_ff = weighted_mean(ssdf, "mean_FF", "volume_ml")
                nvox = sum_value(ssdf, "n_voxels")

                if vol is not None:
                    append_rollup_metric(rows, subject, session, station, "MM_Dixon", group_name,
                                         "volume_ml", vol, "mL", "volume sum")
                if mean_ff is not None:
                    append_rollup_metric(rows, subject, session, station, "MM_Dixon", group_name,
                                         "mean_FF", mean_ff, "fraction", "volume-weighted")
                if nvox is not None:
                    append_rollup_metric(rows, subject, session, station, "MM_Dixon", group_name,
                                         "n_voxels", nvox, "voxels", "voxel sum")

    # T2 rollups. T2 is session-level; station is blank.
    t2 = matrix[matrix["source"] == "MM_T2"].copy()
    for (subject, session), sdf in t2.groupby(["subject", "session"], dropna=False):
        for group in list(MUSCLE_GROUPS.keys()) + list(COMPOSITE_GROUPS.keys()):
            gdf = group_rows(sdf, group)
            if gdf.empty:
                continue

            if "side" not in gdf.columns:
                gdf["side"] = ""

            for side, ssdf in gdf.groupby("side", dropna=False):
                group_name = group if clean_text(side) in {"", "unknown"} else f"{group}_{clean_text(side)}"
                vol = sum_value(ssdf, "volume_ml")
                nvox = sum_value(ssdf, "n_voxels")

                if vol is not None:
                    append_rollup_metric(rows, subject, session, "", "MM_T2", group_name,
                                         "volume_ml", vol, "mL", "volume sum")
                if nvox is not None:
                    append_rollup_metric(rows, subject, session, "", "MM_T2", group_name,
                                         "n_voxels", nvox, "voxels", "voxel sum")


    # WholeBody regional Dixon rollups disabled: Upper/Lower stations may overlap anatomically.
    # WholeBody T2 regional rollups disabled to avoid implying stitched whole-body metrics.
    # WholeBody station-combined rollups disabled: station overlap would double-count tissue.

    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


# ----------------------------
# Config-driven build
# ----------------------------
def build_from_config(config_path: Path):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    data_root = Path(cfg["data_root"])
    subjects = cfg.get("subjects", [])
    sessions = cfg.get("sessions", None)
    stations = cfg.get("stations", ["Upper", "Lower"])
    session_folder_name = cfg.get("session_folder_name", "")

    if sessions is None:
        sessions = [None]

    rows: List[Dict[str, object]] = []

    for subject in subjects:
        for session in sessions:
            session_label = clean_text(session)
            sess_dir = session_dir_for(data_root, subject, session, session_folder_name)

            # Session-level T2 metrics.
            collect_t2(rows, sess_dir, subject, session_label)

            # Station-level metrics.
            for station in stations:
                sdir = station_dir_for(data_root, subject, session, session_folder_name, station)
                if not sdir.exists():
                    print(f"WARNING: station folder not found, skipping summary collection: {sdir}")
                    continue

                collect_musclemap_dixon(rows, sdir, subject, session_label, station)
                collect_fat_compartments(rows, sdir, subject, session_label, station)
                collect_totalseg(rows, sdir, subject, session_label, station)

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        print("WARNING: no metrics found. Summary files not written.")
        return

    # Stable ordering.
    sort_cols = ["subject", "session", "station", "source", "region", "anatomy", "side", "metric"]
    long_df = long_df.sort_values([c for c in sort_cols if c in long_df.columns])

    rollup_df = build_rollups(long_df)
    if not rollup_df.empty:
        rollup_df = rollup_df.sort_values([c for c in sort_cols if c in rollup_df.columns])

    out_long = data_root / "WholeBodySeg_metrics_long.csv"
    out_rollups = data_root / "WholeBodySeg_rollups.csv"

    print("\nNOTE: WholeBody station-combined rollups are disabled to avoid double-counting Upper/Lower overlap.")
    excel_safe_to_csv(long_df, out_long)
    excel_safe_to_csv(rollup_df, out_rollups)

    print("\nSaved:")
    print(f"  {out_long}")
    print(f"  {out_rollups}")
    print(f"Long metric rows: {len(long_df)}")
    print(f"Rollup metric rows: {len(rollup_df)}")


def main():
    ap = argparse.ArgumentParser(description="Build vertical/long WholeBodySeg summary CSVs.")
    ap.add_argument("--config", required=True, help="Path to WholeBodySeg config JSON")
    args = ap.parse_args()
    build_from_config(Path(args.config).resolve())


if __name__ == "__main__":
    main()
