"""
totalseg_station_workflow_full_eroded.py

ASHA TotalSegmentator station workflow.

For one station folder, runs:
1. TotalSegmentator full-resolution GPU/CPU inference on Dixon water image
2. Combine TotalSegmentator masks into one non-MuscleMap-overlap label map
3. Create a permanent 1-voxel eroded label map
4. Extract:
   - Volume metrics from the full label map
   - FF metrics from the eroded label map

Robustness behavior:
- The primary total_mr task remains required/fatal if it fails.
- Optional subtasks (trunk_cavities, abdominal_muscles, extra/QC subtasks) are
  non-fatal: a failure is logged, partial output is removed, and processing continues.

Required scripts in the same folder as this wrapper:
    combine_totalseg_masks_nonMuscleMap.py
    erode_totalseg_labels.py
    extract_totalseg_ff_volume_metrics_full_eroded.py

Example:
    python totalseg_pipeline.py ^
      --subject P008 ^
      --station Upper

Lower station with automatic fallback:
    python totalseg_pipeline.py ^
      --subject P008 ^
      --station Lower ^
      --auto_fast_fallback

By default, this expects the station folder to contain:
    Dixon_W_COMP.nii.gz
    Dixon_FF_map.nii.gz

The folder location, not the filename, defines Upper vs Lower.
"""

import argparse
import shutil
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd


def run(cmd):
    print("\nRUNNING:")
    print(" ".join(str(x) for x in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(str(x) for x in cmd)}")


def find_totalsegmentator_exe() -> str:
    """
    Resolve the TotalSegmentator console script.

    `pip install --user` (used when the interpreter's own site-packages isn't
    writeable) puts console scripts in a per-user Scripts folder that is often
    missing from PATH, so a bare "TotalSegmentator" call can raise
    FileNotFoundError even though the package is installed. Fall back to the
    Scripts folder next to the current interpreter before giving up.
    """
    found = shutil.which("TotalSegmentator")
    if found:
        return found

    exe_name = "TotalSegmentator.exe" if os.name == "nt" else "TotalSegmentator"
    candidates = [
        Path(sys.executable).parent / "Scripts" / exe_name,
        Path(sys.executable).parent.parent / "Scripts" / exe_name,
    ]
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            py_tag = f"Python{sys.version_info.major}{sys.version_info.minor}"
            candidates.append(Path(appdata) / "Python" / py_tag / "Scripts" / exe_name)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "Could not find the TotalSegmentator executable. It should have been "
        "installed with `pip install TotalSegmentator`; if it landed in a "
        "user Scripts folder outside PATH, add that folder to PATH or "
        "reinstall inside a virtualenv/conda env."
    )


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Required {label} not found: {path}")


def run_totalsegmentator_with_optional_fallback(water: Path, ts_dir: Path, use_fast: bool, auto_fast_fallback: bool):
    """
    Run TotalSegmentator. By default this tries full resolution.
    If full resolution fails and auto_fast_fallback is enabled, it deletes the partial output
    folder and reruns with --fast. This is useful for large lower-body stations where
    nnU-Net/TotalSegmentator may finish GPU prediction but fail during CPU RAM-heavy export.
    """
    def make_cmd(fast: bool):
        cmd = [
            find_totalsegmentator_exe(),
            "-i", str(water),
            "-o", str(ts_dir),
            "--task", "total_mr",
        ]
        if fast:
            cmd.append("--fast")
        return cmd

    if use_fast:
        run(make_cmd(True))
        return "fast_requested"

    try:
        run(make_cmd(False))
        return "full_resolution"
    except RuntimeError as err:
        if not auto_fast_fallback:
            raise

        print("\nWARNING: Full-resolution TotalSegmentator failed.")
        print("This is commonly due to CPU RAM during segmentation export on large stations.")
        print("Retrying automatically with --fast...")

        if ts_dir.exists():
            print(f"Deleting partial TotalSegmentator output folder before fallback: {ts_dir}")
            shutil.rmtree(ts_dir)

        run(make_cmd(True))
        return "fast_fallback_after_fullres_failure"


def run_totalsegmentator_task_with_optional_fallback(
    water: Path,
    out_dir: Path,
    task: str,
    use_fast: bool,
    auto_fast_fallback: bool,
):
    """
    Run an optional TotalSegmentator subtask such as trunk_cavities or
    abdominal_muscles. Kept separate from total_mr so these masks are not merged
    into the non-MuscleMap metrics label map.
    """
    def make_cmd(fast: bool):
        cmd = [
            find_totalsegmentator_exe(),
            "-i", str(water),
            "-o", str(out_dir),
            "--task", str(task),
        ]
        if fast:
            cmd.append("--fast")
        return cmd

    if use_fast:
        run(make_cmd(True))
        return "fast_requested"

    try:
        run(make_cmd(False))
        return "full_resolution"
    except RuntimeError:
        if not auto_fast_fallback:
            raise

        no_fast_tasks = {"abdominal_muscles"}
        if str(task) in no_fast_tasks:
            print(f"\nWARNING: Full-resolution TotalSegmentator subtask failed: {task}")
            print(f"Task {task} does not support --fast; not attempting fast fallback.")
            raise

        print(f"\nWARNING: Full-resolution TotalSegmentator subtask failed: {task}")
        print("Retrying automatically with --fast...")

        if out_dir.exists():
            print(f"Deleting partial TotalSegmentator output folder before fallback: {out_dir}")
            shutil.rmtree(out_dir)

        run(make_cmd(True))
        return "fast_fallback_after_fullres_failure"


def maybe_run_totalsegmentator_subtask(
    water: Path,
    out_dir: Path,
    task: str,
    enabled: bool,
    skip_existing: bool,
    overwrite: bool,
    use_fast: bool,
    auto_fast_fallback: bool,
):
    """
    Run or reuse a TotalSegmentator subtask folder.
    """
    if not enabled:
        return "disabled"

    if overwrite and out_dir.exists():
        print(f"Deleting existing TotalSegmentator {task} folder: {out_dir}")
        shutil.rmtree(out_dir)

    if skip_existing and out_dir.exists():
        print(f"Skipping TotalSegmentator {task}; using existing masks in: {out_dir}")
        return "skipped_existing"

    if out_dir.exists() and any(out_dir.glob("*.nii*")) and not overwrite:
        print(f"TotalSegmentator {task} output already exists; using existing masks in: {out_dir}")
        return "existing_reused"

    return run_totalsegmentator_task_with_optional_fallback(
        water=water,
        out_dir=out_dir,
        task=task,
        use_fast=use_fast,
        auto_fast_fallback=auto_fast_fallback,
    )


def run_optional_subtask_safely(
    water: Path,
    out_dir: Path,
    task: str,
    enabled: bool,
    skip_existing: bool,
    overwrite: bool,
    use_fast: bool,
    auto_fast_fallback: bool,
):
    """Run an optional TotalSegmentator subtask without aborting the whole batch.

    The primary ``total_mr`` task is still treated as required and remains fatal if it
    fails.  Auxiliary tasks (for example ``trunk_cavities``) are allowed to fail for
    one station/subject so the rest of the TotalSegmentator workflow and the larger
    multi-subject batch can continue.

    Any partial output directory created by a failed subtask is removed.  This is
    important because a future run must not mistake an incomplete folder containing
    one or more NIfTI files for a successfully completed subtask.
    """
    try:
        return maybe_run_totalsegmentator_subtask(
            water=water,
            out_dir=out_dir,
            task=task,
            enabled=enabled,
            skip_existing=skip_existing,
            overwrite=overwrite,
            use_fast=use_fast,
            auto_fast_fallback=auto_fast_fallback,
        )
    except Exception as err:
        print("\n" + "!" * 78)
        print(f"WARNING: Optional TotalSegmentator subtask failed: {task}")
        print(f"Station working directory: {Path.cwd()}")
        print(f"Error: {type(err).__name__}: {err}")
        print("This subtask failure is NON-FATAL. The pipeline will continue.")

        if out_dir.exists():
            try:
                print(f"Removing partial failed-subtask output: {out_dir}")
                shutil.rmtree(out_dir)
            except Exception as cleanup_err:
                print(f"WARNING: Could not remove partial output {out_dir}: {cleanup_err}")

        print("!" * 78 + "\n")
        return f"FAILED_NONFATAL ({type(err).__name__}: {err})"


def task_to_folder_name(task: str) -> str:
    """Convert a TotalSegmentator task name to a stable output folder name."""
    return "TS_" + str(task).upper().replace("-", "_")


def copy_nifti_masks_to_stage(src_dir: Path, stage_dir: Path, prefix: str):
    """
    Copy NIfTI masks from src_dir into stage_dir with a prefix so masks from
    multiple TotalSegmentator subtasks can be combined without filename clashes.
    """
    if not src_dir.exists():
        print(f"    NOTE: Cannot stage masks; folder not found: {src_dir}")
        return 0

    count = 0
    for p in sorted(list(src_dir.glob("*.nii.gz")) + list(src_dir.glob("*.nii"))):
        out_name = f"{prefix}__{p.name}"
        shutil.copy2(p, stage_dir / out_name)
        count += 1

    print(f"    Staged {count} masks from {src_dir} into {stage_dir}")
    return count


def build_combined_label_staging_dir(
    base_ts_dir: Path,
    extra_subtask_dirs: dict,
    stage_dir: Path,
    include_extra: bool = True,
):
    """
    Build a flat staging folder for the combine script.

    Always stages total_mr masks. Optionally also stages selected extra subtask
    folders. QC-only subtasks should not be passed in extra_subtask_dirs.
    """
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    n_total = copy_nifti_masks_to_stage(base_ts_dir, stage_dir, "total_mr")

    n_extra = 0
    if include_extra:
        for task, out_dir in extra_subtask_dirs.items():
            n_extra += copy_nifti_masks_to_stage(out_dir, stage_dir, str(task))

    print(f"    Label-map staging complete: total_mr={n_total}, extra={n_extra}, stage={stage_dir}")
    return stage_dir




def combine_vertebrae_mr_masks(vertebrae_dir: Path, out_nii: Path, out_csv: Path, out_ctbl: Path):
    """Combine separate vertebrae_mr binary masks into one multi-label NIfTI."""
    vertebrae_dir = Path(vertebrae_dir)
    mask_files = sorted(list(vertebrae_dir.glob("*.nii.gz")) + list(vertebrae_dir.glob("*.nii")))
    if not mask_files:
        print(f"    NOTE: No vertebrae_mr masks found to combine in: {vertebrae_dir}")
        return None

    ref_img = nib.load(str(mask_files[0]))
    combined = np.zeros(ref_img.shape, dtype=np.uint16)
    rows = []

    def clean_name(p):
        name = p.name[:-7] if p.name.lower().endswith(".nii.gz") else p.stem
        for prefix in ("vertebrae_", "vertebra_", "vertebral_"):
            if name.lower().startswith(prefix):
                return name[len(prefix):]
        return name

    for p in mask_files:
        img = nib.load(str(p))
        if img.shape != ref_img.shape:
            raise ValueError(f"vertebrae_mr shape mismatch: {p.name}={img.shape}, reference={ref_img.shape}")
        mask = img.get_fdata() > 0
        if not np.any(mask):
            continue
        label_id = len(rows) + 1
        write_mask = mask & (combined == 0)
        combined[write_mask] = label_id
        rows.append({
            "label_id": label_id,
            "vertebra": clean_name(p),
            "source_file": p.name,
            "n_voxels": int(np.count_nonzero(write_mask)),
        })

    if not rows:
        return None

    nib.save(nib.Nifti1Image(combined, ref_img.affine, ref_img.header), str(out_nii))
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    lines = ["# Color table generated by WholeBodySeg", "# Label Name R G B A", "0 Background 0 0 0 0"]
    for row in rows:
        i = row["label_id"]
        r, g, b = (53*i+67)%256, (97*i+113)%256, (149*i+41)%256
        lines.append(f'{i} {row["vertebra"]} {r} {g} {b} 255')
    Path(out_ctbl).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"    Combined vertebrae_mr label map: {out_nii}")
    print(f"    Vertebra labels written: {len(rows)}")
    return out_nii



def main():
    parser = argparse.ArgumentParser(
        description="Run TotalSegmentator full-res, build full/eroded label maps, and extract FF/volume metrics."
    )
    parser.add_argument("--water", default="Dixon_W_COMP.nii.gz", help="Dixon water image. Default: Dixon_W_COMP.nii.gz")
    parser.add_argument("--fat", default="Dixon_F_COMP.nii.gz", help="Dixon fat image. Used for Upper tissue_types_mr when enabled.")
    parser.add_argument("--ff", default="Dixon_FF_map.nii.gz", help="Dixon FF map. Default: Dixon_FF_map.nii.gz")
    parser.add_argument("--subject", default="", help="Subject ID, e.g. P008")
    parser.add_argument("--session", default="", help="Optional session/day/visit")
    parser.add_argument("--station", required=True, help="Station name, e.g. Upper or Lower")

    parser.add_argument("--ts_out_dir", default="TS_TOTAL_MR_FULL", help="Raw TotalSegmentator output directory.")
    parser.add_argument("--erode_voxels", type=int, default=1, help="Erosion for FF label map. Default: 1.")
    parser.add_argument("--min_voxels", type=int, default=500, help="Minimum voxels for combined label map. Default: 500.")
    parser.add_argument("--skip_totalseg", action="store_true", help="Skip TotalSegmentator if raw masks already exist.")
    parser.add_argument("--overwrite_ts", action="store_true", help="Delete existing TotalSegmentator output folder before rerun.")
    parser.add_argument("--fast", action="store_true", help="Use TotalSegmentator --fast. Default is full resolution.")
    parser.add_argument(
        "--auto_fast_fallback",
        action="store_true",
        help="Try full resolution first; if TotalSegmentator fails, delete partial output and rerun with --fast."
    )

    parser.add_argument("--run_trunk_cavities", action="store_true", help="Run TotalSegmentator --task trunk_cavities.")
    parser.add_argument("--trunk_cavities_out_dir", default="TS_TRUNK_CAVITIES", help="Output folder for trunk_cavities.")
    parser.add_argument("--skip_trunk_cavities", action="store_true", help="Reuse existing trunk_cavities output folder.")
    parser.add_argument("--overwrite_trunk_cavities", action="store_true", help="Delete and rerun trunk_cavities output folder.")

    parser.add_argument("--run_abdominal_muscles", action="store_true", help="Run TotalSegmentator --task abdominal_muscles.")
    parser.add_argument("--abdominal_muscles_out_dir", default="TS_ABDOMINAL_MUSCLES", help="Output folder for abdominal_muscles.")
    parser.add_argument("--skip_abdominal_muscles", action="store_true", help="Reuse existing abdominal_muscles output folder.")
    parser.add_argument("--overwrite_abdominal_muscles", action="store_true", help="Delete and rerun abdominal_muscles output folder.")
    parser.add_argument("--abdominal_muscles_fast", action="store_true", help="Run abdominal_muscles subtask with TotalSegmentator --fast, independent of global --fast.")

    parser.add_argument("--run_extra_subtasks", action="store_true", help="Run additional TotalSegmentator subtasks listed by --extra_subtasks.")
    parser.add_argument("--extra_subtasks", nargs="*", default=[], help="Extra subtasks to run and optionally include in combined label map.")
    parser.add_argument("--qc_only_subtasks", nargs="*", default=[], help="Extra subtasks to run for QC only; not merged into combined label map.")
    parser.add_argument("--upper_only_qc_subtasks", nargs="*", default=[], help="QC-only subtasks that should run only for the Upper station.")
    parser.add_argument("--include_extra_subtasks_in_combined_label", action="store_true", help="Include --extra_subtasks outputs in TS_Organs_Mask.nii.gz.")
    parser.add_argument("--extra_subtasks_stage_dir", default="TS_EXTRA_FOR_LABELS", help="Temporary staging folder used to combine total_mr plus selected extra subtasks.")
    parser.add_argument("--skip_extra_subtasks", action="store_true", help="Reuse existing extra/QC subtask output folders when present.")
    parser.add_argument("--overwrite_extra_subtasks", action="store_true", help="Delete and rerun extra/QC subtask output folders.")
    parser.add_argument(
        "--upper_tissue_types_mr_use_dixon_f",
        action="store_true",
        help="For the Upper station only, run tissue_types_mr on Dixon_F_COMP instead of Dixon_W_COMP.",
    )


    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    cwd = Path.cwd()

    water = Path(args.water)
    fat = Path(args.fat)
    ff = Path(args.ff)
    ts_dir = Path(args.ts_out_dir)
    trunk_cavities_dir = Path(args.trunk_cavities_out_dir)
    abdominal_muscles_dir = Path(args.abdominal_muscles_out_dir)
    extra_stage_dir = Path(args.extra_subtasks_stage_dir)

    require_file(water, "Dixon water image")
    require_file(ff, "Dixon FF map")

    combine_script = script_dir / "combine_totalseg_masks_nonMuscleMap.py"
    erode_script = script_dir / "erode_totalseg_labels.py"
    metrics_script = script_dir / "extract_totalseg_ff_volume_metrics_full_eroded.py"

    require_file(combine_script, "combine script")
    require_file(erode_script, "erosion script")
    require_file(metrics_script, "metrics script")

    full_label = Path("TS_Organs_Mask.nii.gz")
    full_csv = Path("TS_Organs_Mask.csv")
    eroded_label = Path(f"TS_Organs_Mask_eroded{args.erode_voxels}.nii.gz")
    eroded_stats = Path(f"TS_Organs_Mask_eroded{args.erode_voxels}_stats.csv")
    metrics_csv = Path(f"TotalSegmentator_FF_volume_metrics_eroded{args.erode_voxels}.csv")

    if args.overwrite_ts and ts_dir.exists():
        print(f"Deleting existing TotalSegmentator output folder: {ts_dir}")
        shutil.rmtree(ts_dir)

    totalseg_mode = "skipped_existing"
    if not args.skip_totalseg:
        totalseg_mode = run_totalsegmentator_with_optional_fallback(
            water=water,
            ts_dir=ts_dir,
            use_fast=args.fast,
            auto_fast_fallback=args.auto_fast_fallback,
        )
    else:
        if not ts_dir.exists():
            raise FileNotFoundError(f"--skip_totalseg was used, but output directory does not exist: {ts_dir}")
        print(f"Skipping TotalSegmentator; using existing masks in: {ts_dir}")

    # IMPORTANT:
    # Global --fast is intended only for the primary total_mr task.
    # Several TotalSegmentator subtasks either do not support --fast
    # (for example trunk_cavities) or may have task-specific behavior.
    # Therefore, do not propagate args.fast to subtasks.
    #
    # If a subtask fails at full resolution, also do not auto-fallback to
    # --fast here. This keeps subtask calls predictable and prevents crashes
    # from unsupported task/--fast combinations.
    subtask_use_fast = False
    subtask_auto_fast_fallback = False

    trunk_cavities_mode = run_optional_subtask_safely(
        water=water,
        out_dir=trunk_cavities_dir,
        task="trunk_cavities",
        enabled=args.run_trunk_cavities,
        skip_existing=args.skip_trunk_cavities,
        overwrite=args.overwrite_trunk_cavities,
        use_fast=subtask_use_fast,
        auto_fast_fallback=subtask_auto_fast_fallback,
    )

    abdominal_muscles_mode = run_optional_subtask_safely(
        water=water,
        out_dir=abdominal_muscles_dir,
        task="abdominal_muscles",
        enabled=args.run_abdominal_muscles,
        skip_existing=args.skip_abdominal_muscles,
        overwrite=args.overwrite_abdominal_muscles,
        use_fast=bool(args.abdominal_muscles_fast),
        auto_fast_fallback=False,
    )

    extra_subtask_modes = {}
    qc_only_modes = {}

    if args.run_extra_subtasks:
        for task in args.extra_subtasks:
            out_dir = Path(task_to_folder_name(task))
            mode = run_optional_subtask_safely(
                water=water,
                out_dir=out_dir,
                task=task,
                enabled=True,
                skip_existing=args.skip_extra_subtasks,
                overwrite=args.overwrite_extra_subtasks,
                use_fast=subtask_use_fast,
                auto_fast_fallback=subtask_auto_fast_fallback,
            )
            extra_subtask_modes[task] = (mode, out_dir)

        upper_only_qc = {str(x).strip().lower() for x in args.upper_only_qc_subtasks}

        for task in args.qc_only_subtasks:
            task_low = str(task).strip().lower()
            if task_low in upper_only_qc and str(args.station).strip().lower() != "upper":
                print(f"    Skipping QC-only subtask {task} on {args.station}; configured Upper-only.")
                qc_only_modes[task] = ("upper_only_skipped", Path(task_to_folder_name(task)))
                continue

            out_dir = Path(task_to_folder_name(task))
            subtask_input = water
            if (
                str(args.station).strip().lower() == "upper"
                and task_low == "tissue_types_mr"
                and args.upper_tissue_types_mr_use_dixon_f
            ):
                require_file(fat, "Dixon fat image for Upper tissue_types_mr")
                subtask_input = fat
                print(f"    Upper tissue_types_mr input override: using Dixon fat image: {fat}")
            elif task_low == "tissue_types_mr":
                print(f"    tissue_types_mr input: using existing Dixon water path: {water}")

            mode = run_optional_subtask_safely(
                water=subtask_input,
                out_dir=out_dir,
                task=task,
                enabled=True,
                skip_existing=args.skip_extra_subtasks,
                overwrite=args.overwrite_extra_subtasks,
                use_fast=subtask_use_fast,
                auto_fast_fallback=subtask_auto_fast_fallback,
            )
            qc_only_modes[task] = (mode, out_dir)

    vertebrae_task_entry = qc_only_modes.get("vertebrae_mr")
    if vertebrae_task_entry is not None:
        vertebrae_mode, vertebrae_dir = vertebrae_task_entry
        if (
            str(args.station).strip().lower() == "upper"
            and vertebrae_mode not in ("disabled", "upper_only_skipped")
            and Path(vertebrae_dir).exists()
        ):
            combine_vertebrae_mr_masks(
                Path(vertebrae_dir),
                Path("VertebraeMR_labels.nii.gz"),
                Path("VertebraeMR_labels.csv"),
                Path("VertebraeMR_labels.ctbl"),
            )

    combine_mask_dir = ts_dir
    if args.include_extra_subtasks_in_combined_label and extra_subtask_modes:
        include_dirs = {
            task: out_dir
            for task, (mode, out_dir) in extra_subtask_modes.items()
            if mode != "disabled" and out_dir.exists()
        }
        combine_mask_dir = build_combined_label_staging_dir(
            base_ts_dir=ts_dir,
            extra_subtask_dirs=include_dirs,
            stage_dir=extra_stage_dir,
            include_extra=True,
        )

    run([
        sys.executable,
        str(combine_script),
        "--mask_dir", str(combine_mask_dir),
        "--out_nii", str(full_label),
        "--out_csv", str(full_csv),
        "--min_voxels", str(args.min_voxels),
    ])

    run([
        sys.executable,
        str(erode_script),
        "--label_map", str(full_label),
        "--label_csv", str(full_csv),
        "--out_nii", str(eroded_label),
        "--out_csv", str(eroded_stats),
        "--erode_voxels", str(args.erode_voxels),
    ])

    run([
        sys.executable,
        str(metrics_script),
        "--ff_map", str(ff),
        "--label_map_full", str(full_label),
        "--label_map_ff", str(eroded_label),
        "--label_csv", str(full_csv),
        "--out_csv", str(metrics_csv),
        "--subject", args.subject,
        "--session", args.session,
        "--station", args.station,
    ])

    print("\nWorkflow complete.")
    print(f"Working folder: {cwd}")
    print("Created/updated:")
    print(f"  TotalSegmentator mode:      {totalseg_mode}")
    print(f"  Raw TotalSegmentator masks: {ts_dir}")
    print(f"  Trunk cavities mode:        {trunk_cavities_mode}")
    print(f"  Trunk cavities masks:       {trunk_cavities_dir}")
    print(f"  Abdominal muscles mode:     {abdominal_muscles_mode}")
    print(f"  Abdominal muscles masks:    {abdominal_muscles_dir}")
    if extra_subtask_modes:
        print("  Extra subtasks merged/QC:")
        for task, (mode, out_dir) in extra_subtask_modes.items():
            print(f"    {task}: {mode} -> {out_dir}")
    if qc_only_modes:
        print("  QC-only subtasks:")
        for task, (mode, out_dir) in qc_only_modes.items():
            print(f"    {task}: {mode} -> {out_dir}")
    print(f"  Full label map:             {full_label}")
    print(f"  Label CSV:                  {full_csv}")
    print(f"  Eroded label map:           {eroded_label}")
    print(f"  Erosion stats CSV:          {eroded_stats}")
    print(f"  Metrics CSV:                {metrics_csv}")
    print("\nMetric rule:")
    print("  Volume metrics -> full label map")
    print("  FF metrics     -> eroded label map")


def run_totalseg(station_dir, cfg):
    from pathlib import Path
    import sys

    station_dir = Path(station_dir)

    subject = cfg.get("current_subject", "")
    session = cfg.get("current_session", "")
    station = station_dir.name if station_dir.name else str(station_dir)

    argv_old = sys.argv[:]

    sys.argv = [
        "totalseg_pipeline.py",
        "--station", station,
        "--water", cfg.get("water_image_name", "Dixon_W_COMP.nii.gz"),
        "--fat", cfg.get("fat_image_name", "Dixon_F_COMP.nii.gz"),
        "--ff", cfg.get("ff_map_name", "Dixon_FF_map.nii.gz"),
        "--ts_out_dir", cfg.get("totalseg_out_dir", "TS_TOTAL_MR_FULL"),
        "--erode_voxels", str(cfg.get("totalseg_erode_voxels", 1)),
        "--min_voxels", str(cfg.get("totalseg_min_voxels", 500)),
        "--trunk_cavities_out_dir", cfg.get("totalseg_trunk_cavities_folder", cfg.get("trunk_cavities_out_dir", "TS_TRUNK_CAVITIES")),
        "--abdominal_muscles_out_dir", cfg.get("totalseg_abdominal_muscles_folder", cfg.get("abdominal_muscles_out_dir", "TS_ABDOMINAL_MUSCLES")),
    ]

    if cfg.get("run_totalseg_trunk_cavities", False):
        sys.argv += ["--run_trunk_cavities"]

    if cfg.get("totalseg_skip_trunk_cavities", False):
        sys.argv += ["--skip_trunk_cavities"]

    if cfg.get("totalseg_overwrite_trunk_cavities", False):
        sys.argv += ["--overwrite_trunk_cavities"]

    if cfg.get("run_totalseg_abdominal_muscles", False):
        sys.argv += ["--run_abdominal_muscles"]

    if cfg.get("totalseg_skip_abdominal_muscles", False):
        sys.argv += ["--skip_abdominal_muscles"]

    if cfg.get("totalseg_overwrite_abdominal_muscles", False):
        sys.argv += ["--overwrite_abdominal_muscles"]

    if cfg.get("totalseg_abdominal_muscles_fast", False):
        sys.argv += ["--abdominal_muscles_fast"]

    if cfg.get("run_totalseg_extra_subtasks", False):
        sys.argv += ["--run_extra_subtasks"]

        extra_subtasks = cfg.get("totalseg_extra_subtasks", [])
        if extra_subtasks:
            sys.argv += ["--extra_subtasks"] + [str(x) for x in extra_subtasks]

        qc_only_subtasks = cfg.get("totalseg_qc_only_subtasks", [])
        if qc_only_subtasks:
            sys.argv += ["--qc_only_subtasks"] + [str(x) for x in qc_only_subtasks]

        upper_only_qc_subtasks = cfg.get("totalseg_upper_only_qc_subtasks", [])
        if upper_only_qc_subtasks:
            sys.argv += ["--upper_only_qc_subtasks"] + [str(x) for x in upper_only_qc_subtasks]

    if cfg.get("include_extra_subtasks_in_combined_label", False):
        sys.argv += ["--include_extra_subtasks_in_combined_label"]

    if cfg.get("totalseg_skip_extra_subtasks", True):
        sys.argv += ["--skip_extra_subtasks"]

    if cfg.get("totalseg_overwrite_extra_subtasks", False):
        sys.argv += ["--overwrite_extra_subtasks"]

    if cfg.get("upper_tissue_types_mr_use_dixon_f", False):
        sys.argv += ["--upper_tissue_types_mr_use_dixon_f"]

    if cfg.get("totalseg_extra_subtasks_stage_dir", ""):
        sys.argv += ["--extra_subtasks_stage_dir", str(cfg.get("totalseg_extra_subtasks_stage_dir"))]

    if subject:
        sys.argv += ["--subject", subject]

    if session:
        sys.argv += ["--session", session]

    if cfg.get("totalseg_skip_existing", False):
        sys.argv += ["--skip_totalseg"]

    if cfg.get("totalseg_overwrite", False):
        sys.argv += ["--overwrite_ts"]

    if cfg.get("totalseg_fast", False):
        sys.argv += ["--fast"]

    if cfg.get("totalseg_auto_fast_fallback", True):
        sys.argv += ["--auto_fast_fallback"]

    try:
        old_cwd = Path.cwd()
        os.chdir(station_dir)
        main()
    finally:
        os.chdir(old_cwd)
        sys.argv = argv_old


if __name__ == "__main__":
    main()