from pathlib import Path
import argparse
import json

from scripts.dicom_to_nifti_pipeline import run_dicom_to_nifti
from scripts.musclemap_dixon_pipeline import run_musclemap_dixon
from scripts.musclemap_t2_448_pipeline import run_musclemap_t2_448
from scripts.totalseg_pipeline import run_totalseg
from scripts.fat_compartment_pipeline import run_fat_compartments

try:
    from scripts.build_wholebodyseg_summary import build_from_config as build_wholebodyseg_summary
except Exception:
    build_wholebodyseg_summary = None


PROGRESS_PREFIX = "[WBS_PROGRESS]"


def emit_progress(event: str, stage_key: str, subject: str = "", session: str = "", station: str = "", label: str = ""):
    """Emit a machine-readable progress event for the GUI."""
    clean = lambda value: str(value or "").replace("|", "/").replace("\n", " ").strip()
    print(
        "|".join(
            [
                PROGRESS_PREFIX,
                clean(event).upper(),
                clean(stage_key),
                clean(subject),
                clean(session),
                clean(station),
                clean(label),
            ]
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="WholeBodySeg")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()

    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Store the actual config path so submodules can reuse the same config file.
    cfg["config_path"] = str(config_path)

    print("\n=== WholeBodySeg ===")
    print(f"Config: {config_path}")
    print(f"Study: {cfg.get('study_name', 'Unnamed')}")

    print("MuscleMap repo:", cfg.get("musclemap_repo"))

    if cfg.get("run_dicom_to_nifti", False):
        print("\n=== DICOM -> NIfTI Conversion ===")
        emit_progress("START", "dicom", label="DICOM -> NIfTI")
        run_dicom_to_nifti(cfg)
        emit_progress("DONE", "dicom", label="DICOM -> NIfTI")

    data_root = Path(cfg["data_root"])
    subjects = cfg.get("subjects", [])
    sessions = cfg.get("sessions", None)
    stations = cfg.get("stations", [])

    if sessions is None:
        sessions = [None]

    for subject in subjects:
        print(f"\nSubject: {subject}")

        session_folder_name = cfg.get("session_folder_name", "")

        for session in sessions:
            if session is None:
                if session_folder_name:
                    session_dir = data_root / subject / session_folder_name
                else:
                    session_dir = data_root / subject

                session_label = ""
            else:
                if session_folder_name:
                    session_dir = data_root / subject / session / session_folder_name
                else:
                    session_dir = data_root / subject / session

                session_label = session

            cfg["current_subject"] = subject
            cfg["current_session"] = session_label

            print(f"Session folder: {session_dir}")

            for station in stations:
                station_dir = session_dir / station
                print(f"\nProcessing station: {station}")
                print(f"Station folder: {station_dir}")

                if not station_dir.exists():
                    dicom_key = f"_auto_dicom_to_nifti_done__{str(session_dir.resolve())}"
                    if not cfg.get(dicom_key, False):
                        cfg[dicom_key] = True
                        print(f"  Station folder not found; running DICOM -> NIfTI conversion to create it: {station_dir}")
                        try:
                            run_dicom_to_nifti(cfg)
                        except Exception as e:
                            print(f"  WARNING: DICOM -> NIfTI conversion failed: {e}")

                    if not station_dir.exists():
                        print(f"  WARNING: station folder still not found after conversion attempt, skipping: {station_dir}")
                        continue
                    else:
                        print(f"  Station folder created: {station_dir}")

                print("Stage flags:")
                print(f"  run_musclemap_dixon:   {cfg.get('run_musclemap_dixon', True)}")
                print(f"  run_totalseg:          {cfg.get('run_totalseg', True)}")
                print(f"  run_fat_compartments:  {cfg.get('run_fat_compartments', False)}")
                print(f"  run_musclemap_t2_448:  {cfg.get('run_musclemap_t2_448', True)}")

                if cfg.get("run_musclemap_dixon", True):
                    print("\n=== MuscleMap Dixon ===")
                    emit_progress("START", "musclemap_dixon", subject, session_label, station, "MuscleMap Dixon")
                    run_musclemap_dixon(station_dir, cfg)
                    emit_progress("DONE", "musclemap_dixon", subject, session_label, station, "MuscleMap Dixon")
                else:
                    print("\nSkipping MuscleMap Dixon")

                if cfg.get("run_totalseg", True):
                    print("\n=== TotalSegmentator ===")
                    emit_progress("START", "totalseg", subject, session_label, station, "TotalSegmentator")
                    run_totalseg(station_dir, cfg)
                    emit_progress("DONE", "totalseg", subject, session_label, station, "TotalSegmentator")
                else:
                    print("\nSkipping TotalSegmentator because run_totalseg is false")

                if cfg.get("run_fat_compartments", False):
                    print("\n=== Fat compartments ===")
                    emit_progress("START", "fat_compartments", subject, session_label, station, "Fat compartments")
                    run_fat_compartments(station_dir, cfg)
                    emit_progress("DONE", "fat_compartments", subject, session_label, station, "Fat compartments")
                else:
                    print("\nSkipping fat compartments")

                if cfg.get("run_musclemap_t2_448", True):
                    # T2-448 files are session-level in the ASHA/WholeBodySeg layout.
                    # Only run once per subject/session even though this loop iterates
                    # over Upper and Lower station folders.
                    t2_key = f"_master_t2_448_done__{str(session_dir.resolve())}"
                    if cfg.get(t2_key, False):
                        print("\nSkipping MuscleMap T2-448; already processed this session folder")
                    else:
                        cfg[t2_key] = True
                        print("\n=== MuscleMap T2-448 ===")
                        emit_progress("START", "musclemap_t2_448", subject, session_label, "", "MuscleMap T2-448")
                        run_musclemap_t2_448(station_dir, cfg)
                        emit_progress("DONE", "musclemap_t2_448", subject, session_label, "", "MuscleMap T2-448")
                else:
                    print("\nSkipping MuscleMap T2-448")

    if cfg.get("run_summary", True):
        print("\n=== WholeBodySeg summary CSVs ===")
        emit_progress("START", "summary", label="WholeBodySeg summary CSVs")
        if build_wholebodyseg_summary is None:
            print("WARNING: scripts/build_wholebodyseg_summary.py could not be imported; summary skipped.")
        else:
            build_wholebodyseg_summary(config_path)
        emit_progress("DONE", "summary", label="WholeBodySeg summary CSVs")
    else:
        print("\nSkipping WholeBodySeg summary CSVs")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()