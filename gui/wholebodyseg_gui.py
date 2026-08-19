"""
wholebodyseg_gui.py

Simple Tkinter front-end for WholeBodySeg.py.

Lets the user:
- Pick a study config (from configs/*.json, or browse for any config JSON)
- Change the data directory
- Choose which patients (subject folders under the data directory) to process
- Toggle pipeline stages (DICOM->NIfTI, MuscleMap Dixon, TotalSegmentator, fat compartments, T2-448)
- Run WholeBodySeg.py and watch live log output

Run with:  python gui/wholebodyseg_gui.py
"""

import json
import subprocess
import sys
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import environment_setup

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_DIR / "configs"
RUN_CONFIGS_DIR = PROJECT_DIR / "gui" / "run_configs"
WHOLEBODYSEG_SCRIPT = PROJECT_DIR / "WholeBodySeg.py"
ICON_ICO = Path(__file__).resolve().parent / "assets" / "Icon.ico"
ICON_PNG = PROJECT_DIR / "Icon.png"

# (config_key, display_label, indent)
STAGE_FLAGS = [
    ("run_dicom_to_nifti", "Convert DICOM → NIfTI", 0),
    ("dicom_force_reconvert", "Force reconvert existing NIfTI", 1),
    ("dicom_print_series_inventory", "Print DICOM series inventory (debug)", 1),
    ("run_musclemap_dixon", "MuscleMap Dixon segmentation", 0),
    ("run_totalseg", "TotalSegmentator", 0),
    ("totalseg_fast", "TotalSegmentator fast mode", 1),
    ("run_fat_compartments", "Fat compartment analysis", 0),
    ("run_musclemap_t2_448", "MuscleMap T2-448 segmentation", 0),
    ("run_summary", "Build WholeBodySeg summary CSVs", 0),
]

FAT_COMPARTMENT_MODE_OPTIONS = {
    "Build + Metrics": "build_and_metrics",
    "Build Only": "build",
    "Metrics Only (QC Re-export)": "metrics",
}
FAT_COMPARTMENT_MODE_LABELS = {
    value: label for label, value in FAT_COMPARTMENT_MODE_OPTIONS.items()
}
DEFAULT_FAT_COMPARTMENT_MODE = "build_and_metrics"

# Machine-readable progress events emitted by WholeBodySeg.py.
PROGRESS_PREFIX = "[WBS_PROGRESS]"

# Relative runtime weights. These are intentionally approximate; they make the
# progress percentage and ETA more realistic than treating every stage equally.
STAGE_WEIGHTS = {
    "dicom": 15.0,
    "musclemap_dixon": 20.0,
    "totalseg": 40.0,
    "fat_compartments": 10.0,
    "musclemap_t2_448": 10.0,
    "summary": 5.0,
}

DONE_SENTINEL = object()


class WholeBodySegGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("WholeBodySeg Pipeline Runner")
        self.root.geometry("880x760")
        self.root.minsize(760, 600)
        self._set_window_icon()

        self.loaded_config = {}
        self.patient_vars: dict[str, tk.BooleanVar] = {}
        self.stage_vars = {key: tk.BooleanVar(value=False) for key, _, _ in STAGE_FLAGS}
        self.fat_compartment_mode_var = tk.StringVar(
            value=FAT_COMPARTMENT_MODE_LABELS[DEFAULT_FAT_COMPARTMENT_MODE]
        )
        self.fat_compartment_mode_combo = None
        self.proc = None
        self.busy = False
        self.log_queue: "queue.Queue" = queue.Queue()

        self.progress_determinate = False
        self.progress_total = 0
        self.progress_completed = 0
        self.progress_total_weight = 1.0
        self.progress_completed_weight = 0.0
        self.progress_elapsed_at_last_completion = 0.0
        self.completed_progress_events = set()
        self.current_stage_key = ""
        self.current_stage_text = ""
        self.run_start_time = None
        self.stop_requested = False

        self._build_widgets()
        self._populate_config_dropdown()
        self.root.after(100, self._poll_log_queue)
        self.root.after(1000, self._update_progress_clock)

    def _set_window_icon(self):
        """Set the title-bar/taskbar icon from gui/assets/Icon.ico (falls back to Icon.png)."""
        try:
            if ICON_ICO.exists():
                self.root.iconbitmap(default=str(ICON_ICO))
                return
        except tk.TclError:
            pass
        if ICON_PNG.exists():
            # Keep a reference on self; Tk drops the image if it gets garbage collected.
            self._icon_photo = tk.PhotoImage(file=str(ICON_PNG))
            self.root.iconphoto(True, self._icon_photo)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_widgets(self):
        pad = {"padx": 8, "pady": 4}

        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Study config ---
        cfg_frame = ttk.LabelFrame(outer, text="Study config")
        cfg_frame.pack(fill="x", **pad)

        self.config_choice_var = tk.StringVar()
        self.config_combo = ttk.Combobox(cfg_frame, textvariable=self.config_choice_var, state="readonly", width=40)
        self.config_combo.grid(row=0, column=0, padx=6, pady=6, sticky="w")
        self.config_combo.bind("<<ComboboxSelected>>", self._on_config_selected)

        ttk.Button(cfg_frame, text="Browse...", command=self._browse_config).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(cfg_frame, text="Reload", command=self._reload_config).grid(row=0, column=2, padx=6, pady=6)

        self.config_path_var = tk.StringVar(value="(no config loaded)")
        ttk.Label(cfg_frame, textvariable=self.config_path_var, foreground="#555").grid(
            row=1, column=0, columnspan=3, padx=6, pady=(0, 6), sticky="w"
        )

        # --- Data directory ---
        dir_frame = ttk.LabelFrame(outer, text="Data directory")
        dir_frame.pack(fill="x", **pad)

        self.data_root_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.data_root_var).grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        dir_frame.columnconfigure(0, weight=1)
        ttk.Button(dir_frame, text="Browse...", command=self._browse_data_root).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(dir_frame, text="Refresh patients", command=self._refresh_patients).grid(row=0, column=2, padx=6, pady=6)

        # --- Patients ---
        patients_frame = ttk.LabelFrame(outer, text="Patients")
        patients_frame.pack(fill="both", expand=True, **pad)

        btn_row = ttk.Frame(patients_frame)
        btn_row.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(btn_row, text="Select all", command=lambda: self._set_all_patients(True)).pack(side="left")
        ttk.Button(btn_row, text="Select none", command=lambda: self._set_all_patients(False)).pack(side="left", padx=6)

        canvas_holder = ttk.Frame(patients_frame)
        canvas_holder.pack(fill="both", expand=True, padx=6, pady=6)

        self.patients_canvas = tk.Canvas(canvas_holder, borderwidth=0, highlightthickness=0, height=140)
        scrollbar = ttk.Scrollbar(canvas_holder, orient="vertical", command=self.patients_canvas.yview)
        self.patients_inner = ttk.Frame(self.patients_canvas)
        self.patients_inner.bind(
            "<Configure>", lambda e: self.patients_canvas.configure(scrollregion=self.patients_canvas.bbox("all"))
        )
        self.patients_canvas.create_window((0, 0), window=self.patients_inner, anchor="nw")
        self.patients_canvas.configure(yscrollcommand=scrollbar.set)
        self.patients_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Sessions / stations ---
        extra_frame = ttk.Frame(outer)
        extra_frame.pack(fill="x", **pad)
        extra_frame.columnconfigure(1, weight=1)
        extra_frame.columnconfigure(3, weight=1)

        ttk.Label(extra_frame, text="Sessions (comma-sep, blank = none):").grid(row=0, column=0, sticky="w")
        self.sessions_var = tk.StringVar()
        ttk.Entry(extra_frame, textvariable=self.sessions_var).grid(row=0, column=1, sticky="ew", padx=6)

        ttk.Label(extra_frame, text="Stations (comma-sep):").grid(row=0, column=2, sticky="w")
        self.stations_var = tk.StringVar()
        ttk.Entry(extra_frame, textvariable=self.stations_var).grid(row=0, column=3, sticky="ew", padx=6)

        # --- Pipeline stages ---
        stages_frame = ttk.LabelFrame(outer, text="Pipeline stages")
        stages_frame.pack(fill="x", **pad)

        stage_row = 0
        for key, label, indent in STAGE_FLAGS:
            command = self._update_fat_compartment_mode_state if key == "run_fat_compartments" else None
            cb = ttk.Checkbutton(
                stages_frame,
                text=label,
                variable=self.stage_vars[key],
                command=command,
            )
            cb.grid(row=stage_row, column=0, sticky="w", padx=(20 + indent * 20, 6), pady=1)
            stage_row += 1

            if key == "run_fat_compartments":
                ttk.Label(stages_frame, text="Fat compartment mode:").grid(
                    row=stage_row,
                    column=0,
                    sticky="w",
                    padx=(40, 6),
                    pady=(2, 0),
                )
                self.fat_compartment_mode_combo = ttk.Combobox(
                    stages_frame,
                    textvariable=self.fat_compartment_mode_var,
                    state="readonly",
                    values=list(FAT_COMPARTMENT_MODE_OPTIONS.keys()),
                    width=31,
                )
                self.fat_compartment_mode_combo.grid(
                    row=stage_row,
                    column=1,
                    sticky="w",
                    padx=(0, 8),
                    pady=(2, 1),
                )
                stage_row += 1

        self._update_fat_compartment_mode_state()

        # --- Interpreter ---
        interp_frame = ttk.Frame(outer)
        interp_frame.pack(fill="x", **pad)
        interp_frame.columnconfigure(1, weight=1)
        ttk.Label(interp_frame, text="Python interpreter:").grid(row=0, column=0, sticky="w")
        default_interpreter = environment_setup.find_existing_env_python()
        self.interpreter_var = tk.StringVar(value=str(default_interpreter) if default_interpreter else sys.executable)
        ttk.Entry(interp_frame, textvariable=self.interpreter_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(interp_frame, text="Browse...", command=self._browse_interpreter).grid(row=0, column=2)
        self.setup_button = ttk.Button(
            interp_frame, text="Auto Setup Environment", command=self._on_setup_environment
        )
        self.setup_button.grid(row=0, column=3, padx=(6, 0))
        ttk.Label(
            interp_frame,
            text=(
                "Must be the environment with pydicom, nibabel, TotalSegmentator, torch, etc. installed.\n"
                "Don't have one? Click 'Auto Setup Environment' to install everything automatically (conda + packages)."
            ),
            foreground="#555",
            justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 0))

        # --- Run controls ---
        run_frame = ttk.Frame(outer)
        run_frame.pack(fill="x", **pad)
        self.run_button = ttk.Button(run_frame, text="Run pipeline", command=self._on_run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(run_frame, text="Stop", command=self._on_stop, state="disabled")
        self.stop_button.pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(run_frame, textvariable=self.status_var).pack(side="left", padx=(6, 0))

        progress_frame = ttk.Frame(outer)
        progress_frame.pack(fill="x", **pad)
        self.current_stage_var = tk.StringVar(value="Current: —")
        ttk.Label(progress_frame, textvariable=self.current_stage_var).pack(
            anchor="w", pady=(0, 2)
        )
        self.progress = ttk.Progressbar(progress_frame, mode="indeterminate")
        self.progress.pack(fill="x", expand=True)
        self.progress_label_var = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.progress_label_var, foreground="#555").pack(
            anchor="w", pady=(2, 0)
        )

        # --- Log ---
        log_frame = ttk.LabelFrame(outer, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=16, state="disabled", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

    def _update_fat_compartment_mode_state(self):
        """Enable the fat-compartment mode selector only when that stage is selected."""
        if self.fat_compartment_mode_combo is None:
            return
        enabled = self.stage_vars["run_fat_compartments"].get()
        self.fat_compartment_mode_combo.configure(state="readonly" if enabled else "disabled")

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------
    def _populate_config_dropdown(self):
        files = sorted(CONFIGS_DIR.glob("*.json")) if CONFIGS_DIR.is_dir() else []
        self.config_combo["values"] = [f.name for f in files]
        if files:
            self.config_combo.current(0)
            self._load_config_file(files[0])

    def _on_config_selected(self, _event=None):
        name = self.config_choice_var.get()
        if name:
            self._load_config_file(CONFIGS_DIR / name)

    def _browse_config(self):
        path = filedialog.askopenfilename(
            title="Choose study config",
            initialdir=str(CONFIGS_DIR if CONFIGS_DIR.is_dir() else PROJECT_DIR),
            filetypes=[("JSON config", "*.json"), ("All files", "*.*")],
        )
        if path:
            self._load_config_file(Path(path))
            self.config_choice_var.set("")

    def _reload_config(self):
        path = self.config_path_var.get()
        if path and path != "(no config loaded)":
            self._load_config_file(Path(path))

    def _load_config_file(self, path: Path):
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            messagebox.showerror("Failed to load config", f"{path}\n\n{e}")
            return

        self.loaded_config = cfg
        self.config_path_var.set(str(path))

        self.data_root_var.set(cfg.get("data_root", ""))

        sessions = cfg.get("sessions")
        self.sessions_var.set(", ".join(sessions) if sessions else "")

        stations = cfg.get("stations") or []
        self.stations_var.set(", ".join(stations))

        for key, _label, _indent in STAGE_FLAGS:
            self.stage_vars[key].set(bool(cfg.get(key, False)))

        mode_value = str(
            cfg.get("fat_compartment_mode", DEFAULT_FAT_COMPARTMENT_MODE)
        ).strip().lower()
        if mode_value not in FAT_COMPARTMENT_MODE_LABELS:
            mode_value = DEFAULT_FAT_COMPARTMENT_MODE
        self.fat_compartment_mode_var.set(FAT_COMPARTMENT_MODE_LABELS[mode_value])
        self._update_fat_compartment_mode_state()

        self._refresh_patients()

    # ------------------------------------------------------------------
    # Data directory / patients
    # ------------------------------------------------------------------
    def _browse_data_root(self):
        current = self.data_root_var.get()
        path = filedialog.askdirectory(
            title="Choose data directory", initialdir=current if current else str(PROJECT_DIR)
        )
        if path:
            self.data_root_var.set(path)
            self._refresh_patients()

    def _refresh_patients(self):
        for child in self.patients_inner.winfo_children():
            child.destroy()
        self.patient_vars = {}

        data_root = Path(self.data_root_var.get()) if self.data_root_var.get() else None
        if not data_root or not data_root.is_dir():
            ttk.Label(self.patients_inner, text="(data directory not found)", foreground="#a00").pack(anchor="w")
            return

        entries = sorted(p.name for p in data_root.iterdir() if p.is_dir() and not p.name.startswith("."))

        # Heuristic: if this folder's own children look like DICOM export folders
        # (or it already has a "Musclemap Data" output folder), the user has almost
        # certainly pointed at a single patient's folder instead of the study folder
        # that holds one subfolder per patient. Running from here silently converts
        # nothing, because those child names aren't real subject IDs.
        looks_like_patient_folder = any("dicom" in e.lower() for e in entries) or "Musclemap Data" in entries
        if looks_like_patient_folder and data_root.parent.is_dir():
            warn = ttk.Frame(self.patients_inner)
            warn.pack(fill="x", anchor="w", pady=(0, 6))
            ttk.Label(
                warn,
                text=(
                    f"This looks like a single patient's folder (it directly contains DICOM/output\n"
                    f"folders), not a study folder. Point 'Data directory' at the parent folder that\n"
                    f"holds one subfolder per patient instead: {data_root.parent}"
                ),
                foreground="#a00",
                justify="left",
            ).pack(anchor="w")
            ttk.Button(
                warn,
                text=f"Use parent folder instead ({data_root.parent.name})",
                command=lambda: self._use_parent_as_data_root(data_root),
            ).pack(anchor="w", pady=(4, 0))

        preselect = set(self.loaded_config.get("subjects") or [])

        if not entries:
            ttk.Label(self.patients_inner, text="(no subfolders found in this directory)").pack(anchor="w")
            return

        for name in entries:
            var = tk.BooleanVar(value=(not looks_like_patient_folder) and name in preselect)
            ttk.Checkbutton(self.patients_inner, text=name, variable=var).pack(anchor="w")
            self.patient_vars[name] = var

    def _use_parent_as_data_root(self, data_root: Path):
        self.data_root_var.set(str(data_root.parent))
        self._refresh_patients()
        for name, var in self.patient_vars.items():
            if name == data_root.name:
                var.set(True)

    def _set_all_patients(self, value: bool):
        for var in self.patient_vars.values():
            var.set(value)

    def _browse_interpreter(self):
        path = filedialog.askopenfilename(
            title="Choose Python interpreter", filetypes=[("python.exe", "python.exe"), ("All files", "*.*")]
        )
        if path:
            self.interpreter_var.set(path)

    # ------------------------------------------------------------------
    # Run / stop
    # ------------------------------------------------------------------
    def _build_effective_config(self) -> dict:
        if not self.loaded_config:
            raise ValueError("Load a study config first.")

        data_root = self.data_root_var.get().strip()
        if not data_root:
            raise ValueError("Data directory is empty.")

        subjects = [name for name, var in self.patient_vars.items() if var.get()]
        if not subjects:
            raise ValueError("Select at least one patient.")

        cfg = dict(self.loaded_config)
        cfg["data_root"] = data_root
        cfg["subjects"] = subjects

        sessions_text = self.sessions_var.get().strip()
        cfg["sessions"] = [s.strip() for s in sessions_text.split(",") if s.strip()] if sessions_text else None

        stations_text = self.stations_var.get().strip()
        cfg["stations"] = [s.strip() for s in stations_text.split(",") if s.strip()] if stations_text else []

        for key, _label, _indent in STAGE_FLAGS:
            cfg[key] = self.stage_vars[key].get()

        mode_label = self.fat_compartment_mode_var.get().strip()
        cfg["fat_compartment_mode"] = FAT_COMPARTMENT_MODE_OPTIONS.get(
            mode_label,
            DEFAULT_FAT_COMPARTMENT_MODE,
        )

        return cfg

    @staticmethod
    def _estimate_progress_plan(cfg: dict):
        """Return (stage_count, total_weight) for the selected run."""
        n_subjects = len(cfg.get("subjects") or []) or 1
        n_sessions = len(cfg.get("sessions") or [None]) or 1
        n_stations = len(cfg.get("stations") or []) or 1

        stage_count = 0
        total_weight = 0.0

        # DICOM conversion is launched once with the full effective config.
        if cfg.get("run_dicom_to_nifti", False):
            stage_count += 1
            total_weight += STAGE_WEIGHTS["dicom"]

        per_station_multiplier = n_subjects * n_sessions * n_stations
        for flag, stage_key in (
            ("run_musclemap_dixon", "musclemap_dixon"),
            ("run_totalseg", "totalseg"),
            ("run_fat_compartments", "fat_compartments"),
        ):
            if cfg.get(flag):
                stage_count += per_station_multiplier
                total_weight += STAGE_WEIGHTS[stage_key] * per_station_multiplier

        # T2-448 runs once per subject/session, not once per station.
        if cfg.get("run_musclemap_t2_448"):
            multiplier = n_subjects * n_sessions
            stage_count += multiplier
            total_weight += STAGE_WEIGHTS["musclemap_t2_448"] * multiplier

        if cfg.get("run_summary", True):
            stage_count += 1
            total_weight += STAGE_WEIGHTS["summary"]

        return max(stage_count, 1), max(total_weight, 1.0)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def _on_setup_environment(self):
        if self.proc is not None or self.busy:
            return

        self._clear_log()
        self._append_log("Setting up environment (conda + packages) - this can take a while on first run...\n\n")

        # Package download sizes/speeds aren't predictable, so this stays a bouncing
        # indeterminate bar rather than a %/ETA one.
        self.progress_determinate = False
        self.progress.config(mode="indeterminate")
        self.progress_label_var.set("")

        self.busy = True
        self._set_running(True)
        threading.Thread(target=self._run_environment_setup, daemon=True).start()

    def _run_environment_setup(self):
        def on_proc(proc):
            self.proc = proc

        try:
            python_path = environment_setup.setup_environment(self.log_queue.put, on_proc=on_proc)
            self.log_queue.put(("SET_INTERPRETER", str(python_path)))
        except environment_setup.SetupError as e:
            self.log_queue.put(f"\nSetup failed: {e}\n")
        except Exception as e:
            self.log_queue.put(f"\nUnexpected error during setup: {e}\n")
        finally:
            self.proc = None
            self.busy = False
            self.log_queue.put(DONE_SENTINEL)

    def _on_run(self):
        if self.proc is not None or self.busy:
            return

        interpreter = self.interpreter_var.get().strip()
        if not interpreter or not Path(interpreter).exists():
            messagebox.showerror("Invalid interpreter", f"Python interpreter not found:\n{interpreter}")
            return

        if not WHOLEBODYSEG_SCRIPT.exists():
            messagebox.showerror("Missing script", f"Could not find WholeBodySeg.py at:\n{WHOLEBODYSEG_SCRIPT}")
            return

        try:
            cfg = self._build_effective_config()
        except ValueError as e:
            messagebox.showerror("Cannot run", str(e))
            return

        self.progress_total, self.progress_total_weight = self._estimate_progress_plan(cfg)
        self.progress_completed = 0
        self.progress_completed_weight = 0.0
        self.progress_elapsed_at_last_completion = 0.0
        self.completed_progress_events.clear()
        self.current_stage_key = ""
        self.current_stage_text = ""
        self.current_stage_var.set("Current: starting pipeline...")
        self.stop_requested = False
        self.run_start_time = time.time()
        self.progress_determinate = True
        self.progress.stop()
        self.progress.config(mode="determinate", maximum=self.progress_total_weight, value=0)
        self.progress_label_var.set(f"0% (0/{self.progress_total} stages) — elapsed 0s, estimating time remaining...")

        RUN_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = cfg.get("study_name", "study")
        run_config_path = RUN_CONFIGS_DIR / f"gui_run_{study_name}_{ts}.json"
        with open(run_config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        cmd = [interpreter, str(WHOLEBODYSEG_SCRIPT), "--config", str(run_config_path)]

        self._clear_log()
        self._append_log(f"Patients: {', '.join(cfg['subjects'])}\n")
        self._append_log(f"$ {' '.join(cmd)}\n\n")

        self.busy = True
        self._set_running(True)
        threading.Thread(target=self._run_subprocess, args=(cmd,), daemon=True).start()

    def _run_subprocess(self, cmd):
        rc = -1
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in self.proc.stdout:
                self.log_queue.put(line)
            rc = self.proc.wait()
            self.log_queue.put(f"\n--- Process exited with code {rc} ---\n")
        except Exception as e:
            self.log_queue.put(f"\nERROR launching process: {e}\n")
        finally:
            self.proc = None
            self.busy = False
            self.log_queue.put(("PROCESS_EXIT", rc))

    def _on_stop(self):
        if self.proc is not None:
            self.stop_requested = True
            self._append_log("\n--- Stopping process (and any child processes it started) ---\n")
            try:
                # WholeBodySeg.py spawns its own subprocesses (TotalSegmentator, mm_segment.py,
                # dcm2niix), which inherit our stdout pipe. self.proc.terminate() only kills the
                # top-level process, leaving those grandchildren running and the pipe open, so the
                # log-reading loop in _run_subprocess blocks until they exit on their own. Killing
                # the whole process tree closes the pipe immediately.
                subprocess.run(
                    ["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                    capture_output=True,
                )
            except Exception as e:
                self._append_log(f"Failed to stop process: {e}\n")

    def _set_running(self, running: bool):
        self.run_button.config(state="disabled" if running else "normal")
        self.setup_button.config(state="disabled" if running else "normal")
        self.stop_button.config(state="normal" if running else "disabled")
        self.status_var.set("Running..." if running else "Idle")
        if not self.progress_determinate:
            if running:
                self.progress.start(12)
            else:
                self.progress.stop()

    def _handle_progress_line(self, line: str) -> bool:
        """Consume a machine-readable WholeBodySeg progress event."""
        stripped = str(line).strip()
        if not stripped.startswith(PROGRESS_PREFIX + "|"):
            return False

        parts = stripped.split("|", 6)
        if len(parts) != 7:
            return True

        _prefix, event, stage_key, subject, session, station, label = parts
        event = event.upper().strip()
        stage_key = stage_key.strip()
        label = label.strip() or stage_key
        event_id = (stage_key, subject, session, station)

        location = " | ".join(
            part for part in (subject, session, station, label) if str(part).strip()
        )
        if event == "START":
            self.current_stage_key = stage_key
            self.current_stage_text = location or label
            self.current_stage_var.set(f"Current: {self.current_stage_text}")
            self._update_progress_display()
        elif event == "DONE":
            if event_id not in self.completed_progress_events:
                self.completed_progress_events.add(event_id)
                self.progress_completed = min(self.progress_completed + 1, self.progress_total)
                self.progress_completed_weight = min(
                    self.progress_completed_weight + STAGE_WEIGHTS.get(stage_key, 1.0),
                    self.progress_total_weight,
                )
                if self.run_start_time is not None:
                    self.progress_elapsed_at_last_completion = time.time() - self.run_start_time
                self.progress.config(value=self.progress_completed_weight)
            self._update_progress_display()

        return True

    def _update_progress_display(self):
        if not self.progress_determinate or self.run_start_time is None:
            return

        elapsed = time.time() - self.run_start_time
        pct = int(round(100 * self.progress_completed_weight / self.progress_total_weight))
        pct = min(100, max(0, pct))

        remaining_text = "estimating time remaining..."
        if self.progress_completed_weight > 0 and self.progress_elapsed_at_last_completion > 0:
            seconds_per_weight = (
                self.progress_elapsed_at_last_completion / self.progress_completed_weight
            )
            estimated_total = seconds_per_weight * self.progress_total_weight
            remaining = max(0.0, estimated_total - elapsed)
            remaining_text = f"est. {self._format_duration(remaining)} remaining"

        self.progress_label_var.set(
            f"{pct}% ({self.progress_completed}/{self.progress_total} stages) — "
            f"elapsed {self._format_duration(elapsed)}, {remaining_text}"
        )

    def _update_progress_clock(self):
        """Refresh elapsed time/ETA even while a long-running stage is active."""
        if self.progress_determinate and self.busy and self.run_start_time is not None:
            self._update_progress_display()
        self.root.after(1000, self._update_progress_clock)

    def _finish_pipeline_run(self, rc: int):
        self._set_running(False)
        elapsed = time.time() - self.run_start_time if self.run_start_time is not None else 0.0

        if self.stop_requested:
            self.status_var.set("Stopped")
            self.current_stage_var.set(
                f"Stopped during: {self.current_stage_text}" if self.current_stage_text else "Stopped"
            )
            self.progress_label_var.set(
                f"{self.progress_completed}/{self.progress_total} stages — "
                f"stopped after {self._format_duration(elapsed)}"
            )
            return

        if rc == 0:
            self.progress_completed = self.progress_total
            self.progress_completed_weight = self.progress_total_weight
            self.progress.config(value=self.progress_total_weight)
            self.status_var.set("Completed successfully")
            self.current_stage_var.set("Current: completed")
            self.progress_label_var.set(
                f"100% ({self.progress_total}/{self.progress_total} stages) — "
                f"finished after {self._format_duration(elapsed)}"
            )
        else:
            self.status_var.set(f"Failed (exit code {rc})")
            if self.current_stage_text:
                self.current_stage_var.set(f"Failed during: {self.current_stage_text}")
            else:
                self.current_stage_var.set("Current: failed before a stage started")
            self.progress_label_var.set(
                f"{self.progress_completed}/{self.progress_total} stages completed — "
                f"failed after {self._format_duration(elapsed)}"
            )

    # ------------------------------------------------------------------
    # Log widget helpers
    # ------------------------------------------------------------------
    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _append_log(self, text: str):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item is DONE_SENTINEL:
                    # Used by environment setup, which intentionally has indeterminate progress.
                    self._set_running(False)
                elif isinstance(item, tuple) and item[0] == "SET_INTERPRETER":
                    self.interpreter_var.set(item[1])
                elif isinstance(item, tuple) and item[0] == "PROCESS_EXIT":
                    self._finish_pipeline_run(int(item[1]))
                else:
                    # Hide internal progress protocol lines from the visible log.
                    if not self._handle_progress_line(item):
                        self._append_log(item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)


def main():
    root = tk.Tk()
    app = WholeBodySegGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
