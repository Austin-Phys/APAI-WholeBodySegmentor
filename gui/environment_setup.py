"""
environment_setup.py

One-click, no-terminal-required setup of the Python environment WholeBodySeg.py
needs (pydicom, nibabel, pandas, torch, TotalSegmentator, MONAI, etc.).

Mirrors the package set in environment_wholebodyseg_gpu.yml (the author's exported
working environment), but:
- Installs PyTorch from the correct index (PyPI does not host the +cuXXX build tags),
  picking a CUDA build if an NVIDIA GPU is detected, otherwise a CPU-only build.
- Creates/reuses a conda environment named "wholebodyseg" via Miniconda, installing
  Miniconda itself first if no conda installation can be found anywhere on the machine.

Called from the GUI (wholebodyseg_gui.py); can also be run standalone:
    python gui/environment_setup.py
"""

import json
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

ENV_NAME = "wholebodyseg"
PYTHON_VERSION = "3.10"

MINICONDA_INSTALLER_URL = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
DEFAULT_MINICONDA_DIR = Path.home() / "miniconda3"

TORCH_VERSION = "2.5.1"
TORCHVISION_VERSION = "0.20.1"
TORCHAUDIO_VERSION = "2.5.1"

# Everything else from environment_wholebodyseg_gpu.yml's pip section (torch/vision/audio
# are installed separately above, since they need a non-default index URL).
PIP_REQUIREMENTS: List[str] = [
    "acvl-utils==0.2.6",
    "batchgenerators==0.25.3",
    "batchgeneratorsv2==0.3.3",
    "blosc2==4.3.3",
    "connected-components-3d==4.0.0",
    "contourpy==1.3.2",
    "cycler==0.12.1",
    "dicom2nifti==2.6.2",
    "dynamic-network-architectures==0.4.4",
    "einops==0.8.2",
    "fonttools==4.63.0",
    "graphviz==0.21",
    "imagecodecs==2025.3.30",
    "imageio==2.37.3",
    "joblib==1.5.3",
    "kiwisolver==1.5.0",
    "lazy-loader==0.5",
    "matplotlib==3.10.9",
    "monai==1.6.0",
    "nibabel==5.4.2",
    "nnunetv2==2.8.0",
    "numexpr==2.14.1",
    "numpy==2.2.6",
    "pandas==2.3.3",
    "pillow==12.2.0",
    "psutil==7.2.2",
    "pydicom==3.0.2",
    "python-gdcm==3.2.6",
    "pyyaml==6.0.3",
    "scikit-image==0.25.2",
    "scikit-learn==1.7.2",
    "scipy==1.15.3",
    "seaborn==0.13.2",
    "simpleitk==2.5.5",
    "timm==1.0.22",
    "totalsegmentator==2.14.0",
    "tqdm==4.68.3",
]

LogFn = Callable[[str], None]
ProcHook = Callable[[Optional[subprocess.Popen]], None]


class SetupError(RuntimeError):
    pass


def _run(cmd: List[str], log: LogFn, on_proc: Optional[ProcHook] = None, cwd: Optional[str] = None):
    log(f"$ {' '.join(str(c) for c in cmd)}\n")
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    if on_proc:
        on_proc(proc)
    try:
        for line in proc.stdout:
            log(line)
        rc = proc.wait()
    finally:
        if on_proc:
            on_proc(None)
    if rc != 0:
        raise SetupError(f"Command failed (exit {rc}): {' '.join(str(c) for c in cmd)}")


def has_nvidia_gpu() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        result = subprocess.run([exe], capture_output=True, text=True, timeout=15)
        return result.returncode == 0
    except Exception:
        return False


def find_conda_exe() -> Optional[Path]:
    env_conda = os.environ.get("CONDA_EXE")
    if env_conda and Path(env_conda).exists():
        return Path(env_conda)

    # shutil.which() only sees what's on THIS process's PATH. A GUI launched by
    # double-clicking run_gui.bat inherits the plain registry-persisted PATH, which
    # commonly does NOT include a conda install's Scripts dir (the installer only adds
    # it if the user opted in, and "conda init" only wires up shell profiles, not the
    # registry PATH) — so this frequently returns nothing even when conda is present.
    which_conda = shutil.which("conda")
    if which_conda:
        return Path(which_conda)

    roots = [
        Path.home(),
        Path(os.environ.get("PROGRAMDATA", "C:/ProgramData")),
        Path(os.environ.get("LOCALAPPDATA", "")),
        Path("C:/"),
    ]
    names = ["miniconda3", "anaconda3"]
    for root in roots:
        for name in names:
            c = root / name / "Scripts" / "conda.exe"
            if c.exists():
                return c
    return None


def install_miniconda(log: LogFn, on_proc: Optional[ProcHook] = None) -> Path:
    if platform.system() != "Windows":
        raise SetupError("Automatic Miniconda install is only implemented for Windows.")

    log("No conda installation found. Downloading Miniconda (one-time, ~100 MB)...\n")
    tmp_dir = Path(tempfile.mkdtemp(prefix="miniconda_dl_"))
    installer_path = tmp_dir / "Miniconda3-latest-Windows-x86_64.exe"

    last_pct = {"v": -1}

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            if pct != last_pct["v"] and pct % 10 == 0:
                last_pct["v"] = pct
                log(f"  downloading Miniconda... {pct}%\n")

    urllib.request.urlretrieve(MINICONDA_INSTALLER_URL, str(installer_path), _progress)
    log("Download complete. Installing Miniconda silently (a few minutes)...\n")

    install_dir = DEFAULT_MINICONDA_DIR
    cmd = [
        str(installer_path),
        "/InstallationType=JustMe",
        "/AddToPath=0",
        "/RegisterPython=0",
        "/S",
        f"/D={install_dir}",
    ]
    _run(cmd, log, on_proc=on_proc)

    conda_exe = install_dir / "Scripts" / "conda.exe"
    if not conda_exe.exists():
        raise SetupError(f"Miniconda install finished but conda.exe was not found at {conda_exe}")
    log(f"Miniconda installed at {install_dir}\n")
    return conda_exe


def _list_envs(conda_exe: Path) -> dict:
    """Map env name -> env dir Path, as conda itself reports them.

    Don't assume envs live under <conda_root>/envs: on a machine-wide (ProgramData)
    install, a non-admin user can't write there, so conda silently creates new envs
    under the user's own %USERPROFILE%\\.conda\\envs instead.
    """
    result = subprocess.run([str(conda_exe), "env", "list", "--json"], capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    try:
        data = json.loads(result.stdout)
    except Exception:
        return {}
    return {Path(e).name: Path(e) for e in data.get("envs", [])}


def _env_python_path(conda_exe: Path, env_name: str) -> Optional[Path]:
    env_dir = _list_envs(conda_exe).get(env_name)
    return (env_dir / "python.exe") if env_dir else None


def _env_exists(conda_exe: Path, env_name: str) -> bool:
    return env_name in _list_envs(conda_exe)


def find_existing_env_python() -> Optional[Path]:
    """Return the wholebodyseg env's python.exe if it's already been built, else None.

    Used by the GUI to default the interpreter field to the right env instead of
    whatever interpreter happened to launch the GUI itself.
    """
    conda_exe = find_conda_exe()
    if conda_exe is None:
        return None
    py = _env_python_path(conda_exe, ENV_NAME)
    return py if py and py.exists() else None


def setup_environment(log: LogFn, on_proc: Optional[ProcHook] = None) -> Path:
    """Ensure a conda environment with all WholeBodySeg dependencies exists.

    Safe to call repeatedly: reuses the existing conda install / env / already-installed
    packages where possible, so re-running after a partial failure just resumes.

    Returns the path to that environment's python.exe.
    """
    conda_exe = find_conda_exe()
    if conda_exe is None:
        conda_exe = install_miniconda(log, on_proc=on_proc)
    else:
        log(f"Found existing conda install: {conda_exe}\n")

    if _env_exists(conda_exe, ENV_NAME):
        log(f"Conda environment '{ENV_NAME}' already exists — reusing it.\n")
    else:
        log(f"Creating conda environment '{ENV_NAME}' (python={PYTHON_VERSION})...\n")
        _run(
            [str(conda_exe), "create", "-y", "-n", ENV_NAME, f"python={PYTHON_VERSION}"],
            log,
            on_proc=on_proc,
        )

    py = _env_python_path(conda_exe, ENV_NAME)
    if py is None or not py.exists():
        raise SetupError(
            f"Could not find python.exe for conda environment '{ENV_NAME}' after creation "
            f"(conda env list --json did not report it, or its python.exe is missing)."
        )

    log("Upgrading pip...\n")
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip"], log, on_proc=on_proc)

    gpu = has_nvidia_gpu()
    log(f"NVIDIA GPU detected: {gpu}\n")
    torch_index = "https://download.pytorch.org/whl/cu121" if gpu else "https://download.pytorch.org/whl/cpu"
    torch_pins = [f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}", f"torchaudio=={TORCHAUDIO_VERSION}"]

    log(f"Installing PyTorch ({'CUDA 12.1 GPU' if gpu else 'CPU-only'} build) — this is the largest download...\n")
    _run([str(py), "-m", "pip", "install", *torch_pins, "--index-url", torch_index], log, on_proc=on_proc)

    log("Installing remaining dependencies (pydicom, nibabel, TotalSegmentator, MONAI, etc.)...\n")
    _run([str(py), "-m", "pip", "install", *PIP_REQUIREMENTS], log, on_proc=on_proc)

    # The install above has no upper bound on torch (nnunetv2/timm/monai pull it in
    # transitively unpinned), so pip's resolver is free to "upgrade" it to satisfy
    # some other package's constraint — and since that resolution isn't pointed at
    # the CUDA wheel index, it silently swaps in a CPU-only build from PyPI instead,
    # even though the command reports success. Reassert the exact CUDA build as the
    # final step so it wins regardless of what the previous install did to it.
    log("Re-asserting PyTorch CUDA build (in case other packages pulled in a CPU-only torch)...\n")
    _run(
        [str(py), "-m", "pip", "install", "--force-reinstall", "--no-deps", *torch_pins, "--index-url", torch_index],
        log,
        on_proc=on_proc,
    )

    log("\nEnvironment setup complete.\n")
    return py


if __name__ == "__main__":
    import sys

    def _print_log(msg: str):
        sys.stdout.write(msg)
        sys.stdout.flush()

    try:
        python_path = setup_environment(_print_log)
        print(f"\nDone. Use this interpreter in the GUI: {python_path}")
    except SetupError as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)
