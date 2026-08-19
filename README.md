# WholeBodySeg

A pipeline for processing whole-body MRI studies: converting DICOM to NIfTI, running muscle/fat segmentation (via [MuscleMap](https://github.com/MuscleMap/MuscleMap) and [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), computing fat-compartment metrics, and building summary tables — driven by a config file or a Tkinter GUI.

## Features

- **DICOM → NIfTI conversion** using `dcm2niix`, including split upper/lower Dixon stations
- **MuscleMap segmentation** (Dixon and T2-448 models)
- **TotalSegmentator** integration, including trunk cavity, abdominal muscle, and tissue-type sub-tasks
- **Fat compartment analysis** (IMAT / SAT / VAT thresholds, fascia dilation/closing, erosion)
- **Summary CSV generation** across subjects/sessions
- **Tkinter GUI** (`gui/wholebodyseg_gui.py`) for picking a study config, choosing subjects, toggling pipeline stages, and watching live log output

## Repository layout

```
WholeBodySeg.py           Main CLI entry point (runs a pipeline from a config JSON)
configs/                  Per-study config files (ASHA, PAD, U54, template)
scripts/                  Pipeline stages (DICOM->NIfTI, MuscleMap, TotalSegmentator, fat compartments, summary)
conversion/               Standalone DICOM->NIfTI conversion scripts
gui/                      Tkinter GUI front-end for WholeBodySeg.py
tools/                    Bundled dcm2niix.exe
environment_wholebodyseg_gpu.yml   Conda environment spec (GPU / CUDA 12.1)
run_gui.bat               Windows launcher for the GUI
```

Not included in this repo (see `.gitignore`):
- `data/` — subject imaging data (kept local only)
- `MuscleMap/` — clone separately from [MuscleMap/MuscleMap](https://github.com/MuscleMap/MuscleMap)
- `gui/run_configs/` — auto-generated run logs

## Setup

1. Create the conda environment:
   ```
   conda env create -f environment_wholebodyseg_gpu.yml
   conda activate wholebodyseg_gpu
   ```
2. Clone MuscleMap alongside this project:
   ```
   git clone https://github.com/MuscleMap/MuscleMap.git
   ```
3. Copy `configs/config_template.json`, and update `data_root`, `dcm2niix_path`, `musclemap_repo`, and `code_dir` to match your machine's paths.

## Usage

**CLI:**
```
python WholeBodySeg.py --config configs/config_template.json
```

**GUI:**
```
python gui/wholebodyseg_gui.py
```
or, on Windows, double-click `run_gui.bat`.

Each config JSON controls which subjects/sessions/stations to process and which pipeline stages to run (DICOM conversion, MuscleMap Dixon/T2-448, TotalSegmentator + sub-tasks, fat compartments, summary build) along with their thresholds.
