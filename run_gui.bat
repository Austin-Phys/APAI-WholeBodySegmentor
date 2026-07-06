@echo off
cd /d "%~dp0"

set "LAUNCHER_PY=C:\ProgramData\anaconda3\python.exe"
if not exist "%LAUNCHER_PY%" set "LAUNCHER_PY=%LOCALAPPDATA%\miniconda3\python.exe"
if not exist "%LAUNCHER_PY%" set "LAUNCHER_PY=%USERPROFILE%\miniconda3\python.exe"
if not exist "%LAUNCHER_PY%" set "LAUNCHER_PY=%USERPROFILE%\anaconda3\python.exe"

if not exist "%LAUNCHER_PY%" (
    echo Could not find a conda Python install to launch the GUI with.
    echo Looked for: C:\ProgramData\anaconda3\python.exe, %%LOCALAPPDATA%%\miniconda3\python.exe,
    echo             %%USERPROFILE%%\miniconda3\python.exe, %%USERPROFILE%%\anaconda3\python.exe
    echo Install Miniconda from https://www.anaconda.com/download or edit run_gui.bat to point at your python.exe.
    pause
    exit /b 1
)

"%LAUNCHER_PY%" gui\wholebodyseg_gui.py
if errorlevel 1 pause
