@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo TRELLIS2 GLB Encoder - Windows Installer
echo ============================================================

echo.
echo Detecting the Python environment used by ComfyUI...

set "PYTHON="
if exist "%~dp0..\..\python_embeded\python.exe" set "PYTHON=%~dp0..\..\python_embeded\python.exe"
if not defined PYTHON if exist "%~dp0..\..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\..\venv\Scripts\python.exe"
if not defined PYTHON if exist "%~dp0..\..\python.exe" set "PYTHON=%~dp0..\..\python.exe"
if not defined PYTHON if exist "%~dp0..\..\python_embeded\python.exe" set "PYTHON=%~dp0..\..\python_embeded\python.exe"
if not defined PYTHON where py >nul 2>&1 && set "PYTHON=py -3"
if not defined PYTHON where python >nul 2>&1 && set "PYTHON=python"

if not defined PYTHON (
  echo.
  echo ERROR: Could not find a Python interpreter.
  echo Put this folder under ComfyUI\custom_nodes\ and run this file again,
  echo or install Python and ensure it is available on PATH.
  pause
  exit /b 1
)

echo Using: %PYTHON%
echo.
echo Installing direct Python dependencies...
%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo ERROR: Dependency installation failed.
  pause
  exit /b 1
)

echo.
echo Checking core imports...
%PYTHON% -c "import numpy, trimesh, safetensors; print('numpy', numpy.__version__); print('trimesh', trimesh.__version__); print('safetensors', getattr(safetensors, '__version__', 'unknown'))"
if errorlevel 1 (
  echo ERROR: One or more direct dependencies could not be imported.
  pause
  exit /b 1
)

echo.
echo Checking ComfyUI / TRELLIS.2 / O-Voxel imports...
%PYTHON% -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
%PYTHON% -c "import folder_paths; print('ComfyUI folder_paths: OK')"
%PYTHON% -c "import trellis2; print('trellis2: OK')"
if errorlevel 1 echo WARNING: trellis2 is not importable in this Python environment.
%PYTHON% -c "import o_voxel; print('o_voxel: OK')"
if errorlevel 1 echo WARNING: o_voxel is not importable in this Python environment.

%PYTHON% -c "import comfy.model_management; from comfy_api.latest import Types; print('ComfyUI APIs used by this node: OK')"
if errorlevel 1 echo WARNING: Required ComfyUI APIs are not importable.

echo.
echo ============================================================
echo Installation step finished.
echo ============================================================
echo Restart ComfyUI before testing the node.
echo If TRELLIS.2 or O-Voxel is missing, see THIRD_PARTY_NOTICES.md.
echo.
pause
exit /b 0
