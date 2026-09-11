@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHON="
if exist "%~dp0..\..\python_embeded\python.exe" set "PYTHON=%~dp0..\..\python_embeded\python.exe"
if not defined PYTHON if exist "%~dp0..\..\venv\Scripts\python.exe" set "PYTHON=%~dp0..\..\venv\Scripts\python.exe"
if not defined PYTHON if exist "%~dp0..\..\python.exe" set "PYTHON=%~dp0..\..\python.exe"
if not defined PYTHON where py >nul 2>&1 && set "PYTHON=py -3"
if not defined PYTHON where python >nul 2>&1 && set "PYTHON=python"
if not defined PYTHON (
  echo ERROR: Python not found.
  pause
  exit /b 1
)
echo Using: %PYTHON%
echo.
%PYTHON% -c "import sys; print('Python:', sys.version)"
%PYTHON% -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda)"
%PYTHON% -c "import folder_paths; print('ComfyUI folder_paths: OK')"
%PYTHON% -c "import comfy.model_management; from comfy_api.latest import Types; print('ComfyUI APIs: OK')"
%PYTHON% -c "import numpy, trimesh, safetensors; print('Direct deps: OK')"
%PYTHON% -c "import trellis2; print('TRELLIS.2: OK')" || echo TRELLIS.2: MISSING
%PYTHON% -c "import o_voxel; print('O-Voxel: OK')" || echo O-Voxel: MISSING
%PYTHON% -c "import importlib.util; print('Node source files: OK' if importlib.util.find_spec('mesh_encoder') or __import__('os').path.isfile('mesh_encoder.py') else 'Node source files: CHECK')"
echo.
echo Done.
pause
