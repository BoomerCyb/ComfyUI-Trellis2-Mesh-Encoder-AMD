from pathlib import Path
from setuptools import setup, find_packages, Distribution
import torch
if not torch.version.hip: raise RuntimeError("Use the existing ROCm PyTorch environment.")
if not (Path(__file__).parent/'o_voxel/_C.pyd').exists(): raise RuntimeError("Build with ../build_hip.py first.")
class NativeDistribution(Distribution):
    def has_ext_modules(self): return True
setup(packages=find_packages(),package_data={'o_voxel':['*.pyd']},distclass=NativeDistribution)
