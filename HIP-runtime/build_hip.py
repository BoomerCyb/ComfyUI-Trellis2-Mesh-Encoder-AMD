from pathlib import Path
import os, shutil
import torch
if not torch.version.hip:
    raise RuntimeError('Use the existing ROCm PyTorch interpreter.')
from torch.utils.cpp_extension import load
root = Path(__file__).resolve().parent / 'o-voxel'
sources = ['src/hash/hash.cu', 'src/convert/flexible_dual_grid.cpp', 'src/convert/volumetic_attr.cpp', 'src/serialize/api.cu', 'src/serialize/hilbert.cu', 'src/serialize/z_order.cu', 'src/io/svo.cpp', 'src/io/filter_parent.cpp', 'src/io/filter_neighbor.cpp', 'src/rasterize/rasterize.cu', 'src/ext.cpp']
sources = [str(Path(s).with_suffix('.hip')) if s.endswith('.cu') else str(Path(s).with_name(Path(s).stem + '_hip.cpp')) if (root / Path(s).with_name(Path(s).stem + '_hip.cpp')).exists() else s for s in sources]
build = root / '.build'
build.mkdir(exist_ok=True)
incs = [root / 'third_party/eigen', root / 'third_party/amd/hipCUB', root / 'third_party/amd/rocPRIM', root / 'third_party/amd/rocThrust']
m = load(name='_C', sources=[str(root / s) for s in sources], extra_include_paths=[str(p) for p in incs], build_directory=str(build), with_cuda=True, extra_cflags=['-O2'], extra_cuda_cflags=['-O2', '-std=c++20'], verbose=True)
shutil.copy2(m.__file__, root / 'o_voxel/_C.pyd')
print('BUILT O-VOXEL HIP', flush=True)
