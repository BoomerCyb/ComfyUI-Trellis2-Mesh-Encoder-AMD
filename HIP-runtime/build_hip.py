from pathlib import Path
import json, os, shutil, sys
import torch
if not torch.version.hip:
    raise RuntimeError('Use the existing ROCm PyTorch interpreter.')
from torch.utils.cpp_extension import load
root = Path(__file__).resolve().parent / 'o-voxel'
sources = ['src/hash/hash.cu', 'src/convert/flexible_dual_grid.cpp', 'src/convert/volumetic_attr.cpp', 'src/serialize/api.cu', 'src/serialize/hilbert.cu', 'src/serialize/z_order.cu', 'src/io/svo.cpp', 'src/io/filter_parent.cpp', 'src/io/filter_neighbor.cpp', 'src/rasterize/rasterize.cu', 'src/ext.cpp']
sources = [str(Path(s).with_suffix('.hip')) if s.endswith('.cu') else str(Path(s).with_name(Path(s).stem + '_hip.cpp')) if (root / Path(s).with_name(Path(s).stem + '_hip.cpp')).exists() else s for s in sources]
build = root / '.build'
build.mkdir(exist_ok=True)
incs = [root / 'third_party/eigen']
m = load(name='_C', sources=[str(root / s) for s in sources], extra_include_paths=[str(p) for p in incs], build_directory=str(build), with_cuda=True, extra_cflags=['-O2'], extra_cuda_cflags=['-O2', '-std=c++20'], verbose=True)
shutil.copy2(m.__file__, root / 'o_voxel/_C.pyd')
# Recorded so the node can refuse a build made for another PyTorch or GPU before importing it.
(root / 'o_voxel/_build_info.json').write_text(json.dumps({
    'torch': torch.__version__, 'hip': torch.version.hip,
    'architectures': os.environ.get('PYTORCH_ROCM_ARCH', ''), 'python': sys.version.split()[0],
}, indent=2), encoding='utf-8')
print('BUILT O-VOXEL HIP', flush=True)
