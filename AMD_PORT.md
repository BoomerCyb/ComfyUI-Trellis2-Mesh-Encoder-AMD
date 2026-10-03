# AMD changes

O-Voxel sources use HIP with Windows C++ compatibility fixes. The encoder
uses ComfyUI sparse convolution and compatible VAE tensors. Model weights are
loaded strictly. Full image-generation and rendering pipelines are outside
this mesh-encoder port.

Installation uses ComfyUI's ROCm Python through install.py. Native extensions
use PyTorch HIP device properties to target discrete GPUs by default, while
respecting PYTORCH_ROCM_ARCH. Integrated-only systems target their available GPUs.
Tested on RX 9070 XT; other supported ROCm devices require validation.
Original licenses, algorithms and node registrations are preserved.
