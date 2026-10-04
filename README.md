# ComfyUI-Trellis2-Mesh-Encoder-AMD

## AMD / ROCm

The installer uses ComfyUI's Python and stops if setup fails. When installing through EZi, wait for the entire node group to complete before restarting.

This fork keeps the original node interface and adds native HIP support.
Use the ROCm PyTorch installation that runs ComfyUI and a matching HIP SDK.
The source does not select a card model or impose a gfx1201 target. Native
extensions target visible discrete AMD GPUs by default, excluding integrated GPUs
when a discrete GPU is available. Set PYTORCH_ROCM_ARCH to override the targets.
Integrated-only systems remain supported; PyTorch supplies the compiler flags.
An installed binary still needs to match its GPU target, Python and Torch runtime.
Hardware support depends on ROCm/PyTorch; validation here covers RX 9070 XT.

Run `install_requirements.bat` with ComfyUI closed to build/install the native components.
For prerequisites and manual commands, see [COMFYUI_ROCM_BUILD_GUIDE.md](COMFYUI_ROCM_BUILD_GUIDE.md).

ComfyUI AMD installer: [BoomerCyb/ComfyUI-Easy-Install-AMD](https://github.com/BoomerCyb/ComfyUI-Easy-Install-AMD).


A lightweight ComfyUI custom node for taking an existing GLB mesh, converting it into the **TRELLIS.2 O-Voxel Flexible Dual Grid** representation, encoding its shape into a **TRELLIS.2 Shape SLAT**, and returning ComfyUI-compatible outputs for downstream 3D workflows.

This project is intended as a focused mesh-to-TRELLIS.2 bridge. It does **not** bundle the TRELLIS.2 model weights, ComfyUI, PyTorch, CUDA, or the O-Voxel compiled extension.

## What it does

**Node:** `Trellis2: MeshEncoder`

**Input**
- A GLB mesh from ComfyUI's input directory (or an absolute path)
- A TRELLIS.2 shape encoder `.safetensors` file
- The matching encoder `.json` configuration
- TRELLIS.2 resolution (`256`–`2048`, default `1024`)
- A ComfyUI VAE

**Output**
- `shape_latent` — TRELLIS.2 Shape SLAT latent
- `shape_subdivides` — subdivision information produced through the supplied VAE
- `mesh` — the normalized ComfyUI `MESH` representation

The mesh is normalized and re-oriented so that the returned geometry is aligned with the TRELLIS.2 voxel/color frame used by this node.

## Why this node exists

TRELLIS.2 is designed around O-Voxel / Flexible Dual Grid geometry. This node provides a simple way to take an existing GLB and feed its geometry into that representation without requiring a full standalone TRELLIS.2 application.

The node also includes compatibility handling for different TRELLIS.2 / ComfyUI `SparseTensor` locations and loads the shape encoder from its accompanying JSON configuration.

## Requirements

The node expects these components to already exist in the same Python environment used by ComfyUI:

- ComfyUI
- PyTorch (use the build already installed for your ComfyUI)
- TRELLIS.2 Python package
- O-Voxel (`o_voxel`) compiled extension
- NumPy
- trimesh
- safetensors

## Installation

1. Place this repository in `ComfyUI/custom_nodes/ComfyUI-Trellis2-Mesh-Encoder-AMD`.
2. Close ComfyUI and run `install_requirements.bat` using ComfyUI's Python.
3. Restart ComfyUI after installation completes. With the EZi group add-on, wait for all five nodes to finish.

You can also install this node through **Easy Menu → Add-ons → BoomerCyb WTiVo AMD Nodes** in [ComfyUI-Easy-Install-AMD](https://github.com/BoomerCyb/ComfyUI-Easy-Install-AMD).

## Model files

This node does **not** ship with model weights.

For the default encoder, place the matching pair somewhere under your ComfyUI models directory, for example:

```text
ComfyUI/
└── models/
    └── Trellis2/
        └── encoders/
            ├── shape_enc_next_dc_f16c32_fp16.safetensors
            └── shape_enc_next_dc_f16c32_fp16.json
```

The node also accepts an absolute path to the `.safetensors` file.

The `.json` file must sit next to the `.safetensors` file and use the same base filename.

## Compatibility

This is a **ComfyUI custom node**, not a replacement for the full TRELLIS.2 repository.

The node imports ComfyUI APIs plus TRELLIS.2 / O-Voxel components. Its purpose is to add a mesh-encoding step to an existing ComfyUI TRELLIS.2 environment.

## Licensing and commercial use

### This project

The original code in this repository is released under the **MIT License**.

MIT permits commercial use, modification, redistribution, sublicensing, and sale, provided the applicable copyright and license notice is retained.

See [`LICENSE`](LICENSE).

### Third-party software

This repository does **not** bundle the third-party dependencies listed below. They are installed separately or are already present in the user's ComfyUI environment.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the license status and official upstream links.

### TRELLIS.2 and O-Voxel

Microsoft's TRELLIS.2 repository is MIT licensed, and its O-Voxel component is part of that project. The official project also states that some other TRELLIS.2 dependencies have separate licenses, so a full TRELLIS.2 installation should not be treated as if every component had one identical license.

This custom node itself does not bundle those separate components.

### Model weights

The official `microsoft/TRELLIS.2-4B` model repository on Hugging Face is currently marked **MIT**. This node does not bundle those weights; obtain and use them separately under their own license terms.

### Important commercial-use distinction

This project is designed to be commercially usable. However, the phrase **"fully free"** should not be interpreted as "every component of a complete TRELLIS.2 + ComfyUI + CUDA installation is MIT." ComfyUI, PyTorch, CUDA/NVIDIA components, and other optional packages retain their own licenses and terms.

Do not redistribute those external components under this project's MIT license.

## Official upstream links

- **This project:** https://github.com/Mstafa-awad/ComfyUI-Trellis2-Mesh-Encoder
- **TRELLIS.2:** https://github.com/microsoft/TRELLIS.2
- **TRELLIS.2 License:** https://github.com/microsoft/TRELLIS.2/blob/main/LICENSE
- **TRELLIS.2-4B model:** https://huggingface.co/microsoft/TRELLIS.2-4B
- **TRELLIS.2-4B model license:** https://huggingface.co/microsoft/TRELLIS.2-4B
- **ComfyUI:** https://github.com/Comfy-Org/ComfyUI
- **ComfyUI License:** https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE
- **O-Voxel documentation:** https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel
- **NumPy:** https://github.com/numpy/numpy
- **NumPy License:** https://github.com/numpy/numpy/blob/main/LICENSE.txt
- **trimesh:** https://github.com/mikedh/trimesh
- **trimesh License:** https://github.com/mikedh/trimesh/blob/main/LICENSE.md
- **safetensors:** https://github.com/huggingface/safetensors
- **safetensors License:** https://github.com/huggingface/safetensors/blob/main/LICENSE
- **PyTorch:** https://github.com/pytorch/pytorch
- **PyTorch licensing information:** https://github.com/pytorch/pytorch/blob/main/LICENSE

## Disclaimer

This README is a practical software-licensing summary, not legal advice. Third-party licenses and terms should be reviewed before redistributing a complete bundled runtime, installer, Docker image, or model package.

## AMD Edition Changes - 2026-10-03

- Uses ComfyUI's Python and reports installation failures before restarting.
- Builds native HIP extensions for the active ROCm environment; matching HIP SDK and Visual Studio C++ Build Tools are required.
- Supports group installation through [ComfyUI-Easy-Install-AMD](https://github.com/BoomerCyb/ComfyUI-Easy-Install-AMD).

Original node by [Mstafa-awad / MostAadTech](https://github.com/Mstafa-awad). AMD fork maintained by [BoomerCyb](https://github.com/BoomerCyb). Original license and third-party credits are retained.
