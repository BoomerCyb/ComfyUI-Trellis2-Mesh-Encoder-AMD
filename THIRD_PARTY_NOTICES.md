# Third-Party Notices

This repository contains the custom-node source code for **ComfyUI-TRELLIS2-Mesh-Encoder**. The third-party components below are **not bundled with this repository**; they are expected to be installed separately or supplied by the existing ComfyUI/TRELLIS.2 environment.

The license descriptions below are intended to make the dependency boundaries clear. Always consult the official upstream license before redistributing third-party software.

## 1. TRELLIS.2 / O-Voxel

**Project:** Microsoft TRELLIS.2  
**Repository:** https://github.com/microsoft/TRELLIS.2  
**License:** MIT  
**License text:** https://github.com/microsoft/TRELLIS.2/blob/main/LICENSE  
**O-Voxel:** https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel

The node imports TRELLIS.2 and O-Voxel APIs but does not bundle the upstream repository in this package.

Microsoft currently describes TRELLIS.2 as MIT licensed and separately notes that some optional dependencies used by the full project have their own licenses.

## 2. TRELLIS.2-4B model weights

**Model:** Microsoft `TRELLIS.2-4B`  
**Repository:** https://huggingface.co/microsoft/TRELLIS.2-4B  
**License shown by Hugging Face:** MIT

The node does **not** bundle model weights. Any downloaded weights remain subject to the model repository's own license and terms.

## 3. NumPy

**Project:** NumPy  
**Repository:** https://github.com/numpy/numpy  
**License:** BSD-3-Clause  
**License:** https://github.com/numpy/numpy/blob/main/LICENSE.txt

Used directly by the node for array handling and conversion.

## 4. trimesh

**Project:** trimesh  
**Repository:** https://github.com/mikedh/trimesh  
**License:** MIT  
**License:** https://github.com/mikedh/trimesh/blob/main/LICENSE.md

Used directly for loading and processing GLB mesh data.

## 5. safetensors

**Project:** safetensors  
**Repository:** https://github.com/huggingface/safetensors  
**License:** Apache-2.0  
**License:** https://github.com/huggingface/safetensors/blob/main/LICENSE

Used directly to load the shape encoder weights.

## 6. PyTorch

**Project:** PyTorch  
**Repository:** https://github.com/pytorch/pytorch  
**License information:** https://github.com/pytorch/pytorch/blob/main/LICENSE

The main PyTorch project uses BSD-3-Clause licensing, while the current PyTorch package metadata also identifies additional third-party license components. This repository does **not** bundle PyTorch.

Use the PyTorch installation already associated with the user's ComfyUI environment and comply with the licenses of the exact build being redistributed.

## 7. ComfyUI

**Project:** ComfyUI  
**Repository:** https://github.com/Comfy-Org/ComfyUI  
**License:** GPL-3.0  
**License:** https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE

This package is a ComfyUI custom node and relies on ComfyUI APIs. ComfyUI itself remains under its own GPL-3.0 license. This repository does not relicense ComfyUI under MIT and does not bundle the ComfyUI application.

## 8. CUDA / NVIDIA software

The node may run in environments using NVIDIA CUDA and drivers. Those components are separate software from this project and carry their own NVIDIA terms.

NVIDIA information: https://www.nvidia.com/en-us/drivers/

Do not treat CUDA, NVIDIA drivers, or an NVIDIA runtime installation as being licensed under this project's MIT license.

## Commercial-use summary

The **custom-node source in this repository is MIT licensed and commercially usable**.

The direct Python dependencies used by the node are permissively licensed as identified above.

A complete environment may additionally contain ComfyUI, PyTorch third-party components, CUDA/NVIDIA software, model weights, and other packages with separate terms. Those components are **not automatically covered by this repository's MIT license**.

For a commercial redistribution of the complete environment, audit the exact packages and versions that you ship.
