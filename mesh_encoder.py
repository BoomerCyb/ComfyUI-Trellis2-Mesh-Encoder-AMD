import os
import json
import traceback
import numpy as np
import torch
import trimesh
import folder_paths
import comfy.model_management
import comfy.model_patcher
from comfy_api.latest import Types
import comfy.latent_formats

_REBUILD_HINT = " Run install_requirements.bat in this node's folder, then restart ComfyUI."


def _o_voxel_build_problem():
    """Why the installed o_voxel cannot be loaded safely, or None.

    Read before importing: a native module built for another PyTorch can crash
    the process instead of raising an ImportError.
    """
    import importlib.util
    from pathlib import Path
    spec = importlib.util.find_spec("o_voxel")
    if spec is None or not spec.origin:
        return None
    try:
        built = json.loads((Path(spec.origin).parent / "_build_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # built before build records existed
    if built.get("torch") and built["torch"] != torch.__version__:
        return f"o_voxel was built for PyTorch {built['torch']}, but ComfyUI now runs {torch.__version__}." + _REBUILD_HINT
    architectures = [arch for arch in built.get("architectures", "").split(";") if arch]
    if architectures and torch.cuda.is_available():
        current = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
        if current not in architectures:
            return f"o_voxel was built for {', '.join(architectures)}, but this GPU is {current}." + _REBUILD_HINT
    return None


_problem = _o_voxel_build_problem()
if _problem:
    o_voxel = None
    _OVOXEL_IMPORT_ERROR = RuntimeError(_problem)
else:
    try:
        import o_voxel
    except Exception as e:
        o_voxel = None
        _OVOXEL_IMPORT_ERROR = e

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:
    load_safetensors = None

_ENCODER_CACHE = {}
# Peak encoder activation memory per surface voxel (measured on gfx1201, with margin).
ENCODER_BYTES_PER_VOXEL = 1600

def _get_sparse_tensor_class():
    try:
        from trellis2.modules.sparse import SparseTensor
        return SparseTensor
    except ImportError:
        pass
    try:
        import comfy.ldm.trellis2.vae as trellis_vae
        st = getattr(trellis_vae, "SparseTensor", None)
        if st is None:
            st = getattr(trellis_vae, "TrellisSparseTensor", None)
        if st is not None:
            return st
    except ImportError:
        pass
    raise ImportError(
        "Could not locate SparseTensor in trellis2.modules.sparse "
        "or comfy.ldm.trellis2.vae"
    )

def _resolve_encoder_path(value):
    value = (value or "").strip()
    model_root = folder_paths.models_dir
    candidates = []
    if value:
        if os.path.isabs(value):
            candidates.append(value)
        candidates.append(value + ".safetensors")
        candidates += [
            os.path.join(model_root, "Trellis2", value),
            os.path.join(model_root, "Trellis2", value + ".safetensors"),
            os.path.join(model_root, "Trellis2", "encoders", value),
            os.path.join(model_root, "Trellis2", "encoders", value + ".safetensors"),
            os.path.join(model_root, value),
            os.path.join(model_root, value + ".safetensors"),
        ]
    candidates += [
        os.path.join(model_root, "Trellis2", "encoders", "shape_enc_next_dc_f16c32_fp16.safetensors"),
        os.path.join(model_root, "Trellis2", "shape_enc_next_dc_f16c32_fp16.safetensors"),
        os.path.join(model_root, "trellis2", "shape_enc_next_dc_f16c32_fp16.safetensors"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "Could not locate shape encoder safetensors file. Searched candidates:\n"
        + "\n".join(f"  - {c}" for c in candidates[:6])
    )

class _EncoderHolder(torch.nn.Module):
    """Container for ComfyUI's ModelPatcher, which assigns `model.device`; the
    TRELLIS encoder exposes `device` as a read-only property."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder


def _load_shape_encoder(path):
    path = os.path.abspath(path)
    if path in _ENCODER_CACHE:
        return _ENCODER_CACHE[path]

    # --- FIX FOR IMPORT ---
    import sys
    trellis_node_path = os.path.join(os.path.dirname(__file__), "..", "ComfyUI-Trellis2")
    if os.path.isdir(trellis_node_path) and os.path.abspath(trellis_node_path) not in sys.path:
        sys.path.insert(0, os.path.abspath(trellis_node_path))
    # ----------------------

    base_path = path[:-12] if path.lower().endswith(".safetensors") else path
    json_path = base_path + ".json"
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Config JSON not found at: {json_path}")

    import trellis2.models

    try:
        print(f"[TRELLIS2 GLB Encoder] Loading shape encoder config from {json_path}")
        with open(json_path, "r") as f:
            config = json.load(f)

        class_name = config.get("name", "FlexiDualGridVaeEncoder")
        model_args = config.get("args", {})
        print(f"[TRELLIS2 GLB Encoder] Config requests model class: '{class_name}'")

        # Try to get the requested class
        ModelClass = getattr(trellis2.models, class_name, None)

        # FIX: If the JSON name doesn't exist in trellis2.models, fallback to the known default
        if ModelClass is None:
            print(f"[TRELLIS2 GLB Encoder] Warning: Class '{class_name}' not found in trellis2.models.")
            print(f"[TRELLIS2 GLB Encoder] Falling back to default encoder: 'FlexiDualGridVaeEncoder'")
            ModelClass = getattr(trellis2.models, "FlexiDualGridVaeEncoder", None)

        if ModelClass is None:
            available = [m for m in dir(trellis2.models) if not m.startswith('_')]
            raise AttributeError(
                f"Could not find ANY encoder class in trellis2.models. Available classes: {available}"
            )

        print(f"[TRELLIS2 GLB Encoder] Instantiating model: {ModelClass.__name__} with args: {model_args}")
        encoder = ModelClass(**model_args)

        print(f"[TRELLIS2 GLB Encoder] Loading weights from {path}")
        state_dict = load_safetensors(path, device="cpu")
        encoder.load_state_dict(state_dict, strict=True)

    except Exception as e:
        traceback.print_exc()
        raise ImportError(f"Failed to load shape encoder: {e}") from e

    encoder.eval()
    # Let ComfyUI's model management load, offload and unload the encoder like
    # its own models, instead of pinning it in VRAM for the whole session.
    patcher = comfy.model_patcher.CoreModelPatcher(
        _EncoderHolder(encoder),
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
    )
    _ENCODER_CACHE[path] = patcher
    return patcher

def _load_mesh(path):
    asset = trimesh.load(path, force="scene", process=False)
    if isinstance(asset, trimesh.Trimesh):
        mesh = asset
    elif isinstance(asset, trimesh.Scene):
        parts = [
            g.copy() for g in asset.geometry.values()
            if isinstance(g, trimesh.Trimesh) and len(g.vertices) and len(g.faces)
        ]
        if not parts:
            raise ValueError("GLB contains no triangle mesh geometry.")
        mesh = trimesh.util.concatenate(parts)
    else:
        raise TypeError(f"Unsupported mesh type: {type(asset).__name__}")

    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mesh.faces = np.asarray(mesh.faces, dtype=np.int64)

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("GLB mesh is empty.")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("GLB contains non-finite vertex coordinates.")
    return mesh

def _preprocess_mesh(mesh):
    v = np.asarray(mesh.vertices, dtype=np.float32).copy()
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    center = (vmin + vmax) / 2.0
    extent = float((vmax - vmin).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Mesh has invalid/zero bounding-box extent.")

    v = (v - center) * (0.99999 / extent)
    normalized_yup = v.copy()

    w = v.copy()
    tmp = w[:, 1].copy()
    w[:, 1] = -w[:, 2]
    w[:, 2] = tmp

    work_mesh = trimesh.Trimesh(
        vertices=w,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    return work_mesh, normalized_yup

def _encode_shape_slat(mesh, resolution, encoder_patcher):
    if o_voxel is None:
        raise ImportError(
            f"o_voxel is not available in this ComfyUI environment: {_OVOXEL_IMPORT_ERROR}"
            + ("" if isinstance(_OVOXEL_IMPORT_ERROR, RuntimeError) else _REBUILD_HINT)
        ) from _OVOXEL_IMPORT_ERROR

    SparseTensor = _get_sparse_tensor_class()
    vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))
    faces = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64))

    print(f"[TRELLIS2 GLB Encoder] O-Voxel mesh conversion at {resolution}^3...")
    voxel_indices, dual_vertices, intersected = (
        o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices.cpu(),
            faces.cpu(),
            grid_size=int(resolution),
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )
    )

    # Ask ComfyUI to make room for the encoder and its activations, which scale
    # with the number of surface voxels, before anything is moved to the GPU.
    comfy.model_management.load_models_gpu(
        [encoder_patcher],
        memory_required=int(voxel_indices.shape[0]) * ENCODER_BYTES_PER_VOXEL,
        force_full_load=True,
    )
    encoder = encoder_patcher.model.encoder

    device = comfy.model_management.get_torch_device()
    coords = torch.cat(
        [torch.zeros_like(voxel_indices[:, :1]), voxel_indices], dim=-1
    ).to(torch.int32)

    feats_v = (dual_vertices * resolution - voxel_indices).to(device)
    feats_i = intersected.float().to(device)
    coords_dev = coords.to(device)

    sparse_vertices = SparseTensor(feats=feats_v, coords=coords_dev)
    sparse_intersected = SparseTensor(feats=feats_i, coords=coords_dev)

    with torch.inference_mode():
        shape_slat = encoder(
            sparse_vertices,
            sparse_intersected,
            sample_posterior=False,
        )

    if not isinstance(shape_slat, SparseTensor):
        shape_slat = SparseTensor(
            feats=shape_slat.feats,
            coords=shape_slat.coords,
        )
    return shape_slat

def _make_latent(shape_slat, resolution):
    fmt = comfy.latent_formats.Trellis2ShapeSLAT()
    normalized = fmt.process_in(shape_slat.feats)
    coords = shape_slat.coords.to(torch.int32)

    if normalized.ndim != 2 or normalized.shape[-1] != 32:
        raise ValueError(
            f"Unexpected shape latent {tuple(normalized.shape)}; expected [N,32]."
        )

    samples = normalized.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1).contiguous()
    return {
        "samples": samples,
        "coords": coords.cpu(),
        "coord_counts": torch.tensor([coords.shape[0]], dtype=torch.int64),
        "coord_resolution": int(resolution) // 16,
        "type": "trellis2",
        "model_frame": "z_up",
    }

def _make_subdivides(shape_slat, vae, resolution):
    device = comfy.model_management.get_torch_device()
    if type(vae.first_stage_model).__module__.startswith("comfy."):
        from comfy.ldm.trellis2.vae import SparseTensor
    else:
        SparseTensor = _get_sparse_tensor_class()

    raw = SparseTensor(
        feats=shape_slat.feats.to(device),
        coords=shape_slat.coords.to(device),
    )

    vae.prepare_decode(
        torch.Size([1, 32, int(shape_slat.feats.shape[0]), 1])
    )

    with torch.inference_mode():
        _mesh_unused, subs = vae.first_stage_model.decode_shape_slat(
            raw.to(vae.vae_dtype),
            int(resolution),
        )
    return subs

class Trellis2MeshEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "mesh_filename": (
                    "STRING",
                    {"default": "input_mesh.glb"},
                ),
                "shape_encoder": (
                    "STRING",
                    {"default": "shape_enc_next_dc_f16c32_fp16.safetensors"},
                ),
                "resolution": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 128},
                ),
            },
            "optional": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("LATENT", "SHAPE_SUBDIVIDES", "MESH")
    RETURN_NAMES = ("shape_latent", "shape_subdivides", "mesh")
    FUNCTION = "encode_mesh"
    CATEGORY = "TRELLIS2/Custom"

    def encode_mesh(self, vae, mesh_filename, shape_encoder, resolution, **kwargs):
        try:
            mesh_path = mesh_filename
            if not os.path.isabs(mesh_path):
                mesh_path = os.path.join(
                    folder_paths.get_input_directory(),
                    mesh_filename,
                )
            mesh_path = os.path.abspath(mesh_path)

            if not os.path.isfile(mesh_path):
                raise FileNotFoundError(f"GLB not found: {mesh_path}")

            raw_mesh = _load_mesh(mesh_path)
            work_mesh, normalized_yup = _preprocess_mesh(raw_mesh)

            print(
                f"[TRELLIS2 GLB Encoder] Mesh: "
                f"{len(raw_mesh.vertices):,} verts / {len(raw_mesh.faces):,} tris"
            )

            encoder_path = _resolve_encoder_path(shape_encoder)
            encoder_patcher = _load_shape_encoder(encoder_path)

            shape_slat = _encode_shape_slat(
                work_mesh, int(resolution), encoder_patcher
            )

            shape_latent = _make_latent(shape_slat, int(resolution))
            shape_subdivides = _make_subdivides(
                shape_slat, vae, int(resolution)
            )

            mesh_output = Types.MESH(
                vertices=torch.from_numpy(normalized_yup).unsqueeze(0),
                faces=torch.from_numpy(
                    np.asarray(raw_mesh.faces, dtype=np.int64)
                ).unsqueeze(0),
            )

            print(
                "[TRELLIS2 GLB Encoder] SUCCESS — "
                f"{int(shape_slat.coords.shape[0]):,} sparse tokens."
            )
            return (shape_latent, shape_subdivides, mesh_output)

        except Exception as exc:
            print("[TRELLIS2 GLB Encoder] ERROR:")
            traceback.print_exc()
            raise RuntimeError(
                f"TRELLIS2 GLB encoder failed: {exc}"
            ) from exc

NODE_CLASS_MAPPINGS = {
    "Trellis2MeshEncoder": Trellis2MeshEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2MeshEncoder": "Trellis2: MeshEncoder",
}