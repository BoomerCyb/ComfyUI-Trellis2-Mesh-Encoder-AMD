from pathlib import Path
import datetime
import shutil
import sysconfig
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
# Written into each installed runtime folder, to recognize packages this installer owns.
MARKER = ".installed-by-trellis2-mesh-encoder-amd"
KEEP_BACKUPS = 3


def _prune_backups():
    backups = sorted(p for p in (ROOT / ".build/runtime-backups").iterdir() if p.is_dir())
    for old in backups[:-KEEP_BACKUPS]:
        shutil.rmtree(old, ignore_errors=True)


def main():
    import torch
    if not torch.version.hip:
        raise RuntimeError("Use the ROCm PyTorch environment that runs ComfyUI.")
    if not (ROOT / "o-voxel/o_voxel/_C.pyd").is_file():
        raise RuntimeError("Run build_hip.py first.")
    target = Path(sysconfig.get_paths()["platlib"]).resolve()
    sources = [ROOT / "trellis2", ROOT / "o-voxel/o_voxel"]
    backup = ROOT / ".build/runtime-backups" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup.mkdir(parents=True)
    changed = []
    try:
        for source in sources:
            destination = target / source.name
            if destination.is_symlink() or destination.resolve().parent != target:
                raise RuntimeError("Unexpected runtime destination: " + str(destination))
            old = backup / source.name
            if destination.exists():
                if not (destination / MARKER).exists():
                    print(f"Replacing a {source.name} package that this node did not install "
                          f"(another node or pip may own it); it is backed up to {old}.", flush=True)
                shutil.move(str(destination), str(old))
            changed.append((destination, old))
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__"))
            (destination / MARKER).write_text("Installed by ComfyUI-Trellis2-Mesh-Encoder-AMD HIP-runtime/install_runtime.py\n")
        subprocess.check_call([sys.executable, "-c", "import torch; import o_voxel._C, trellis2; print('Installed runtime loaded')"])
    except Exception:
        for destination, old in reversed(changed):
            if destination.exists():
                shutil.rmtree(destination)
            if old.exists():
                shutil.move(str(old), str(destination))
        raise
    print("Previous runtime folders, if present, were backed up to:", backup)
    _prune_backups()


if __name__ == "__main__":
    main()
