from pathlib import Path
import datetime
import shutil
import sysconfig
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


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
                shutil.move(str(destination), str(old))
            changed.append((destination, old))
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__"))
        subprocess.check_call([sys.executable, "-c", "import torch; import o_voxel._C, trellis2; print('Installed runtime loaded')"])
    except Exception:
        for destination, old in reversed(changed):
            if destination.exists():
                shutil.rmtree(destination)
            if old.exists():
                shutil.move(str(old), str(destination))
        raise
    print("Previous runtime folders, if present, were backed up to:", backup)


if __name__ == "__main__":
    main()
