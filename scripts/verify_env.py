"""Environment check for the GPU training box, Colab or a laptop (mini sprint 1.2).

Bu dosya ne işe yarar? Kurulumun çalıştığını 30 saniyede kanıtlar: PyTorch sürümü, CUDA (GPU)
görünüyor mu, karışık hassasiyet (AMP) matris çarpımı çalışıyor mu, OpenSlide sentetik bir
piramit TIFF'i (ya da verilen gerçek slaytı) açabiliyor mu, h5py/OpenCV/timm yükleniyor mu.
Hata varsa sıfırdan farklı çıkış kodu verir; ``--require-cuda`` ile GPU yoksa da hata verir.

Usage:
    uv run --group histo python scripts/verify_env.py
    uv run --group histo python scripts/verify_env.py --require-cuda --slide /data/x.svs
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
import tempfile
import time
from pathlib import Path


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'OK ' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def check_imports() -> bool:
    ok = True
    for module in ("numpy", "torch", "h5py", "cv2", "openslide", "tifffile", "timm", "mlflow"):
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "?")
            ok &= _check(f"import {module}", True, f"version {version}")
        except Exception as exc:  # report, keep checking the rest
            ok &= _check(f"import {module}", False, repr(exc))
    return ok


def check_torch(require_cuda: bool) -> bool:
    import torch

    ok = True
    cuda = torch.cuda.is_available()
    detail = f"torch {torch.__version__}, cuda available={cuda}"
    if cuda:
        detail += f", device={torch.cuda.get_device_name(0)}, cuda {torch.version.cuda}"
        free, total = torch.cuda.mem_get_info()
        detail += f", memory {total / 1e9:.1f} GB ({free / 1e9:.1f} GB free)"
    ok &= _check("torch.cuda", cuda or not require_cuda, detail)
    device = "cuda" if cuda else "cpu"
    dtype = torch.float16 if cuda else torch.bfloat16
    try:
        a = torch.randn(512, 512, device=device)
        started = time.perf_counter()
        with torch.autocast(device_type=device, dtype=dtype):
            c = a @ a
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        ok &= _check(
            "mixed precision matmul", c.dtype == dtype, f"{c.dtype} in {elapsed * 1e3:.1f} ms"
        )
    except Exception as exc:
        ok &= _check("mixed precision matmul", False, repr(exc))
    return ok


def check_openslide(slide: Path | None) -> bool:
    import openslide

    from models.histo.wsi.reader import SlideReader
    from models.histo.wsi.synthetic import SyntheticSlideSpec, write_synthetic_slide

    ok = _check(
        "openslide", True, f"python {openslide.__version__}, lib {openslide.__library_version__}"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = slide or Path(tmp) / "synthetic.tif"
        if slide is None:
            write_synthetic_slide(path, SyntheticSlideSpec(width=1024, height=768))
        try:
            with SlideReader(path, mpp_override=0.5 if slide is None else None) as reader:
                info = reader.info
                patch = reader.read_patch(0, 0, 64, max(info.mpp or 0.5, 0.5))
                detail = (
                    f"{path.name}: {info.width}x{info.height}, {info.level_count} levels, "
                    f"mpp={info.mpp}, magnification={info.magnification}, patch {patch.shape}"
                )
                ok &= _check("open slide + read_patch", patch.shape == (64, 64, 3), detail)
        except Exception as exc:
            ok &= _check("open slide + read_patch", False, repr(exc))
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--require-cuda", action="store_true", help="fail when no GPU is visible")
    parser.add_argument(
        "--slide", type=Path, default=None, help="a real WSI to open instead of a synthetic one"
    )
    args = parser.parse_args(argv)
    print(f"python {sys.version.split()[0]} on {platform.platform()}")
    ok = check_imports()
    ok &= check_torch(args.require_cuda)
    ok &= check_openslide(args.slide)
    print("RESULT:", "environment OK" if ok else "environment has problems (see FAIL lines)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
