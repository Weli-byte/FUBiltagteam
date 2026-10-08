"""Seed management. numpy and torch are optional: only seeded when installed."""

import importlib
import os
import random
from typing import Any


def _optional(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, torch (CPU and CUDA) for reproducible runs.

    Args:
        seed: Non-negative integer seed.
        deterministic: Also request deterministic torch kernels (slower, but repeatable).
            Non-deterministic ops only warn instead of raising.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np = _optional("numpy")
    if np is not None:
        np.random.seed(seed % 2**32)
    torch = _optional("torch")
    if torch is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for ``DataLoader``: seeds python and numpy per worker."""
    torch = _optional("torch")
    if torch is None:
        raise RuntimeError("seed_worker requires torch")
    worker_seed = int(torch.initial_seed()) % 2**32
    random.seed(worker_seed)
    np = _optional("numpy")
    if np is not None:
        np.random.seed(worker_seed)


def make_generator(seed: int) -> Any:
    """Return a seeded ``torch.Generator`` to pass to ``DataLoader(generator=...)``."""
    torch = _optional("torch")
    if torch is None:
        raise RuntimeError("make_generator requires torch")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
