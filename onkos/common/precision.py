"""Shared helpers for mixed precision (AMP) and gradient checkpointing."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, cast

import torch
from torch.utils.checkpoint import checkpoint


def autocast_context(enabled: bool, device: str) -> AbstractContextManager[Any]:
    """Return an autocast context (bfloat16 on CPU, float16 on CUDA); no-op when disabled."""
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
    return cast(
        AbstractContextManager[Any],
        torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled),
    )


def make_grad_scaler(enabled: bool, device: str) -> Any:
    """Return a ``GradScaler`` that only scales on CUDA with AMP enabled."""
    return torch.amp.GradScaler("cuda", enabled=enabled and device.startswith("cuda"))


def maybe_checkpoint(fn: Callable[..., Any], *args: Any, enabled: bool) -> Any:
    """Run ``fn(*args)`` with gradient checkpointing (less memory, more compute) if ``enabled``."""
    if not enabled:
        return fn(*args)
    return checkpoint(fn, *args, use_reentrant=False)
