"""Checkpoint save/load: keeps ``last.pt`` and ``best.pt`` with optimizer state and metadata."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch
from omegaconf import DictConfig, OmegaConf

from onkos.common.gitinfo import git_commit_hash


class CheckpointManager:
    """Writes atomic checkpoints into one directory.

    Each file stores model and optimizer ``state_dict``s, step, metric, config and git hash.
    Files are loaded with ``weights_only=True`` so no arbitrary pickled code can run (ADR-0004).
    """

    def __init__(self, directory: Path | str, mode: Literal["min", "max"] = "min") -> None:
        self.directory = Path(directory)
        self.mode = mode
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def last_path(self) -> Path:
        return self.directory / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    def _is_better(self, metric: float, best: float) -> bool:
        return metric < best if self.mode == "min" else metric > best

    def best_metric(self) -> float | None:
        """Metric stored in ``best.pt``, or ``None`` if there is no best checkpoint yet."""
        if not self.best_path.exists():
            return None
        metric = torch.load(self.best_path, map_location="cpu", weights_only=True)["metric"]
        return None if metric is None else float(metric)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        step: int,
        metric: float | None = None,
        config: DictConfig | Mapping[str, Any] | None = None,
    ) -> bool:
        """Save ``last.pt`` and, if ``metric`` improved, ``best.pt``.

        Returns:
            ``True`` if this call also updated ``best.pt``.
        """
        plain_config: Any = (
            OmegaConf.to_container(config, resolve=True)
            if isinstance(config, DictConfig)
            else dict(config or {})
        )
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "metric": metric,
            "config": plain_config,
            "git_hash": git_commit_hash(),
        }
        self._atomic_save(payload, self.last_path)
        if metric is None:
            return False
        best = self.best_metric()
        if best is None or self._is_better(metric, best):
            self._atomic_save(payload, self.best_path)
            return True
        return False

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        which: Literal["last", "best"] = "last",
        map_location: str = "cpu",
    ) -> dict[str, Any]:
        """Restore model (and optimizer) state; return the remaining metadata."""
        path = self.last_path if which == "last" else self.best_path
        payload = torch.load(path, map_location=map_location, weights_only=True)
        model.load_state_dict(payload["model"])
        if optimizer is not None and payload["optimizer"] is not None:
            optimizer.load_state_dict(payload["optimizer"])
        return {k: payload[k] for k in ("step", "metric", "config", "git_hash")}

    @staticmethod
    def _atomic_save(payload: dict[str, Any], path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
