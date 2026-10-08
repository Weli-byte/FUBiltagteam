"""MLflow experiment tracking (ADR-0003): run naming, config logging, git hash tag."""

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf

from onkos.common.gitinfo import git_commit_hash

_NAME_PART = re.compile(r"^[A-Za-z0-9_.]+$")


def build_run_name(module: str, experiment: str, seed: int) -> str:
    """Standard run name ``<module>-<experiment>-<seed>``."""
    for part in (module, experiment):
        if not _NAME_PART.match(part):
            raise ValueError(f"invalid run-name part {part!r} (allowed: letters, digits, _ and .)")
    return f"{module}-{experiment}-{seed}"


def flatten_config(cfg: DictConfig) -> dict[str, str]:
    """Flatten a config into ``{"train.lr": "0.05", ...}`` for ``mlflow.log_params``."""
    flat: dict[str, str] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else str(key), value)
        else:
            flat[prefix] = str(node)

    walk("", OmegaConf.to_container(cfg, resolve=True))
    return flat


@contextmanager
def start_run(cfg: DictConfig) -> Iterator[Any]:
    """Open an MLflow run named from the config, logging params and the git commit hash.

    The tracking URI comes from ``MLFLOW_TRACKING_URI`` if set, else ``cfg.tracking.uri``.
    """
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", str(cfg.tracking.uri)))
    mlflow.set_experiment(str(cfg.tracking.experiment))
    name = build_run_name(str(cfg.run.module), str(cfg.run.experiment), int(cfg.seed))
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(flatten_config(cfg))
        mlflow.set_tag("git_commit", git_commit_hash() or "unknown")
        yield run
