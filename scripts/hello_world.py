"""'Hello world' experiment: seeded training logged to MLflow.

Usage: uv run --group train python scripts/hello_world.py train.steps=100 seed=42
"""

import sys
from pathlib import Path

import torch

from onkos.common.config import load_config
from onkos.common.smoke import run_smoke
from onkos.common.tracking import start_run


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs", sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with start_run(cfg):
        losses = run_smoke(cfg, device=device, log_mlflow=True)
    print(f"device={device} steps={len(losses)} first={losses[0]:.6f} last={losses[-1]:.6f}")


if __name__ == "__main__":
    main()
