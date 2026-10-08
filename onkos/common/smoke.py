"""Tiny deterministic training loop used as the 'hello world' experiment and seed test."""

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from onkos.common.checkpoint import CheckpointManager
from onkos.common.precision import autocast_context, make_grad_scaler
from onkos.common.reproducibility import make_generator, seed_worker, set_seed


def run_smoke(cfg: DictConfig, device: str = "cpu", log_mlflow: bool = False) -> list[float]:
    """Train a linear regressor on synthetic data and return the per-step losses.

    Two calls with the same config and seed must return identical loss lists.
    """
    set_seed(int(cfg.seed))
    data_gen = torch.Generator().manual_seed(int(cfg.seed))
    x = torch.randn(cfg.data.n_samples, cfg.model.in_features, generator=data_gen)
    true_w = torch.randn(cfg.model.in_features, cfg.model.out_features, generator=data_gen)
    y = x @ true_w
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(int(cfg.seed)),
    )
    model = torch.nn.Linear(cfg.model.in_features, cfg.model.out_features).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.lr)
    scaler = make_grad_scaler(bool(cfg.train.amp), device)
    manager = CheckpointManager(cfg.train.checkpoint_dir, mode="min")

    losses: list[float] = []
    step = 0
    while step < cfg.train.steps:
        for batch_x, batch_y in loader:
            with autocast_context(bool(cfg.train.amp), device):
                loss = torch.nn.functional.mse_loss(model(batch_x.to(device)), batch_y.to(device))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            if log_mlflow:
                import mlflow

                mlflow.log_metric("loss", losses[-1], step=step)
            step += 1
            if step >= cfg.train.steps:
                break
    manager.save(model, optimizer, step=step, metric=losses[-1], config=cfg)
    return losses
