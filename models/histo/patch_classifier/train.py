"""Training loop for the patch-level tumor/normal baseline (mini sprint 1.9).

Bu dosya ne işe yarar? Seçilen modeli (ViT-S/16 veya ResNet-50) örnek paketindeki patch'lerle
ince ayar yapar: karışık hassasiyet (AMP: daha hızlı ve az bellek), gradyan biriktirme, katman
bazlı öğrenme hızı, sınıf ağırlıklı kayıp, her epoch sonunda doğrulama kümesinde AUROC, erken
durdurma ve ``onkos.common.checkpoint`` ile ``last.pt``/``best.pt`` kaydı. Aynı config + seed
aynı sonucu vermeli (GPU'da bit-bit garanti yok; CPU'da testle doğrulanır).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from models.histo.patch_classifier.data import Rows, class_weights, make_loader
from models.histo.patch_classifier.metrics import auroc, average_precision
from models.histo.patch_classifier.models import build_model, layerwise_lr_groups
from models.histo.stain.augment import AugmentConfig, HistoAugment, dihedral_views
from onkos.common.checkpoint import CheckpointManager
from onkos.common.precision import autocast_context, make_grad_scaler
from onkos.common.reproducibility import set_seed

log = logging.getLogger(__name__)
MetricLogger = Callable[[str, float, int], None]


@dataclass(frozen=True)
class TrainConfig:
    model: str = "vit_small"
    pretrained: bool = True
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.05
    layer_decay: float = 0.75
    warmup_fraction: float = 0.1
    amp: bool = True
    grad_accum: int = 1
    early_stop_patience: int = 3
    num_workers: int = 0
    image_px: int = 224
    class_weighting: bool = True
    max_steps_per_epoch: int | None = None  # for smoke tests
    device: str = "auto"
    seed: int = 42
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    tta: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_auroc: float
    val_auprc: float
    lr: float
    seconds: float


@dataclass
class TrainOutcome:
    best_val_auroc: float
    best_epoch: int
    history: list[EpochRecord]
    checkpoint_dir: Path
    val_y: NDArray[np.int64]
    val_p: NDArray[np.float64]
    val_slide_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_val_auroc": self.best_val_auroc,
            "best_epoch": self.best_epoch,
            "history": [asdict(h) for h in self.history],
            "checkpoint_dir": str(self.checkpoint_dir),
        }


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if total <= 0:
        return 1.0
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: str,
    amp: bool,
    criterion: nn.Module | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64], list[str], float]:
    """Tumor probabilities for every batch; returns ``(y, p, slide_ids, mean_loss)``."""
    model.eval()
    ys: list[int] = []
    ps: list[float] = []
    slides: list[str] = []
    losses: list[float] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device)
        with autocast_context(amp, device):
            logits = model(images)
            if criterion is not None:
                losses.append(float(criterion(logits.float(), labels).item()))
        prob = torch.softmax(logits.float(), dim=1)[:, 1]
        ys += [int(v) for v in labels.cpu()]
        ps += [float(v) for v in prob.cpu()]
        slides += list(batch["slide_id"])
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return np.asarray(ys, dtype=np.int64), np.asarray(ps, dtype=np.float64), slides, mean_loss


@torch.no_grad()
def predict_tta(
    model: nn.Module, loader: DataLoader[dict[str, Any]], device: str, amp: bool
) -> tuple[NDArray[np.int64], NDArray[np.float64], list[str]]:
    """Average the tumor probability over the 8 dihedral views of each patch."""
    from models.histo.patches.dataset import IMAGENET_MEAN, IMAGENET_STD

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    model.eval()
    ys: list[int] = []
    ps: list[float] = []
    slides: list[str] = []
    for batch in loader:
        images = batch["image"].to(device)
        raw = ((images * std + mean).clamp(0, 1) * 255).round().to(torch.uint8)
        raw_np = raw.permute(0, 2, 3, 1).cpu().numpy()
        probs = torch.zeros(len(raw_np), device=device)
        for view_idx in range(8):
            views = np.stack([dihedral_views(img)[view_idx] for img in raw_np])
            x = torch.from_numpy(views).permute(0, 3, 1, 2).float().div(255.0).to(device)
            x = (x - mean) / std
            with autocast_context(amp, device):
                probs += torch.softmax(model(x).float(), dim=1)[:, 1]
        ps += [float(v) for v in (probs / 8.0).cpu()]
        ys += [int(v) for v in batch["label"]]
        slides += list(batch["slide_id"])
    return np.asarray(ys, dtype=np.int64), np.asarray(ps, dtype=np.float64), slides


def train_classifier(
    cfg: TrainConfig,
    pack_dir: Path,
    train_rows: Rows,
    val_rows: Rows,
    out_dir: Path,
    log_metric: MetricLogger | None = None,
) -> TrainOutcome:
    """Fine-tune ``cfg.model`` on ``train_rows``; keep the epoch with the best validation AUROC."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    augment = HistoAugment(cfg.augment, seed=cfg.seed)
    train_loader = make_loader(
        pack_dir, train_rows, cfg.batch_size, True, cfg.seed, cfg.num_workers, augment, cfg.image_px
    )
    val_loader = make_loader(
        pack_dir, val_rows, cfg.batch_size, False, cfg.seed, cfg.num_workers, None, cfg.image_px
    )
    model = build_model(cfg.model, num_classes=2, pretrained=cfg.pretrained).to(device)
    groups = layerwise_lr_groups(model, cfg.lr, cfg.layer_decay, cfg.weight_decay)
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    weights = torch.tensor(class_weights(train_rows) if cfg.class_weighting else [1.0, 1.0])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    scaler = make_grad_scaler(cfg.amp, device)
    manager = CheckpointManager(out_dir, mode="max")

    steps_per_epoch = len(train_loader)
    if cfg.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)
    total_steps = max(1, steps_per_epoch * cfg.epochs // cfg.grad_accum)
    warmup = int(cfg.warmup_fraction * total_steps)

    history: list[EpochRecord] = []
    best_auroc, best_epoch, stale = -math.inf, -1, 0
    best_eval: tuple[NDArray[np.int64], NDArray[np.float64], list[str]] | None = None
    global_step = 0
    for epoch in range(cfg.epochs):
        started = time.perf_counter()
        model.train()
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device)
            with autocast_context(cfg.amp, device):
                loss = criterion(model(images).float(), labels) / cfg.grad_accum
            scaler.scale(loss).backward()
            losses.append(float(loss.item()) * cfg.grad_accum)
            if (step + 1) % cfg.grad_accum == 0 or step + 1 == steps_per_epoch:
                factor = _cosine_with_warmup(global_step, total_steps, warmup)
                for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
                    group["lr"] = base * factor
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        y, p, slides, val_loss = predict(model, val_loader, device, cfg.amp, criterion)
        val_auroc, val_auprc = auroc(y, p), average_precision(y, p)
        record = EpochRecord(
            epoch=epoch,
            train_loss=float(np.mean(losses)) if losses else float("nan"),
            val_loss=val_loss,
            val_auroc=val_auroc,
            val_auprc=val_auprc,
            lr=float(optimizer.param_groups[-1]["lr"]),
            seconds=time.perf_counter() - started,
        )
        history.append(record)
        log.info("epoch %d: %s", epoch, record)
        if log_metric is not None:
            for key in ("train_loss", "val_loss", "val_auroc", "val_auprc"):
                log_metric(key, float(getattr(record, key)), epoch)
        metric = val_auroc if not math.isnan(val_auroc) else -1.0
        improved = manager.save(model, optimizer, step=epoch, metric=metric, config=cfg.to_dict())
        if improved:
            best_auroc, best_epoch, stale = metric, epoch, 0
            best_eval = (y, p, slides)
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                log.info("early stopping after %d epochs without improvement", stale)
                break
    if best_eval is None:  # every epoch had NaN AUROC (e.g. one-class validation set)
        best_eval = predict(model, val_loader, device, cfg.amp)[:3]
    return TrainOutcome(
        best_val_auroc=best_auroc,
        best_epoch=best_epoch,
        history=history,
        checkpoint_dir=out_dir,
        val_y=best_eval[0],
        val_p=best_eval[1],
        val_slide_ids=best_eval[2],
    )


def load_best(cfg: TrainConfig, checkpoint_dir: Path, device: str | None = None) -> nn.Module:
    """Rebuild the architecture and load ``best.pt`` (weights only, ADR-0004)."""
    device = device or resolve_device(cfg.device)
    model = build_model(cfg.model, num_classes=2, pretrained=False).to(device)
    CheckpointManager(checkpoint_dir, mode="max").load(model, which="best", map_location=device)
    return model
