"""Model builders for the patch baseline: ViT-S/16, ViT-B/16, ResNet-50 (mini sprint 1.9).

Bu dosya ne işe yarar? ``timm`` kütüphanesinden ImageNet ile ön eğitilmiş modelleri 2 sınıflı
(tümör/normal) başlıkla kurar ve katman bazlı öğrenme hızı (erken katmanlar az, son katmanlar çok
güncellenir) için parametre gruplarını üretir. Testler için ağırlık indirmeyen küçük bir CNN de
vardır. Ağırlık lisansları (Apache-2.0, timm model kartları) ``docs/histo/vit_baseline_v1.md``'de.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

TIMM_NAMES: dict[str, str] = {
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "resnet50": "resnet50.a1_in1k",
}


class TinyCNN(nn.Module):
    """Very small CNN used only for smoke tests (no download, runs on CPU in seconds)."""

    def __init__(self, num_classes: int = 2, width: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


def build_model(name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """``name`` in ``TIMM_NAMES`` (or a raw timm model name) or ``"tiny_cnn"``."""
    if name == "tiny_cnn":
        return TinyCNN(num_classes=num_classes)
    import timm

    timm_name = TIMM_NAMES.get(name, name)
    model: nn.Module = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    return model


def _layer_id(param_name: str, depth: int, kind: str) -> int:
    """Map a parameter name to a layer index 0..depth+1 (0 = stem, depth+1 = head)."""
    if kind == "vit":
        if param_name.startswith(("cls_token", "pos_embed", "patch_embed")):
            return 0
        if param_name.startswith("blocks."):
            return int(param_name.split(".")[1]) + 1
        return depth + 1
    if kind == "resnet":
        if param_name.startswith("layer"):
            return int(param_name[5])  # layer1..layer4 -> 1..4
        if param_name.startswith(("conv1", "bn1")):
            return 0
        return depth + 1
    return depth + 1


def layerwise_lr_groups(
    model: nn.Module, base_lr: float, decay: float, weight_decay: float
) -> list[dict[str, Any]]:
    """Parameter groups with ``lr = base_lr * decay ** (depth + 1 - layer_id)``.

    ViT: per transformer block; ResNet: per stage; other models: a single group. Biases and
    normalisation weights get no weight decay (standard practice).
    """
    if hasattr(model, "blocks") and hasattr(model, "patch_embed"):
        kind, depth = "vit", len(cast(nn.Sequential, model.blocks))
    elif hasattr(model, "layer4"):
        kind, depth = "resnet", 4
    else:
        kind, depth = "flat", 0
    groups: dict[tuple[int, bool], dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer = _layer_id(name, depth, kind) if kind != "flat" else depth + 1
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        key = (layer, no_decay)
        if key not in groups:
            scale = decay ** (depth + 1 - layer) if kind != "flat" else 1.0
            groups[key] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "layer_id": layer,
            }
        groups[key]["params"].append(param)
    return [groups[k] for k in sorted(groups)]
