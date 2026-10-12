"""Slide-level data splitting and loaders for the patch baseline (mini sprint 1.9).

Bu dosya ne işe yarar? Örnek paketteki patch satırlarını **slayt bazında** eğitim/doğrulama
kümelerine ayırır (aynı slaytın patch'leri iki kümeye dağılamaz; bu, patch düzeyi sızıntının
ana kaynağıdır), kümelerin kesişmediğini doğrular ve PyTorch ``DataLoader`` nesnelerini kurar.
CAMELYON16 resmi test slaytları ("test") yalnızca ``final_evaluation=True`` ile açılır.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from models.histo.patches.dataset import SamplePackDataset, Transform
from models.histo.patches.sample_pack import read_rows
from onkos.common.reproducibility import make_generator, seed_worker

Rows = list[dict[str, Any]]


def load_labelled_rows(pack_dir: Path) -> Rows:
    """All rows of ``patches.csv`` that carry a tumor/normal label."""
    rows = read_rows(pack_dir / "patches.csv")
    return [r for r in rows if int(r["label"]) in (0, 1)]


def slide_ids(rows: Rows) -> set[str]:
    return {str(r["slide_id"]) for r in rows}


def assert_slide_disjoint(*groups: Rows) -> None:
    """Raise if any slide contributes patches to more than one group."""
    seen: dict[str, int] = {}
    for idx, rows in enumerate(groups):
        for slide in slide_ids(rows):
            if slide in seen and seen[slide] != idx:
                raise AssertionError(f"slide {slide} appears in groups {seen[slide]} and {idx}")
            seen[slide] = idx


def split_rows_by_slide(rows: Rows, val_fraction: float, seed: int) -> tuple[Rows, Rows]:
    """Hold out whole slides for validation, stratified by the slide's majority label."""
    rng = np.random.default_rng(seed)
    by_slide: dict[str, list[int]] = {}
    for r in rows:
        by_slide.setdefault(str(r["slide_id"]), []).append(int(r["label"]))
    strata: dict[int, list[str]] = {0: [], 1: []}
    for slide, labels in by_slide.items():
        strata[int(round(float(np.mean(labels))))].append(slide)
    val: set[str] = set()
    for slides in strata.values():
        slides = sorted(slides)
        rng.shuffle(slides)
        n_val = int(round(len(slides) * val_fraction))
        if len(slides) >= 2 and n_val == 0:
            n_val = 1
        val.update(slides[:n_val])
    train_rows = [r for r in rows if str(r["slide_id"]) not in val]
    val_rows = [r for r in rows if str(r["slide_id"]) in val]
    assert_slide_disjoint(train_rows, val_rows)
    return train_rows, val_rows


def rows_for_split(rows: Rows, split: str, final_evaluation: bool = False) -> Rows:
    """Rows of one pack split; ``test`` is locked unless ``final_evaluation=True``."""
    if split == "test" and not final_evaluation:
        raise PermissionError(
            "The CAMELYON16 test split is locked: open it only for the final evaluation "
            "(final_evaluation=True). Never use it for model or hyper-parameter selection."
        )
    return [r for r in rows if str(r["split"]) == split]


def class_weights(rows: Rows) -> list[float]:
    """Inverse-frequency weights ``[w_normal, w_tumor]`` normalised to mean 1."""
    labels = np.array([int(r["label"]) for r in rows])
    counts = np.array([(labels == 0).sum(), (labels == 1).sum()], dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return [float(w) for w in weights]


def make_loader(
    pack_dir: Path,
    rows: Rows,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    transform: Transform | None = None,
    image_px: int | None = 224,
) -> DataLoader[dict[str, Any]]:
    dataset = SamplePackDataset(pack_dir, rows, transform=transform, image_px=image_px)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
        drop_last=False,
    )


def label_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    labels = [int(r["label"]) for r in rows]
    return {
        "normal": labels.count(0),
        "tumor": labels.count(1),
        "slides": len(slide_ids(list(rows))),
    }
