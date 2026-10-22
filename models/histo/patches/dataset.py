"""PyTorch datasets that read patches on demand (mini sprint 1.3 / 1.7 / 1.9).

Bu dosya ne işe yarar? Eğitim sırasında patch'leri HDF5'teki koordinatlardan OpenSlide ile anında
okur (``PatchCoordsDataset``) ya da örnek paketteki PNG dosyalarını yükler (``SamplePackDataset``).
Her ikisi de ``DataLoader`` ile çok işlemli çalışır; slayt dosyası her işlemde ayrı açılır.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from models.histo.patches.h5 import read_coords_h5
from models.histo.wsi.reader import SlideReader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Transform = Callable[[NDArray[np.uint8]], NDArray[np.uint8]]


def to_tensor(rgb: NDArray[np.uint8], normalize: bool = True) -> torch.Tensor:
    """uint8 HWC -> float32 CHW in [0, 1], optionally ImageNet-normalised."""
    array = np.array(rgb, dtype=np.uint8, copy=True)  # writable copy: torch requires it
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor


class PatchCoordsDataset(Dataset[dict[str, Any]]):
    """Patches of one slide, located by a coordinate HDF5 file, read lazily from the slide."""

    def __init__(
        self,
        slide_path: Path,
        coords_h5: Path,
        mpp_override: float | None = None,
        transform: Transform | None = None,
        normalize: bool = True,
    ) -> None:
        self.slide_path = Path(slide_path)
        self.coords, self.tissue_fraction, self.attrs = read_coords_h5(Path(coords_h5))
        self.mpp_override = mpp_override
        self.transform = transform
        self.normalize = normalize
        self._reader: SlideReader | None = None  # opened lazily, once per worker process

    def __len__(self) -> int:
        return int(len(self.coords))

    def _get_reader(self) -> SlideReader:
        if self._reader is None:
            self._reader = SlideReader(self.slide_path, mpp_override=self.mpp_override)
        return self._reader

    def __getitem__(self, index: int) -> dict[str, Any]:
        x, y = (int(v) for v in self.coords[index])
        rgb = self._get_reader().read_patch(x, y, self.attrs.patch_size_px, self.attrs.target_mpp)
        if self.transform is not None:
            rgb = self.transform(rgb)
        return {
            "image": to_tensor(rgb, self.normalize),
            "x": x,
            "y": y,
            "slide_id": self.attrs.slide_id,
            "index": index,
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_reader"] = None  # OpenSlide handles are not picklable; reopen in each worker
        return state


class SamplePackDataset(Dataset[dict[str, Any]]):
    """Labelled PNG patches from a ``patch_v0`` sample pack (``patches.csv`` rows)."""

    def __init__(
        self,
        root: Path,
        rows: list[dict[str, Any]],
        transform: Transform | None = None,
        normalize: bool = True,
        image_px: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.rows = rows
        self.transform = transform
        self.normalize = normalize
        self.image_px = image_px

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import cv2

        row = self.rows[index]
        path = self.root / str(row["image"])
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        if self.image_px is not None and rgb.shape[0] != self.image_px:
            resized = cv2.resize(rgb, (self.image_px, self.image_px), interpolation=cv2.INTER_AREA)
            rgb = np.asarray(resized, dtype=np.uint8)
        if self.transform is not None:
            rgb = self.transform(rgb)
        return {
            "image": to_tensor(rgb, self.normalize),
            "label": int(row["label"]),
            "slide_id": str(row["slide_id"]),
            "x": int(row["x"]),
            "y": int(row["y"]),
        }
