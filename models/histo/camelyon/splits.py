"""CAMELYON16 official train/test split and centre metadata (mini sprint 1.5).

Bu dosya ne işe yarar? CAMELYON16'nın resmi bölmesini slayt adlarından türetir
(``normal_###``/``tumor_###`` = eğitim, ``test_###`` = test), resmi ``evaluation/reference.csv``
dosyasından slayt etiketi, lezyon sınıfı (negative/micro/macro) ve merkez bilgisini okur, eğitim
ve test listelerinin kesişmediğini doğrular ve ``camelyon_splits.json`` üretir. Sayılar dosya
listesinden sayılır; belleğe güvenilmez.

Resmi kaynak notları (AWS Open Data kopyası, 2026-10-13'te doğrulandı): 399 slayt (159 normal,
111 tumor, 129 test); ``normal_086`` yanlış sınıflandığı için ``tumor_111`` oldu; ``test_049``
kopya olduğu için çıkarıldı; ``test_114`` ve 20 tumor slaytı tam anotasyonlu değildir.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRAIN_RE = re.compile(r"^(normal|tumor)_(\d{3})$")
TEST_RE = re.compile(r"^test_(\d{3})$")

# From the official README (AWS Open Data copy); keep in sync with configs/histo/camelyon.yaml.
PARTIALLY_ANNOTATED: tuple[str, ...] = (
    "test_114",
    "tumor_010",
    "tumor_015",
    "tumor_018",
    "tumor_020",
    "tumor_025",
    "tumor_029",
    "tumor_033",
    "tumor_034",
    "tumor_044",
    "tumor_046",
    "tumor_051",
    "tumor_054",
    "tumor_055",
    "tumor_056",
    "tumor_067",
    "tumor_079",
    "tumor_085",
    "tumor_092",
    "tumor_095",
    "tumor_110",
)


@dataclass(frozen=True)
class ReferenceRow:
    slide_id: str
    label: str  # tumor | normal
    lesion_class: str | None  # negative | micro | macro | itc (as given in the file)
    centre: str | None


def slide_id_from_path(path: str | Path) -> str:
    """``.../tumor_001.tif`` -> ``tumor_001`` (lower-case, extension removed)."""
    return Path(path).stem.lower()


def official_split(slide_id: str) -> str:
    """``train`` or ``test`` from the CAMELYON16 naming convention; raises for anything else."""
    if TRAIN_RE.match(slide_id):
        return "train"
    if TEST_RE.match(slide_id):
        return "test"
    raise ValueError(f"not a CAMELYON16 slide id: {slide_id!r}")


def slide_label_from_id(slide_id: str) -> str | None:
    """``tumor``/``normal`` for training slides (from the name); ``None`` for test slides."""
    match = TRAIN_RE.match(slide_id)
    return match.group(1) if match else None


def _normalise_label(value: str) -> str:
    value = value.strip().lower()
    if value in ("tumor", "tumour", "positive", "metastases"):
        return "tumor"
    if value in ("normal", "negative"):
        return "normal"
    raise ValueError(f"unexpected label {value!r}")


def read_reference_csv(path: Path) -> dict[str, ReferenceRow]:
    """Parse ``evaluation/reference.csv`` (columns ``image,type,class,center``; header optional).

    Without a header the columns are taken positionally in that order. Slide ids are
    lower-cased and stripped of extensions so they match :func:`slide_id_from_path`.
    """
    rows: dict[str, ReferenceRow] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header: list[str] | None = None
        for raw in reader:
            if not raw or not raw[0].strip():
                continue
            cells = [c.strip() for c in raw]
            if header is None and cells[0].lower() in ("image", "slide", "slide_id"):
                header = [c.lower() for c in cells]
                continue
            columns = header or ["image", "type", "class", "center"]
            record = dict(zip(columns, cells, strict=False))
            slide_id = slide_id_from_path(record["image"])
            if not (TRAIN_RE.match(slide_id) or TEST_RE.match(slide_id)):
                continue
            rows[slide_id] = ReferenceRow(
                slide_id=slide_id,
                label=_normalise_label(record.get("type", "")),
                lesion_class=(record.get("class") or None),
                centre=(record.get("center") or None),
            )
    return rows


def build_splits(
    slide_ids: Iterable[str],
    reference: Mapping[str, ReferenceRow] | None = None,
    partially_annotated: Iterable[str] = PARTIALLY_ANNOTATED,
) -> dict[str, Any]:
    """Assemble the split document; raises if train and test overlap or an id is unknown.

    Training labels come from the slide name and, when a reference row exists, must agree with
    it (a mismatch raises: the data copy would be inconsistent with the official listing).
    """
    partial = set(partially_annotated)
    ref = reference or {}
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for slide_id in sorted({s.lower() for s in slide_ids}):
        split = official_split(slide_id)
        row = ref.get(slide_id)
        entry: dict[str, Any] = {
            "slide_id": slide_id,
            "centre": row.centre if row else None,
            "lesion_class": row.lesion_class if row else None,
            "partially_annotated": slide_id in partial,
        }
        if split == "train":
            name_label = slide_label_from_id(slide_id)
            if row is not None and row.label != name_label:
                raise ValueError(f"{slide_id}: name says {name_label}, reference says {row.label}")
            entry["label"] = name_label
            train.append(entry)
        else:
            entry["label"] = row.label if row else None
            test.append(entry)
    overlap = {e["slide_id"] for e in train} & {e["slide_id"] for e in test}
    if overlap:
        raise ValueError(f"train/test overlap: {sorted(overlap)[:5]}")
    counts = {
        "train": len(train),
        "train_tumor": sum(e["label"] == "tumor" for e in train),
        "train_normal": sum(e["label"] == "normal" for e in train),
        "test": len(test),
        "test_tumor": sum(e["label"] == "tumor" for e in test),
        "test_normal": sum(e["label"] == "normal" for e in test),
        "test_unlabelled": sum(e["label"] is None for e in test),
    }
    centres = sorted({e["centre"] for e in train + test if e["centre"]})
    return {
        "dataset": "CAMELYON16",
        "rule": "official split from slide names; test slides are never used for training, "
        "model selection or hyper-parameter tuning",
        "counts": counts,
        "centres": centres,
        "train": train,
        "test": test,
    }


def write_splits(doc: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(doc), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_splits(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return doc


def split_of(doc: Mapping[str, Any]) -> dict[str, str]:
    """``slide_id -> split`` lookup from a split document."""
    out = {e["slide_id"]: "train" for e in doc["train"]}
    out.update({e["slide_id"]: "test" for e in doc["test"]})
    return out
