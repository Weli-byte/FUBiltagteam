"""Build the patch_v0 sample pack from a slide list (mini sprint 1.3).

Bu dosya ne işe yarar? Ekibin beklediği küçük örnek veri setini üretir: listedeki **gerçek**
slaytlardan doku içeren 256x256 patch'ler PNG olarak kaydedilir, ``patches.csv`` ve
``pack_meta.json`` yazılır, her slayt için kutuların çizildiği doğrulama görseli üretilir.
Sentetik/sahte slayt üretmez; sentetik slaytlar yalnızca ``tests/`` altındaki test düzeneğindedir.

Slide list CSV columns: slide_id,path,source,split,slide_label,annotation_xml,mpp_override,centre
  - source: camelyon16 | tcga        - split: train | test | unassigned
  - slide_label: tumor | normal | (empty)   - annotation_xml: ASAP XML path or empty
  - centre: scanning centre / tissue source site (optional; used by the stain comparison)
  - relative ``path`` / ``annotation_xml`` entries resolve against the CSV's own directory
    (e.g. ``../raw/camelyon/...`` from ``data/manifests/``)
  - values are case-insensitive but validated; a CAMELYON16 slide id (``normal_###``,
    ``tumor_###``, ``test_###``) must carry its official split (``test_###`` -> ``test``)

Usage:
    uv run --group histo python scripts/make_sample_pack.py     # configs/histo/patch_v0.yaml
    uv run --group histo python scripts/make_sample_pack.py run.slide_list=x.csv pack.seed=1
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from models.histo.camelyon.splits import TEST_RE, TRAIN_RE, official_split
from models.histo.config import load_histo_config, section
from models.histo.patches.grid import PatchGridConfig
from models.histo.patches.sample_pack import PackSlide, SamplePackConfig, build_sample_pack
from models.histo.wsi.tissue import TissueMaskConfig

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "test", "unassigned")
LABELS = ("tumor", "normal")
COLUMNS = "slide_id,path,source,split,slide_label,annotation_xml,mpp_override,centre"


def _normalise_label(value: str, slide_id: str) -> str | None:
    label = value.strip().lower()
    if not label:
        return None
    aliases = {"tumour": "tumor", "positive": "tumor", "negative": "normal"}
    label = aliases.get(label, label)
    if label not in LABELS:
        raise ValueError(f"{slide_id}: slide_label {value!r} not in {LABELS}")
    return label


def _normalise_split(value: str, slide_id: str) -> str:
    split = (value.strip().lower() or "unassigned").replace("validation", "val")
    if split not in SPLITS:
        raise ValueError(f"{slide_id}: split {value!r} not in {SPLITS}")
    if TRAIN_RE.match(slide_id) or TEST_RE.match(slide_id):
        expected = official_split(slide_id)
        if split != expected:
            raise ValueError(
                f"{slide_id}: CAMELYON16 official split is {expected!r}, list says {split!r} "
                "(the official test split is locked)"
            )
    return split


def read_pack_slides(path: Path) -> list[PackSlide]:
    """Parse the slide list; validate labels/splits so a typo can never flip a label."""
    slides: list[PackSlide] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            slide_id = row["slide_id"].strip()
            slide_path = Path(row["path"].strip())
            if not slide_path.is_absolute():
                slide_path = path.parent / slide_path
            xml = (row.get("annotation_xml") or "").strip()
            xml_path = Path(xml) if xml else None
            if xml_path is not None and not xml_path.is_absolute():
                xml_path = path.parent / xml_path
            mpp = (row.get("mpp_override") or "").strip()
            slides.append(
                PackSlide(
                    slide_id=slide_id,
                    path=slide_path,
                    source=(row.get("source") or "unknown").strip().lower(),
                    split=_normalise_split(row.get("split") or "", slide_id),
                    slide_label=_normalise_label(row.get("slide_label") or "", slide_id),
                    annotation_xml=xml_path,
                    mpp_override=float(mpp) if mpp else None,
                    centre=(row.get("centre") or "").strip() or None,
                )
            )
    if len({s.slide_id for s in slides}) != len(slides):
        raise ValueError("slide_id values must be unique")
    return slides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="patch_v0")
    parser.add_argument("--slide-list", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. pack.seed=1")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_histo_config(args.config, args.overrides)
    pack_cfg = SamplePackConfig(**section(cfg, "pack"))
    grid_cfg = PatchGridConfig(**section(cfg, "grid"))
    mask_cfg = TissueMaskConfig(**section(cfg, "tissue_mask"))
    slide_list = args.slide_list or ROOT / str(cfg.run.slide_list)
    out_dir = args.out or ROOT / str(cfg.run.out_dir)
    if not slide_list.exists():
        print(f"HATA: slayt listesi bulunamadı: {slide_list} (sütunlar: {COLUMNS})")
        return 2
    slides = read_pack_slides(slide_list)
    missing = [s.slide_id for s in slides if not s.path.exists()]
    if missing:
        print(f"HATA: slayt dosyası bulunamadı: {missing[:5]} (toplam {len(missing)})")
        return 2
    rows = build_sample_pack(slides, out_dir, pack_cfg, grid_cfg, mask_cfg)
    labelled = sum(int(r["label"]) in (0, 1) for r in rows)
    print(f"{len(rows)} patches from {len({r['slide_id'] for r in rows})} slides -> {out_dir}")
    print(
        f"labelled: {labelled} (tumor {sum(r['label'] == 1 for r in rows)}, "
        f"normal {sum(r['label'] == 0 for r in rows)})"
    )
    print("Check overlays/ by eye (30 random patches + one slide's boxes) before sharing the pack.")
    print("UYARI: data/ altındaki çıktıları git'e EKLEME; DVC ile izleyin.")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
