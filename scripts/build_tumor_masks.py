"""Tumor masks and the official split document for CAMELYON16 (mini sprint 1.5).

Bu dosya ne işe yarar? İndirilen ASAP XML anotasyonlarını okuyup her slayt için düşük
çözünürlüklü ikili tümör maskesi (PNG) üretir, rastgele seçilen slaytlarda maske-slayt bindirme
görselleri (manuel doğrulama) kaydeder, slayt listesinden resmi eğitim/test bölmesini ve merkez
bilgisini ``camelyon16_splits.json`` olarak yazar ve kısa bir envanter raporu basar.

Usage:
    uv run --group histo python scripts/build_tumor_masks.py
    uv run --group histo python scripts/build_tumor_masks.py masks.downsample=16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from models.histo.camelyon.annotations import (
    overlay_mask,
    parse_asap_xml,
    polygons_to_mask,
    unknown_groups,
)
from models.histo.camelyon.splits import (
    build_splits,
    read_reference_csv,
    slide_id_from_path,
    write_splits,
)
from models.histo.config import load_histo_config
from models.histo.wsi.reader import SlideReader

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("build_tumor_masks")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="camelyon")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import cv2

    cfg = load_histo_config(args.config, args.overrides)
    raw = ROOT / str(cfg.download.dest_root)
    images_dir = raw / str(cfg.source.prefixes.images)
    ann_dir = raw / str(cfg.source.prefixes.annotations)
    reference = raw / str(cfg.source.reference_csv)
    out_dir = ROOT / str(cfg.masks.out_dir)
    overlay_dir = ROOT / str(cfg.masks.overlay_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    tumor_groups = tuple(str(g) for g in cfg.masks.tumor_groups)
    exclusion_groups = tuple(str(g) for g in cfg.masks.exclusion_groups)
    downsample = float(cfg.masks.downsample)

    slides = sorted(images_dir.glob("*.tif")) if images_dir.exists() else []
    xmls = sorted(ann_dir.glob("*.xml")) if ann_dir.exists() else []
    print(f"slides: {len(slides)}  annotations: {len(xmls)}")
    ref = read_reference_csv(reference) if reference.exists() else {}
    if not ref:
        log.warning("reference.csv not found at %s: test labels/centres will be empty", reference)

    # Official split document from the FULL bucket listing (all 399 slides), not from whatever
    # happens to be downloaded; local files are only used for masks. Fallback: local files.
    manifest = ROOT / str(cfg.download.manifest)
    if manifest.exists():
        import csv

        with manifest.open(encoding="utf-8", newline="") as handle:
            ids = [
                slide_id_from_path(row["key"])
                for row in csv.DictReader(handle)
                if row["key"].startswith(str(cfg.source.prefixes.images))
            ]
        print(f"split document from manifest {manifest.name}: {len(ids)} slides")
    else:
        ids = [slide_id_from_path(p) for p in (slides or xmls)]
        log.warning("manifest %s not found: split document built from local files only", manifest)
    doc = build_splits(ids, ref, [str(s) for s in cfg.splits.partially_annotated])
    write_splits(doc, ROOT / str(cfg.splits.out_json))
    print(json.dumps(doc["counts"], indent=2), f"\ncentres: {doc['centres']}")

    rng = np.random.default_rng(int(cfg.masks.seed))
    by_id = {slide_id_from_path(p): p for p in slides}
    sample_ids = (
        set(
            rng.choice(
                sorted(by_id), size=min(int(cfg.masks.overlay_sample), len(by_id)), replace=False
            )
        )
        if by_id
        else set()
    )
    unknown: dict[str, set[str]] = {}
    n_masks = 0
    for xml in xmls:
        slide_id = slide_id_from_path(xml)
        polygons = parse_asap_xml(xml)
        extra = unknown_groups(polygons, tumor_groups, exclusion_groups)
        if extra:
            unknown[slide_id] = extra
        slide_path = by_id.get(slide_id)
        if slide_path is None:
            log.info("%s: annotation without image (images not downloaded?)", slide_id)
            continue
        with SlideReader(slide_path, mpp_override=float(cfg.masks.mpp_override)) as reader:
            mask = polygons_to_mask(
                polygons,
                reader.info.width,
                reader.info.height,
                downsample,
                tumor_groups,
                exclusion_groups,
            )
            cv2.imwrite(str(out_dir / f"{slide_id}.png"), mask * 255)
            n_masks += 1
            if slide_id in sample_ids:
                thumb, thumb_ds = reader.thumbnail(1024)
                overlay = overlay_mask(thumb, mask)
                cv2.imwrite(
                    str(overlay_dir / f"{slide_id}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                )
        meta = {
            "slide_id": slide_id,
            "downsample": downsample,
            "width": reader.info.width,
            "height": reader.info.height,
            "tumor_fraction": float(mask.mean()),
            "n_polygons": len(polygons),
        }
        (out_dir / f"{slide_id}.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    print(f"masks written: {n_masks} -> {out_dir}; overlays: {len(sample_ids)} -> {overlay_dir}")
    if unknown:
        print("UYARI: bilinmeyen anotasyon grupları (maskeye alınmadı):")
        for slide_id, groups in sorted(unknown.items()):
            print(f"  {slide_id}: {sorted(groups)}")
    print(
        "Bindirme görsellerini gözle kontrol edin (20 slayt); sonucu "
        "docs/histo/camelyon_veri_notu.md dosyasına yazın."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
