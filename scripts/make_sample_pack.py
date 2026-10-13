"""Build the patch_v0 sample pack from a slide list (mini sprint 1.3).

Bu dosya ne işe yarar? Ekibin beklediği küçük örnek veri setini üretir: listedeki slaytlardan
doku içeren 256x256 patch'ler PNG olarak kaydedilir, ``patches.csv`` ve ``pack_meta.json``
yazılır, her slayt için kutuların çizildiği doğrulama görseli üretilir.

Slide list CSV columns: slide_id,path,source,split,slide_label,annotation_xml,mpp_override
  - source: camelyon16 | tcga        - split: train | test | unassigned
  - slide_label: tumor | normal | (empty)   - annotation_xml: ASAP XML path or empty

Usage:
    uv run --group histo python scripts/make_sample_pack.py     # configs/histo/patch_v0.yaml
    uv run --group histo python scripts/make_sample_pack.py run.slide_list=x.csv pack.seed=1
    uv run --group histo python scripts/make_sample_pack.py --demo outputs/demo   # synthetic
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from models.histo.config import load_histo_config, section
from models.histo.patches.grid import PatchGridConfig
from models.histo.patches.sample_pack import PackSlide, SamplePackConfig, build_sample_pack
from models.histo.wsi.tissue import TissueMaskConfig

ROOT = Path(__file__).resolve().parents[1]


def read_pack_slides(path: Path) -> list[PackSlide]:
    slides: list[PackSlide] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            slide_path = Path(row["path"])
            if not slide_path.is_absolute():
                slide_path = path.parent / slide_path
            xml = (row.get("annotation_xml") or "").strip()
            xml_path = Path(xml) if xml else None
            if xml_path is not None and not xml_path.is_absolute():
                xml_path = path.parent / xml_path
            mpp = (row.get("mpp_override") or "").strip()
            slides.append(
                PackSlide(
                    slide_id=row["slide_id"].strip(),
                    path=slide_path,
                    source=(row.get("source") or "unknown").strip(),
                    split=(row.get("split") or "unassigned").strip(),
                    slide_label=(row.get("slide_label") or "").strip() or None,
                    annotation_xml=xml_path,
                    mpp_override=float(mpp) if mpp else None,
                )
            )
    return slides


def demo_slides(directory: Path) -> list[PackSlide]:
    """Synthetic slides so the whole pipeline can be exercised without real data."""
    from models.histo.wsi.synthetic import (
        SyntheticSlideSpec,
        rect_polygon,
        write_asap_xml,
        write_synthetic_slide,
    )

    # A fake CAMELYON16 layout so build_tumor_masks.py can be exercised too.
    images = directory / "CAMELYON16" / "images"
    annotations = directory / "CAMELYON16" / "annotations"
    evaluation = directory / "CAMELYON16" / "evaluation"
    for folder in (images, annotations, evaluation):
        folder.mkdir(parents=True, exist_ok=True)
    # Two synthetic 'centres' with different stain colours (for the stain comparison demo).
    colours = {"centre_a": (226, 160, 196), "centre_b": (205, 150, 215)}
    slides: list[PackSlide] = []
    rows = ["image,type,class,center"]
    tumor_rect = (900, 300, 1300, 700)
    for i in range(4):
        name = f"tumor_{i + 1:03d}"
        centre = "centre_a" if i % 2 == 0 else "centre_b"
        spec = SyntheticSlideSpec(seed=i, tumor_rect=tumor_rect, tissue_color=colours[centre])
        write_synthetic_slide(images / f"{name}.tif", spec)
        write_asap_xml(annotations / f"{name}.xml", [("A0", "Tumor", rect_polygon(tumor_rect))])
        slides.append(
            PackSlide(
                name, images / f"{name}.tif", centre, "train", "tumor", annotations / f"{name}.xml"
            )
        )
        rows.append(f"{name},Tumor,macro,{centre}")
    for i in range(2):
        name = f"normal_{i + 1:03d}"
        centre = "centre_a" if i % 2 == 0 else "centre_b"
        spec = SyntheticSlideSpec(seed=10 + i, tumor_rect=None, tissue_color=colours[centre])
        write_synthetic_slide(images / f"{name}.tif", spec)
        slides.append(PackSlide(name, images / f"{name}.tif", centre, "train", "normal"))
        rows.append(f"{name},Normal,negative,{centre}")
    spec = SyntheticSlideSpec(seed=20, tumor_rect=tumor_rect, pen_rect=(400, 200, 700, 260))
    write_synthetic_slide(images / "test_001.tif", spec)
    write_asap_xml(annotations / "test_001.xml", [("A0", "Tumor", rect_polygon(tumor_rect))])
    slides.append(
        PackSlide(
            "test_001",
            images / "test_001.tif",
            "centre_a",
            "test",
            None,
            annotations / "test_001.xml",
        )
    )
    rows.append("test_001,Tumor,micro,centre_a")
    (evaluation / "reference.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    # Slide list for extract_patches.py / run_wsi_qc.py (relative paths resolve to the CSV dir).
    lines = ["slide_id,path,mpp_override"] + [
        f"{s.slide_id},{s.path.relative_to(directory).as_posix()}," for s in slides
    ]
    (directory / "slides.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return slides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="patch_v0")
    parser.add_argument(
        "--demo", type=Path, default=None, help="write synthetic slides here and pack them"
    )
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. pack.seed=1")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_histo_config(args.config, args.overrides)
    pack_cfg = SamplePackConfig(**section(cfg, "pack"))
    grid_cfg = PatchGridConfig(**section(cfg, "grid"))
    mask_cfg = TissueMaskConfig(**section(cfg, "tissue_mask"))
    if args.demo is not None:
        slides = demo_slides(args.demo / "raw")
        out_dir = args.demo / "patch_v0"
    else:
        slides = read_pack_slides(ROOT / str(cfg.run.slide_list))
        out_dir = ROOT / str(cfg.run.out_dir)
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
