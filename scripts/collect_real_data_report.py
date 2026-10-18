"""Collect the real-data run outputs into Markdown tables (numbers only from files on disk).

Bu dosya ne işe yarar? ``docs/histo/raporlar/sprint1_gercek_veri_ham_tablolar.md`` dosyasını
diskteki gerçek çıktılardan (QC parquet, koordinat özetleri, paket meta verisi, boya
karşılaştırması, eğitim sonuçları) üretir; rapora elle sayı yazılmaz. Her slaytı OpenSlide ile
açar ve TCGA dosyalarının md5'ini yeniden hesaplar (3,4 GB için birkaç dakika sürebilir). Girdi
dosyaları yoksa (``data/`` ve ``outputs/`` git'e girmez) betik hata verir; sentetik ya da
varsayılan değer üretmez.

Usage:
    uv run --group histo python scripts/collect_real_data_report.py --out <çıktı.md>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from models.histo.wsi.reader import SlideReader

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data/manifests"
SLIDE_LIST = MANIFESTS / "sprint1_gercek_ornek_slaytlar.csv"
TCGA_PROJECTS = ("TCGA-LUAD", "TCGA-BRCA", "TCGA-COAD")
QC_RESULTS = ROOT / "data/interim/qc_results.parquet"
COORDS_SUMMARY = ROOT / "data/processed/patches/coords_v1/_summary.json"
COORDS_RESULTS = ROOT / "data/processed/patches/coords_v1.results.csv"
PACK_META = ROOT / "data/processed/patch_v0/pack_meta.json"
STAIN_DIR = ROOT / "outputs/sprint1_real/stain"
TRAIN_DIR = ROOT / "outputs/sprint1_real/patch_baseline"
QC_COLUMNS = (
    "slide_id",
    "tissue_fraction",
    "n_tissue_tiles",
    "blur_fraction",
    "pen_fraction",
    "dark_fraction",
    "laplacian_var_median",
    "decision",
)


def md5_of(path: Path) -> str:
    """Return the hex MD5 of a file, streamed in 1 MiB blocks."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_md5s() -> dict[str, str]:
    """Map GDC file id -> md5 from the committed TCGA selection manifests."""
    out: dict[str, str] = {}
    for project in TCGA_PROJECTS:
        path = MANIFESTS / f"{project}_dx_slides_selected.tsv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                out[row["id"]] = row["md5"]
    return out


def short_id(text: str, width: int) -> str:
    return text[:width] + ("…" if len(text) > width else "")


def slide_section(lines: list[str]) -> None:
    lines += [
        "## Slaytlar (4a/4b)",
        "",
        "| slide_id | kaynak | dosya | MB | md5 | boyut (px) | seviye | mpp | büyütme | vendor |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    md5s = expected_md5s()
    with SLIDE_LIST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = (SLIDE_LIST.parent / row["path"]).resolve()
        if not path.exists():
            lines.append(f"| {row['slide_id']} | - | - | - | eksik | - | - | - | - | - |")
            continue
        source = "camelyon16" if "camelyon" in path.as_posix().lower() else "tcga"
        md5 = "-"
        if source == "tcga":
            md5 = "OK" if md5_of(path) == md5s.get(row["slide_id"]) else "HATA"
        # TCGA file names carry the patient barcode; keep only the first 12 characters + DX1.
        name = path.name if source == "camelyon16" else path.name.split(".")[0][:12] + "-...DX1"
        with SlideReader(path) as reader:
            info = reader.info
            mpp = "-" if info.mpp is None else f"{info.mpp:.4f}"
            lines.append(
                f"| {short_id(row['slide_id'], 8)} | {source} | {name} | "
                f"{path.stat().st_size / 1e6:.0f} | {md5} | {info.width}x{info.height} | "
                f"{info.level_count} | {mpp} | {info.magnification} | {info.vendor} |"
            )
    lines.append("")


def qc_section(lines: list[str]) -> None:
    qc = pd.read_parquet(QC_RESULTS)
    lines += [
        "## QC (4c)",
        "",
        "| " + " | ".join(QC_COLUMNS) + " |",
        "|" + "---|" * len(QC_COLUMNS),
    ]
    for _, row in qc.iterrows():
        cells: list[str] = []
        for column in QC_COLUMNS:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.3f}" if abs(value) < 100 else f"{value:.0f}")
            else:
                cells.append(short_id(str(value), 10))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"Karar dağılımı: {qc['decision'].value_counts().to_dict()}", ""]


def extraction_section(lines: list[str]) -> None:
    summary = json.loads(COORDS_SUMMARY.read_text(encoding="utf-8"))
    results = pd.read_csv(COORDS_RESULTS)
    lines += [
        "## Patch koordinat çıkarma (4d)",
        "",
        f"Özet: {summary}",
        "",
        "| slide_id | durum | patch | aday | sn |",
        "|---|---|---|---|---|",
    ]
    for _, row in results.sort_values("slide_id").iterrows():
        lines.append(
            f"| {short_id(str(row['slide_id']), 10)} | {row['status']} | {row['n_patches']} | "
            f"{row['n_candidates']} | {row['seconds']:.2f} |"
        )
    lines.append("")


def pack_section(lines: list[str]) -> None:
    meta = json.loads(PACK_META.read_text(encoding="utf-8"))
    lines += [
        "## patch_v0 gerçek örnek paket",
        "",
        f"Sayılar: {json.dumps(meta['counts'], ensure_ascii=False)}",
        f"Başarısız: {meta['failed']}",
        "",
    ]


def stain_section(lines: list[str]) -> None:
    stats = json.loads((STAIN_DIR / "stain_comparison.json").read_text(encoding="utf-8"))
    lines += [
        "## Boya normalizasyonu karşılaştırması (4e)",
        "",
        f"Merkez anahtarı: {stats['centre_key']}; merkezler: {stats['centres']}; "
        f"patch sayısı: {stats['n_patches']}",
        "",
        (STAIN_DIR / "stain_comparison.md").read_text(encoding="utf-8"),
    ]


def training_section(lines: list[str]) -> None:
    raw = (TRAIN_DIR / "results.json").read_text(encoding="utf-8")
    results: dict[str, Any] = json.loads(raw)
    lines += [
        "## ViT-S/16 vs ResNet-50 kısa koşu (4g) — NİHAİ DEĞİL",
        "",
        (TRAIN_DIR / "results_table.md").read_text(encoding="utf-8"),
    ]
    for name, model in results["models"].items():
        for seed in model["seeds"]:
            val = seed["val"]
            lines.append(
                f"- {name} seed {seed['seed']}: en iyi epoch {seed['best_epoch']}, "
                f"{len(seed['history'])} epoch koştu, AUROC {val['auroc']['point']:.3f} "
                f"[{val['auroc']['lower']:.3f}, {val['auroc']['upper']:.3f}], "
                f"AUPRC {val['auprc']['point']:.3f}, F1@0.5 {val['f1']['point']:.3f}, "
                f"doğruluk {val['accuracy']['point']:.3f}"
            )


def build_report(today: str) -> str:
    lines: list[str] = [
        f"# Sprint 1 Gerçek Veri: Ham Çıktı Tabloları ({today})",
        "",
        "> `scripts/collect_real_data_report.py` ile `data/` ve `outputs/sprint1_real/` altındaki "
        "gerçek çıktılardan otomatik üretildi; elle düzenlenmez. Yorum için "
        "`sprint1_gercek_veri_raporu.md`.",
        "",
    ]
    slide_section(lines)
    qc_section(lines)
    extraction_section(lines)
    pack_section(lines)
    stain_section(lines)
    training_section(lines)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sprint 1 gerçek veri ham tablolarını üretir.")
    parser.add_argument(
        "--out", type=Path, default=None, help="Markdown çıktı dosyası (varsayılan: stdout)"
    )
    args = parser.parse_args(argv)
    text = build_report(date.today().strftime("%d.%m.%Y"))
    out: Path | None = args.out
    if out is None:
        sys.stdout.buffer.write(text.encode("utf-8"))
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        print(f"written: {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
