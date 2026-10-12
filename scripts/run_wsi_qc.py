"""Slide quality control over a slide list (mini sprint 1.6).

Bu dosya ne işe yarar? Her slayt için QC metriklerini hesaplar (doku oranı, bulanıklık, kalem
izi, koyu alan), eşiklerle "use / suspect / exclude" kararı verir, sonuçları parquet'e yazar,
manuel inceleme için rastgele slaytların maske bindirme görsellerini kaydeder ve toplu bir
Markdown rapor (dağılımlar, dışlama listesi ve gerekçeler) üretir.

Usage:
    uv run --group histo python scripts/run_wsi_qc.py                     # configs/histo/qc.yaml
    uv run --group histo python scripts/run_wsi_qc.py qc.blur_laplacian_var_min=30
    uv run --group histo python scripts/run_wsi_qc.py --slide-list x.csv --out qc.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.histo.camelyon.annotations import overlay_mask
from models.histo.config import load_histo_config, section
from models.histo.patches.extract import read_slide_list
from models.histo.wsi.qc import HsvRange, QcConfig, compute_qc, decide, qc_row
from models.histo.wsi.reader import SlideReader
from models.histo.wsi.tissue import TissueMaskConfig, tissue_mask

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("run_wsi_qc")


def build_qc_config(cfg_qc: dict[str, Any], cfg_mask: dict[str, Any]) -> QcConfig:
    ranges: tuple[HsvRange, ...] = tuple(
        ((int(lo[0]), int(lo[1]), int(lo[2])), (int(hi[0]), int(hi[1]), int(hi[2])))
        for lo, hi in cfg_qc.pop("pen_ranges")
    )
    return QcConfig(pen_ranges=ranges, tissue_mask=TissueMaskConfig(**cfg_mask), **cfg_qc)


def report_markdown(frame: pd.DataFrame, cfg: QcConfig, slide_list: Path) -> str:
    counts = frame["decision"].value_counts().to_dict()
    excluded = frame[frame["decision"] == "exclude"].sort_values("slide_id")
    suspect = frame[frame["decision"] == "suspect"].sort_values("slide_id")
    stats = frame[["tissue_fraction", "blur_fraction", "pen_fraction", "dark_fraction"]].describe()
    lines = [
        "# WSI Kalite Kontrol Raporu",
        "",
        f"> `scripts/run_wsi_qc.py` tarafından üretildi. Girdi listesi: `{slide_list.name}`; "
        f"{len(frame)} slayt.",
        "> Eşikler `configs/histo/qc.yaml` içindedir; kalibrasyon notu `docs/histo/wsi_qc.md`.",
        "",
        "## Karar dağılımı",
        "",
        "| Karar | Slayt |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(counts.items())],
        "",
        "## Metrik özetleri",
        "",
        "| Metrik | ortalama | medyan | %90 | maks |",
        "|---|---|---|---|---|",
    ]
    for name in stats.columns:
        col = frame[name]
        lines.append(
            f"| {name} | {col.mean():.3f} | {col.median():.3f} | {col.quantile(0.9):.3f} "
            f"| {col.max():.3f} |"
        )
    lines += [
        "",
        "## Eşikler",
        "",
        f"- tissue_fraction >= {cfg.tissue_fraction_min}, n_tissue_tiles >= {cfg.min_tissue_tiles}",
        f"- blur_fraction <= {cfg.blur_fraction_max} "
        f"(Laplacian var < {cfg.blur_laplacian_var_min} bulanık)",
        f"- pen_fraction <= {cfg.pen_fraction_max}, dark_fraction <= {cfg.dark_fraction_max}",
        f"- 'suspect': metrik eşiğin {cfg.suspect_ratio:.0%}'ini aşarsa",
        "",
        f"## Dışlanan slaytlar ({len(excluded)})",
        "",
        "| slide_id | gerekçe |",
        "|---|---|",
        *[f"| {r.slide_id} | {r.reasons} |" for r in excluded.itertuples()],
        "",
        f"## Şüpheli slaytlar ({len(suspect)})",
        "",
        "| slide_id | gerekçe |",
        "|---|---|",
        *[f"| {r.slide_id} | {r.reasons} |" for r in suspect.itertuples()],
        "",
        "## Manuel doğrulama",
        "",
        "Rastgele seçilen slaytların maske bindirme görselleri `overlay_dir` altındadır. Gözle",
        "inceleme sonucunu (uyum yüzdesi, yanlış dışlananlar) `docs/histo/wsi_qc.md`'ye işleyin.",
        "Dışlananların kanser türü / kurum dağılımını da orada raporlayın (QC yanlılığı kontrolü).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="qc")
    parser.add_argument("--slide-list", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="qc_results.parquet path")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--overlay-dir", type=Path, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_histo_config(args.config, args.overrides)
    qc_cfg = build_qc_config(section(cfg, "qc"), section(cfg, "tissue_mask"))
    slide_list = args.slide_list or ROOT / str(cfg.run.slide_list)
    out = args.out or ROOT / str(cfg.run.out_parquet)
    report = args.report or ROOT / str(cfg.run.report)
    overlay_dir = args.overlay_dir or ROOT / str(cfg.run.overlay_dir)
    specs = read_slide_list(slide_list)
    rng = np.random.default_rng(int(cfg.run.seed))
    sample = set(
        rng.choice(len(specs), size=min(int(cfg.run.manual_sample), len(specs)), replace=False)
    )

    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        try:
            with SlideReader(spec.path, mpp_override=spec.mpp_override) as reader:
                metrics = compute_qc(reader, spec.slide_id, qc_cfg)
                decision = decide(metrics, qc_cfg)
                rows.append(qc_row(metrics, decision))
                if idx in sample:
                    import cv2

                    thumb, _ = reader.thumbnail(1024)
                    overlay = overlay_mask(
                        thumb, tissue_mask(thumb, qc_cfg.tissue_mask).astype(np.uint8), (0, 200, 0)
                    )
                    overlay_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(overlay_dir / f"{spec.slide_id}.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                    )
            log.info("%s: %s %s", spec.slide_id, decision.decision, "; ".join(decision.reasons))
        except Exception as exc:
            log.exception("%s failed", spec.slide_id)
            rows.append(
                {"slide_id": spec.slide_id, "decision": "exclude", "reasons": f"error: {exc!r}"}
            )
    frame = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(report_markdown(frame, qc_cfg, slide_list), encoding="utf-8")
    print(frame["decision"].value_counts().to_string())
    print(
        f"results -> {out}\nreport  -> {report}\noverlays -> {overlay_dir} ({len(sample)} slides)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
