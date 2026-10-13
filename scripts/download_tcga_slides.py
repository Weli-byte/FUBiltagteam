"""List, select and download TCGA diagnostic H&E slides from the GDC (mini sprint 1.4).

Bu dosya ne işe yarar? GDC API'sinden (ulusal kanser veri portalının program arayüzü) seçili
projelerin **açık erişimli** tanı slaytlarını (DX, FFPE) listeler, barkodları ayrıştırıp bir
envanter çıkarır, hasta başına bir slayt seçer (yazılı kural), proje liderinin hasta listesiyle
kesiştirir, gdc-client uyumlu manifest ve toplu rapor yazar; ``--download`` ile seçilen
slaytları md5 doğrulamalı ve kesintiden devam edebilen biçimde indirir.

GDC transport helpers are reused from ``models.omics.gdc`` (project leader's 1.3 module) at the
script level; feature modules themselves stay independent (ADR-0001).

Usage:
    uv run --group histo --group data python scripts/download_tcga_slides.py   # list + inventory
    uv run --group histo --group data python scripts/download_tcga_slides.py --download \
        selection.limit=5
    uv run --group histo --group data python scripts/download_tcga_slides.py --download --metadata
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import requests

from models.histo.config import load_histo_config
from models.histo.tcga.inventory import (
    build_slide_inventory,
    inventory_summary,
    patient_slide_map,
    select_slides,
    slide_metadata,
    summary_markdown,
)
from models.omics.gdc import (
    GdcFile,
    download_file,
    fetch_data_release,
    query_files,
    target_path,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("download_tcga_slides")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="tcga_slides")
    parser.add_argument("--download", action="store_true", help="download the selected slides")
    parser.add_argument(
        "--metadata", action="store_true", help="read mpp/levels of downloaded slides"
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_histo_config(args.config, args.overrides)
    projects = [str(p) for p in cfg.projects]
    manifest_dir = ROOT / str(cfg.paths.manifest_dir)
    interim_dir = ROOT / str(cfg.paths.interim_dir)
    raw_dir = ROOT / str(cfg.paths.raw_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    release = fetch_data_release(session, api=str(cfg.gdc.api))
    print(f"GDC data release: {release}")
    files: list[GdcFile] = []
    for project in projects:
        found = query_files("slides", project, session, api=str(cfg.gdc.api))
        files += found
        print(
            f"{project:10s} {len(found):6d} diagnostic slides  "
            f"{sum(f.size for f in found) / 1e9:8.1f} GB"
        )
    inventory = build_slide_inventory([asdict(f) for f in files])

    qc_path = ROOT / str(cfg.paths.qc_results)
    qc = pd.read_parquet(qc_path) if qc_path.exists() else None
    inventory = select_slides(inventory, qc)
    leader_path = ROOT / str(cfg.paths.leader_patient_map)
    patients: list[str] = []
    if leader_path.exists():
        patients = sorted(pd.read_parquet(leader_path)["patient_barcode"].astype(str))
    else:
        log.warning(
            "project leader's patient_map not found at %s; intersection is empty", leader_path
        )
    table = patient_slide_map(inventory, patients)
    inventory.to_parquet(interim_dir / "slide_inventory.parquet", index=False)
    table.to_parquet(interim_dir / "patient_slide_map.parquet", index=False)

    selected = inventory[inventory["selected"]]
    if bool(cfg.selection.only_in_leader_list) and patients:
        selected = selected[selected["patient_barcode"].isin(set(patients))]
    limit = cfg.selection.limit
    if limit is not None:
        selected = selected.head(int(limit))
    by_id = {f.file_id: f for f in files}
    chosen = [by_id[s] for s in selected["slide_id"]]
    for project in projects:
        write_manifest(
            [f for f in chosen if f.project_id == project],
            manifest_dir / f"{project}_dx_slides_selected.tsv",
        )
    summary = inventory_summary(inventory, table)
    report = ROOT / str(cfg.paths.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    note = (
        f"- İndirme planı: {len(chosen)} slayt, {sum(f.size for f in chosen) / 1e9:.1f} GB "
        f"(only_in_leader_list={bool(cfg.selection.only_in_leader_list)}, limit={limit})."
    )
    if not patients:
        note += (
            "\n- **Not:** proje liderinin hasta listesi (`patient_map.parquet`) bu makinede yoktu; "
            "kesişim sütunları 0 görünür. Liste geldiğinde betik yeniden çalıştırılır."
        )
    report.write_text(summary_markdown(summary, release, note), encoding="utf-8")
    print(summary.to_string())
    print(note)
    if not args.download:
        print("--download verilmedi: yalnızca envanter ve manifestler yazıldı.")
        return 0

    failed: list[tuple[str, str]] = []
    counts = {"downloaded": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=int(cfg.gdc.workers)) as pool:
        futures = {
            pool.submit(
                download_file, f, raw_dir, api=str(cfg.gdc.api), attempts=int(cfg.gdc.attempts)
            ): f
            for f in chosen
        }
        for done, future in enumerate(as_completed(futures), start=1):
            f = futures[future]
            try:
                counts[future.result()] += 1
            except Exception as exc:
                failed.append((f.file_id, repr(exc)))
            print(f"{done}/{len(futures)} done, {len(failed)} failed", flush=True)
    print(f"downloaded={counts['downloaded']} skipped={counts['skipped']} failed={len(failed)}")
    if failed:
        (interim_dir / "tcga_slides_failed.tsv").write_text(
            "file_id\terror\n" + "\n".join(f"{i}\t{e}" for i, e in failed) + "\n", encoding="utf-8"
        )
    if args.metadata:
        meta_rows = []
        for f in chosen:
            path = target_path(f, raw_dir)
            if path.exists():
                try:
                    meta_rows.append({"slide_id": f.file_id, **slide_metadata(path)})
                except Exception as exc:
                    meta_rows.append({"slide_id": f.file_id, "error": repr(exc)})
        pd.DataFrame(meta_rows).to_parquet(interim_dir / "slide_metadata.parquet", index=False)
        print(f"slide metadata -> {interim_dir / 'slide_metadata.parquet'}")
    print("UYARI: data/raw altındaki slaytları git'e EKLEME; DVC ile izleyin.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
