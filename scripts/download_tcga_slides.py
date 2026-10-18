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
    # small real sample straight from the committed manifests (no GDC query, manifests untouched):
    uv run --group histo --group data python scripts/download_tcga_slides.py --from-manifests \
        --download selection.per_project_limit=3 selection.min_size_mb=100 selection.max_size_mb=500
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import requests

from models.histo.config import load_histo_config
from models.histo.tcga.barcode import try_parse_slide_barcode
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


def files_from_manifests(
    manifest_dir: Path,
    projects: list[str],
    per_project: int,
    min_size_mb: float | None,
    max_size_mb: float | None,
) -> list[GdcFile]:
    """Deterministic small sample from the committed ``<project>_dx_slides_selected.tsv`` files.

    Rule: rows within the size band, sorted by file id, first ``per_project`` per project.
    Patient/sample barcodes are parsed from the file name (GDC manifests carry no case fields).
    """
    files: list[GdcFile] = []
    for project in projects:
        path = manifest_dir / f"{project}_dx_slides_selected.tsv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        rows = [
            r
            for r in rows
            if (min_size_mb is None or int(r["size"]) >= min_size_mb * 1e6)
            and (max_size_mb is None or int(r["size"]) <= max_size_mb * 1e6)
        ]
        rows.sort(key=lambda r: r["id"])
        for r in rows[:per_project]:
            barcode = try_parse_slide_barcode(r["filename"])
            files.append(
                GdcFile(
                    kind="slides",
                    file_id=r["id"],
                    file_name=r["filename"],
                    size=int(r["size"]),
                    md5=r["md5"],
                    project_id=project,
                    patient_barcode=barcode.patient_barcode if barcode else "",
                    sample_barcode=barcode.sample_barcode if barcode else None,
                    sample_type=barcode.sample_type if barcode else None,
                )
            )
    return files


def download_all(
    chosen: list[GdcFile], raw_dir: Path, interim_dir: Path, api: str, workers: int, attempts: int
) -> list[tuple[str, str]]:
    """Download ``chosen`` (md5-verified, resumable); return the failures and write a log table."""
    failed: list[tuple[str, str]] = []
    rows: list[dict[str, object]] = []
    counts = {"downloaded": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_file, f, raw_dir, api=api, attempts=attempts): f for f in chosen
        }
        for done, future in enumerate(as_completed(futures), start=1):
            f = futures[future]
            try:
                status = future.result()
                counts[status] += 1
            except Exception as exc:
                status = "failed"
                failed.append((f.file_id, repr(exc)))
            rows.append(
                {
                    "slide_id": f.file_id,
                    "project_id": f.project_id,
                    "size_bytes": f.size,
                    "md5_expected": f.md5,
                    "status": status,
                    "path": str(target_path(f, raw_dir)),
                }
            )
            print(f"{done}/{len(futures)} {f.file_id} {status}", flush=True)
    print(f"downloaded={counts['downloaded']} skipped={counts['skipped']} failed={len(failed)}")
    pd.DataFrame(rows).to_csv(interim_dir / "tcga_slides_download_log.csv", index=False)
    if failed:
        (interim_dir / "tcga_slides_failed.tsv").write_text(
            "file_id\terror\n" + "\n".join(f"{i}\t{e}" for i, e in failed) + "\n", encoding="utf-8"
        )
    return failed


def write_metadata(chosen: list[GdcFile], raw_dir: Path, interim_dir: Path) -> None:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="tcga_slides")
    parser.add_argument("--download", action="store_true", help="download the selected slides")
    parser.add_argument(
        "--metadata", action="store_true", help="read mpp/levels of downloaded slides"
    )
    parser.add_argument(
        "--from-manifests",
        action="store_true",
        help="skip the GDC query; take a small sample from the committed manifests",
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

    if args.from_manifests:
        sel = cfg.selection
        chosen = files_from_manifests(
            manifest_dir,
            projects,
            int(sel.per_project_limit),
            None if sel.min_size_mb is None else float(sel.min_size_mb),
            None if sel.max_size_mb is None else float(sel.max_size_mb),
        )
        for f in chosen:
            print(f"{f.project_id:10s} {f.file_id} {f.size / 1e6:7.0f} MB")
        print(f"sample: {len(chosen)} slides, {sum(f.size for f in chosen) / 1e9:.2f} GB")
        if not args.download:
            return 0
        failed = download_all(
            chosen,
            raw_dir,
            interim_dir,
            str(cfg.gdc.api),
            int(cfg.gdc.workers),
            int(cfg.gdc.attempts),
        )
        if args.metadata:
            write_metadata(chosen, raw_dir, interim_dir)
        print("UYARI: data/raw altındaki slaytları git'e EKLEME; DVC ile izleyin.")
        return 1 if failed else 0

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

    failed = download_all(
        chosen, raw_dir, interim_dir, str(cfg.gdc.api), int(cfg.gdc.workers), int(cfg.gdc.attempts)
    )
    if args.metadata:
        write_metadata(chosen, raw_dir, interim_dir)
    print("UYARI: data/raw altındaki slaytları git'e EKLEME; DVC ile izleyin.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
