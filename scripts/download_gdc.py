"""Download TCGA open-access omics data from the GDC and build the patient map (mini sprint 1.3).

Typical use::

    uv run --group data python scripts/download_gdc.py --dry-run     # query only, print sizes
    uv run --group data python scripts/download_gdc.py --limit 3     # try 3 files per kind
    uv run --group data python scripts/download_gdc.py               # full download (resumable)

Re-running is safe: finished, md5-verified files are skipped.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import requests

from models.omics.gdc import (
    DEFAULT_PROJECTS,
    FILE_KINDS,
    GdcFile,
    build_patient_map,
    download_file,
    fetch_clinical,
    fetch_data_release,
    inventory,
    inventory_markdown,
    query_files,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_KINDS = ("rnaseq", "maf")  # slides are listed only, never downloaded here


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--projects", nargs="+", default=list(DEFAULT_PROJECTS))
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "gdc")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data" / "manifests")
    parser.add_argument("--interim-dir", type=Path, default=ROOT / "data" / "interim")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "veri_envanteri.md")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="download only N files per kind")
    parser.add_argument(
        "--include-normal", action="store_true", help="also download non-tumor RNA-seq"
    )
    parser.add_argument("--dry-run", action="store_true", help="query and report only")
    return parser.parse_args(argv)


def _gb(files: list[GdcFile]) -> float:
    return sum(f.size for f in files) / 1e9


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    session = requests.Session()
    release = fetch_data_release(session)
    print(f"GDC data release: {release}")

    files: list[GdcFile] = []
    clinical: list[dict[str, object]] = []
    for project in args.projects:
        for kind in FILE_KINDS:
            found = query_files(kind, project, session)
            files += found
            print(f"{project:10s} {kind:7s} {len(found):6d} files  {_gb(found):8.2f} GB")
        clinical += fetch_clinical(project, session)

    selected: list[GdcFile] = []
    for kind in DOWNLOAD_KINDS:
        kind_files = [f for f in files if f.kind == kind]
        if kind == "rnaseq" and not args.include_normal:
            kind_files = [f for f in kind_files if f.is_primary_tumor]
        for project in args.projects:
            subset = [f for f in kind_files if f.project_id == project]
            write_manifest(subset, args.manifest_dir / f"{project}_{kind}.tsv")
        if args.limit is not None:
            kind_files = kind_files[: args.limit]
        selected += kind_files
    for project in args.projects:
        write_manifest(
            [f for f in files if f.kind == "slides" and f.project_id == project],
            args.manifest_dir / f"{project}_slides.tsv",
        )
    meta = {
        "data_release": release,
        "queried_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "projects": args.projects,
        "kinds": FILE_KINDS,
        "include_normal": args.include_normal,
    }
    (args.manifest_dir / "query_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    patient_map = build_patient_map(clinical, files)
    args.interim_dir.mkdir(parents=True, exist_ok=True)
    patient_map.to_parquet(args.interim_dir / "patient_map.parquet", index=False)
    table = inventory(patient_map)
    args.report.write_text(inventory_markdown(table, release, args.projects), encoding="utf-8")
    print(table.to_string())
    print(f"\nTo download: {len(selected)} files, {_gb(selected):.2f} GB -> {args.out}")
    if args.dry_run:
        print("--dry-run: nothing downloaded.")
        return 0

    failed: list[tuple[GdcFile, str]] = []
    counts = {"downloaded": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_file, f, args.out): f for f in selected}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                counts[future.result()] += 1
            except Exception as exc:  # report every failure, keep going
                failed.append((futures[future], repr(exc)))
            if done % 50 == 0 or done == len(futures):
                print(f"{done}/{len(futures)} done, {len(failed)} failed", flush=True)
    print(f"downloaded={counts['downloaded']} skipped={counts['skipped']} failed={len(failed)}")
    print(
        "UYARI: indirilen verileri git add ile EKLEME; DVC kullanın (docs/veri_indirme_rehberi.md)."
    )
    if failed:
        lines = ["file_id\tfile_name\terror"] + [
            f"{f.file_id}\t{f.file_name}\t{e}" for f, e in failed
        ]
        (args.interim_dir / "gdc_failed.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"See {args.interim_dir / 'gdc_failed.tsv'}; re-run to retry only these.")
        return 1
    return 0


def cli() -> int:
    try:
        return main()
    except RuntimeError as exc:
        print(
            f"\nHATA: GDC'ye ulaşılamadı veya istek başarısız: {exc}\n"
            "Ağ erişimini (api.gdc.cancer.gov) ve internet bağlantınızı kontrol edin; "
            "komutu tekrar çalıştırmak güvenlidir.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(cli())
