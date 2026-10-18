"""List and download CAMELYON16 from the AWS Open Data bucket (mini sprint 1.5).

Bu dosya ne işe yarar? Herkese açık S3 kovasındaki CAMELYON16 dosyalarını listeler, manifest
(dosya adı, boyut) yazar ve seçilen klasörleri (anotasyonlar, değerlendirme dosyası, isteğe
bağlı görüntüler) kesintiden devam edebilen biçimde indirir. Kova adresi ve lisans (CC0)
``configs/histo/camelyon.yaml`` içindedir ve resmi kayıttan doğrulanmıştır (2026-10-13).

Usage:
    uv run --group histo python scripts/download_camelyon.py --list-only
    uv run --group histo python scripts/download_camelyon.py      # annotations + evaluation
    uv run --group histo python scripts/download_camelyon.py download.limit=5 \
        download.include=[annotations,evaluation,images]
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from models.histo.camelyon.download import (
    RemoteObject,
    download_object,
    list_bucket,
    write_manifest,
)
from models.histo.config import load_histo_config

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="camelyon")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_histo_config(args.config, args.overrides)
    base = str(cfg.source.base_url)
    session = requests.Session()

    objects: list[RemoteObject] = []
    for name in cfg.source.prefixes:
        found = list_bucket(base, str(cfg.source.prefixes[name]), session)
        objects += found
        print(f"{name:12s} {len(found):5d} files  {sum(o.size for o in found) / 1e9:9.2f} GB")
    for key in (str(cfg.source.readme), str(cfg.source.license)):
        objects.append(RemoteObject(key, 0, None))
    manifest = ROOT / str(cfg.download.manifest)
    write_manifest([o for o in objects if o.size > 0], manifest)
    print(f"manifest -> {manifest}")
    if args.list_only:
        return 0

    include = {str(x) for x in cfg.download.include}
    wanted = [
        o
        for o in objects
        if any(o.key.startswith(str(cfg.source.prefixes[n])) for n in include) or o.size == 0
    ]
    images_prefix = str(cfg.source.prefixes.images)
    only = cfg.download.get("only")
    if only:  # explicit slide ids, e.g. download.only=[tumor_091,normal_108]
        wanted_ids = {str(s).lower() for s in only}
        images = [
            o
            for o in wanted
            if o.key.startswith(images_prefix)
            and o.key.split("/")[-1].rsplit(".", 1)[0].lower() in wanted_ids
        ]
        missing = wanted_ids - {o.key.split("/")[-1].rsplit(".", 1)[0].lower() for o in images}
        if missing:
            print(f"UYARI: kovada bulunamayan slaytlar: {sorted(missing)}")
        wanted = [o for o in wanted if not o.key.startswith(images_prefix)] + images
    limit = cfg.download.limit
    if limit is not None and not only:
        images = [o for o in wanted if o.key.startswith(images_prefix)][: int(limit)]
        wanted = [o for o in wanted if not o.key.startswith(images_prefix)] + images
    dest = ROOT / str(cfg.download.dest_root)
    print(
        f"downloading {len(wanted)} objects ({sum(o.size for o in wanted) / 1e9:.2f} GB) -> {dest}"
    )
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=int(cfg.download.workers)) as pool:
        futures = {
            pool.submit(
                download_object,
                base,
                o,
                dest,
                None,
                8,
                120.0,
                verify_md5=bool(cfg.download.verify_md5),
            ): o
            for o in wanted
            if o.size > 0
        }
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                status = future.result()
            except Exception as exc:
                failed.append((futures[future].key, repr(exc)))
                status = "failed"
            print(f"{done}/{len(futures)} {futures[future].key} {status}", flush=True)
    for o in wanted:  # README / license have unknown size in this listing: fetch directly
        if o.size == 0:
            target = dest / o.key
            target.parent.mkdir(parents=True, exist_ok=True)
            response = session.get(f"{base}/{o.key}", timeout=60)
            if response.ok:
                target.write_bytes(response.content)
    print(f"failed={len(failed)}")
    for key, error in failed:
        print(f"FAILED {key}: {error}")
    print(
        "UYARI: data/raw altındaki dosyaları git'e EKLEME; DVC ile izleyin. "
        "Lisans: license.txt (CC0)."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
