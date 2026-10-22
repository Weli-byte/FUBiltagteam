"""Resumable downloads from a public (anonymous) S3-style bucket over HTTPS (mini sprint 1.5).

Bu dosya ne işe yarar? CAMELYON verisinin AWS Open Data kopyasını (herkese açık S3 kovası)
indirmek için kova listesini okur, bir manifest (dosya adı, boyut) yazar ve dosyaları kaldığı
yerden devam edebilen (HTTP Range) biçimde indirir. Boyut doğrulaması yapar; md5 doğrulaması
kova ETag'i çok parçalı yüklemelerde md5 olmadığı için yalnızca tek parçalı nesnelerde mümkündür.
Kova adresi konfigürasyondan gelir ve resmi kaynaktan doğrulanmalıdır.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


@dataclass(frozen=True)
class RemoteObject:
    key: str
    size: int
    etag: str | None

    @property
    def md5(self) -> str | None:
        """ETag is the md5 only for single-part uploads (no ``-N`` suffix)."""
        if self.etag and re.fullmatch(r"[0-9a-f]{32}", self.etag):
            return self.etag
        return None


def list_bucket(
    base_url: str,
    prefix: str,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> list[RemoteObject]:
    """List objects under ``prefix`` using the S3 ListObjectsV2 XML API (anonymous)."""
    session = session or requests.Session()
    objects: list[RemoteObject] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        response = session.get(base_url.rstrip("/") + "/", params=params, timeout=timeout)
        response.raise_for_status()
        root = ET.fromstring(response.content)  # noqa: S314 - public listing from the bucket
        for item in root.findall(f"{S3_NS}Contents"):
            key = item.findtext(f"{S3_NS}Key") or ""
            size = int(item.findtext(f"{S3_NS}Size") or 0)
            etag = (item.findtext(f"{S3_NS}ETag") or "").strip('"') or None
            if key and not key.endswith("/"):
                objects.append(RemoteObject(key, size, etag))
        truncated = (root.findtext(f"{S3_NS}IsTruncated") or "false").lower() == "true"
        token = root.findtext(f"{S3_NS}NextContinuationToken") if truncated else None
        if not token:
            return objects


def write_manifest(objects: Iterable[RemoteObject], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "size", "etag"])
        for obj in objects:
            writer.writerow([obj.key, obj.size, obj.etag or ""])


def read_manifest(path: Path) -> list[RemoteObject]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            RemoteObject(row["key"], int(row["size"]), row["etag"] or None)
            for row in csv.DictReader(handle)
        ]


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download_object(
    base_url: str,
    obj: RemoteObject,
    dest_root: Path,
    session: requests.Session | None = None,
    attempts: int = 8,
    timeout: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
    verify_md5: bool = True,
) -> str:
    """Download ``obj`` to ``dest_root/key``; resumes a partial ``.part`` with HTTP Range.

    Returns ``'skipped'`` (complete file already present) or ``'downloaded'``.
    """
    session = session or requests.Session()
    final = dest_root / obj.key
    if final.exists() and final.stat().st_size == obj.size:
        if not verify_md5 or obj.md5 is None or md5_of(final) == obj.md5:
            return "skipped"
        final.unlink()
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_suffix(final.suffix + ".part")
    url = base_url.rstrip("/") + "/" + obj.key
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have > obj.size:
                part.unlink()
                have = 0
            if have < obj.size:  # a complete .part (interrupted verify/rename) skips the request
                headers = {"Range": f"bytes={have}-"} if have else {}
                with session.get(url, headers=headers, stream=True, timeout=timeout) as response:
                    if have and response.status_code == 416:  # range not satisfiable: restart
                        part.unlink()
                        raise OSError(f"server rejected resume offset {have} for {obj.key}")
                    if have and response.status_code != 206:
                        have = 0  # server ignored the range: start over
                    response.raise_for_status()
                    with part.open("ab" if have else "wb") as out:
                        for block in response.iter_content(chunk_size=1 << 20):
                            out.write(block)
            if part.stat().st_size != obj.size:
                raise OSError(f"size mismatch for {obj.key}: {part.stat().st_size} != {obj.size}")
            if verify_md5 and obj.md5 is not None and md5_of(part) != obj.md5:
                part.unlink()
                raise OSError(f"md5 mismatch for {obj.key}")
            os.replace(part, final)
            return "downloaded"
        except (requests.RequestException, OSError) as exc:
            last = exc
            log.warning("attempt %d/%d failed for %s: %r", attempt + 1, attempts, obj.key, exc)
            if attempt < attempts - 1:
                sleep(min(60.0, 2.0 ** (attempt + 1)))
    raise RuntimeError(f"download failed for {obj.key}: {last!r}") from last
