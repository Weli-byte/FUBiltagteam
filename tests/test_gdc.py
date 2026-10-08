"""Tests for the GDC helpers. The local HTTP server below is a test double for transport logic
(retry, md5, resume); its payloads are arbitrary bytes, not data and not served by the project."""

import hashlib
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("requests")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import pandas as pd  # noqa: E402

from models.omics.gdc import (  # noqa: E402
    GdcFile,
    Md5MismatchError,
    build_filters,
    build_patient_map,
    download_file,
    inventory,
    md5_of,
    parse_file_hit,
    query_files,
    select_primary_rnaseq,
    write_manifest,
)

PAYLOAD = b"gene\tcount\nA\t1\n"


def _file(
    file_id: str = "f1",
    kind: str = "rnaseq",
    patient: str = "P-1",
    sample: str | None = "P-1-01A",
    sample_type: str | None = "Primary Tumor",
    payload: bytes = PAYLOAD,
) -> GdcFile:
    return GdcFile(
        kind,
        file_id,
        f"{file_id}.tsv",
        len(payload),
        hashlib.md5(payload).hexdigest(),
        "TCGA-TEST",
        patient,
        sample,
        sample_type,
    )


def _hit(**override: Any) -> dict[str, Any]:
    hit: dict[str, Any] = {
        "file_id": "abc",
        "file_name": "x/abc.tsv",
        "file_size": 10,
        "md5sum": "d41d8cd98f00b204e9800998ecf8427e",
        "cases": [
            {
                "submitter_id": "P-1",
                "project": {"project_id": "TCGA-TEST"},
                "samples": [{"submitter_id": "P-1-01A", "sample_type": "Primary Tumor"}],
            }
        ],
    }
    return {**hit, **override}


def test_filters_are_open_access_and_project_scoped() -> None:
    flt = build_filters("rnaseq", "TCGA-LUAD")
    pairs = {c["content"]["field"]: c["content"]["value"] for c in flt["content"]}
    assert pairs["access"] == "open" and pairs["cases.project.project_id"] == "TCGA-LUAD"
    assert pairs["analysis.workflow_type"] == "STAR - Counts"
    with pytest.raises(ValueError):
        build_filters("nope", "TCGA-LUAD")


def test_parse_hit_and_malformed_records_fail_loudly() -> None:
    f = parse_file_hit(_hit(), "rnaseq")
    assert f.file_name == "abc.tsv" and f.is_primary_tumor and f.patient_barcode == "P-1"
    with pytest.raises(ValueError, match="abc"):
        parse_file_hit(_hit(cases=[]), "rnaseq")
    with pytest.raises(ValueError, match="abc"):
        parse_file_hit({"file_id": "abc"}, "rnaseq")


def test_multi_sample_file_has_no_single_sample() -> None:
    cases = [
        {
            "submitter_id": "P-1",
            "project": {"project_id": "T"},
            "samples": [
                {"submitter_id": "a", "sample_type": "Primary Tumor"},
                {"submitter_id": "b", "sample_type": "Blood Derived Normal"},
            ],
        }
    ]
    f = parse_file_hit(_hit(cases=cases), "maf")
    assert f.sample_barcode is None and not f.is_primary_tumor


def test_manifest_has_gdc_client_columns(tmp_path: Path) -> None:
    write_manifest([_file()], tmp_path / "m.tsv")
    header, row = (tmp_path / "m.tsv").read_text().splitlines()
    assert header == "id\tfilename\tmd5\tsize\tstate" and row.startswith("f1\tf1.tsv\t")


def test_primary_selection_prefers_lowest_sample_barcode_and_ignores_normals() -> None:
    files = [
        _file("f3", sample="P-1-01B"),
        _file("f2", sample="P-1-01A"),
        _file("f9", sample="P-1-11A", sample_type="Solid Tissue Normal"),
        _file("f4", patient="P-2", sample="P-2-11A", sample_type="Solid Tissue Normal"),
    ]
    chosen = select_primary_rnaseq(files)
    assert set(chosen) == {"P-1"} and chosen["P-1"].file_id == "f2"


def test_patient_map_counts_and_inventory() -> None:
    clinical = [{"patient_barcode": p, "project_id": "TCGA-TEST"} for p in ("P-1", "P-2", "P-3")]
    files = [
        _file("r1", patient="P-1"),
        _file("r1b", patient="P-1", sample="P-1-01B"),
        _file("r2", patient="P-2", sample="P-2-01A"),
        _file("m1", "maf", "P-1", None, None),
        _file("m3", "maf", "P-3", None, None),
        _file("s1", "slides", "P-1", None, None),
    ]
    pm = build_patient_map(clinical, files).set_index("patient_barcode")
    assert (
        pm.loc["P-1", "n_rnaseq_primary_candidates"] == 2
        and pm.loc["P-1", "rnaseq_file_id"] == "r1"
    )
    assert pd.isna(pm.loc["P-3", "rnaseq_file_id"]) and pm.loc["P-3", "n_maf"] == 1
    table = inventory(build_patient_map(clinical, files))
    total = table.loc["TOTAL"]
    assert (
        total["patients"],
        total["has_rnaseq"],
        total["rnaseq_and_maf"],
        total["all_three"],
    ) == (3, 2, 1, 1)
    assert pm.index.is_unique  # a patient is never counted twice


@pytest.fixture
def server() -> Iterator[tuple[str, dict[str, Any]]]:
    state: dict[str, Any] = {"hits": 0, "fail_first": 0, "corrupt_first": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            state["hits"] += 1
            if state["fail_first"] > 0:
                state["fail_first"] -= 1
                self.send_response(503)
                self.end_headers()
                return
            body = b"corrupted" if state["corrupt_first"] > 0 else PAYLOAD
            if state["corrupt_first"] > 0:
                state["corrupt_first"] -= 1
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    httpd.shutdown()
    httpd.server_close()


def test_download_verifies_md5_and_second_run_skips(
    tmp_path: Path, server: tuple[str, dict[str, Any]]
) -> None:
    api, state = server
    f = _file()
    assert download_file(f, tmp_path, api=api, sleep=lambda _: None) == "downloaded"
    assert md5_of(tmp_path / "TCGA-TEST" / "rnaseq" / "f1" / "f1.tsv") == f.md5
    hits = state["hits"]
    assert download_file(f, tmp_path, api=api, sleep=lambda _: None) == "skipped"
    assert state["hits"] == hits  # no new request: resumable


def test_download_retries_transient_errors_and_bad_checksums(
    tmp_path: Path, server: tuple[str, dict[str, Any]]
) -> None:
    api, state = server
    state["fail_first"], state["corrupt_first"] = 1, 1
    assert download_file(_file(), tmp_path, api=api, sleep=lambda _: None) == "downloaded"
    assert not list(tmp_path.rglob("*.part"))


def test_persistent_md5_mismatch_raises_and_leaves_no_file(
    tmp_path: Path, server: tuple[str, dict[str, Any]]
) -> None:
    api, state = server
    state["corrupt_first"] = 99
    with pytest.raises(Md5MismatchError):
        download_file(_file(), tmp_path, api=api, attempts=3, sleep=lambda _: None)
    assert not list(tmp_path.rglob("f1.tsv*"))


def test_query_files_paginates_until_total(tmp_path: Path) -> None:
    hits_all = [_hit(file_id=f"id{i}") for i in range(5)]
    seen: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            start, size = body["from"], body["size"]
            reply = {"data": {"hits": hits_all[start : start + size], "pagination": {"total": 5}}}
            payload = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        found = query_files(
            "rnaseq", "TCGA-TEST", api=f"http://127.0.0.1:{httpd.server_address[1]}", page_size=2
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert [f.file_id for f in found] == [f"id{i}" for i in range(5)]
    assert [b["from"] for b in seen] == [0, 2, 4]
