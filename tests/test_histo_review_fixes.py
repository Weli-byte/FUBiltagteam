"""Regression tests for defects found in the Sprint 1 real-data code review (2026-10-18)."""

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("openslide")
pytest.importorskip("cv2")
pytest.importorskip("tifffile")
pytest.importorskip("h5py")
pytest.importorskip("numpy")
pytest.importorskip("requests")

from models.histo.camelyon.download import RemoteObject, download_object  # noqa: E402
from models.histo.config import redact_path  # noqa: E402
from models.histo.patches.extract import SlideSpec, extract_one  # noqa: E402
from models.histo.patches.grid import PatchGridConfig  # noqa: E402
from models.histo.wsi.tissue import TissueMaskConfig  # noqa: E402
from scripts.make_sample_pack import read_pack_slides  # noqa: E402
from tests.histo_fixtures import make_slide  # noqa: E402

PAYLOAD = bytes(range(256)) * 20


@pytest.fixture
def s3_like() -> Iterator[tuple[str, dict[str, Any]]]:
    """Serves PAYLOAD with Range support and answers 416 when the offset is past the end."""
    state: dict[str, Any] = {"gets": 0, "ranges": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            state["gets"] += 1
            header = self.headers.get("Range")
            start = int(header.split("=")[1].split("-")[0]) if header else 0
            state["ranges"].append(start)
            if start >= len(PAYLOAD):
                self.send_response(416)
                self.end_headers()
                return
            body = PAYLOAD[start:]
            self.send_response(206 if header else 200)
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


def test_complete_part_file_is_finished_without_a_request(
    tmp_path: Path, s3_like: tuple[str, dict[str, Any]]
) -> None:
    base, state = s3_like
    obj = RemoteObject("CAMELYON16/images/x.tif", len(PAYLOAD), None)
    final = tmp_path / obj.key
    final.parent.mkdir(parents=True)
    final.with_suffix(".tif.part").write_bytes(PAYLOAD)  # interrupted after the last byte
    assert download_object(base, obj, tmp_path, sleep=lambda _: None) == "downloaded"
    assert state["gets"] == 0 and final.read_bytes() == PAYLOAD


def test_416_restarts_from_scratch(tmp_path: Path, s3_like: tuple[str, dict[str, Any]]) -> None:
    """A resume offset the server rejects (416) must restart from byte 0, not loop on 416."""
    base, state = s3_like
    obj = RemoteObject("CAMELYON16/images/y.tif", len(PAYLOAD) + 100, None)  # listing too long
    final = tmp_path / obj.key
    final.parent.mkdir(parents=True)
    final.with_suffix(".tif.part").write_bytes(PAYLOAD)  # offset == real length -> 416
    with pytest.raises(RuntimeError):
        download_object(base, obj, tmp_path, attempts=2, sleep=lambda _: None)
    assert state["ranges"] == [len(PAYLOAD), 0]  # 416 on the resume, then a fresh full request


def test_extraction_regenerates_when_config_changes(tmp_path: Path) -> None:
    path, _ = make_slide(tmp_path, "slide_cfg")
    mask = TissueMaskConfig(min_object_px=20, max_hole_px=200)
    grid_a = PatchGridConfig(patch_px=128, target_mpp=0.5, thumbnail_px=512)
    grid_b = PatchGridConfig(patch_px=64, target_mpp=0.5, thumbnail_px=512)
    first = extract_one(SlideSpec("slide_cfg", path), tmp_path / "out", grid_a, mask)
    assert first.status == "done"
    assert (
        extract_one(SlideSpec("slide_cfg", path), tmp_path / "out", grid_a, mask).status
        == "skipped"
    )
    changed = extract_one(SlideSpec("slide_cfg", path), tmp_path / "out", grid_b, mask)
    assert changed.status == "done" and changed.n_patches > first.n_patches


def test_redact_path_removes_barcoded_file_names() -> None:
    path = Path("data/raw/tcga_slides/TCGA-LUAD/slides/uuid-1/TCGA-75-7030-01Z-00-DX1.ABCD.svs")
    message = f"FileNotFoundError: {path} / {path.name} / TCGA-75-7030-01Z-00-DX1"
    out = redact_path(message, path, "uuid-1")
    assert "TCGA-75" not in out and "uuid-1" in out


def _write_list(tmp_path: Path, rows: str) -> Path:
    csv_path = tmp_path / "list.csv"
    csv_path.write_text(
        "slide_id,path,source,split,slide_label,annotation_xml,mpp_override,centre\n" + rows,
        encoding="utf-8",
    )
    return csv_path


def test_pack_list_normalises_and_validates_labels(tmp_path: Path) -> None:
    rows = (
        "abc-1,slide.svs,TCGA,Unassigned,Tumor,,0.5,TSS75\ntumor_001,t.tif,camelyon16,Train,,,,\n"
    )
    slides = read_pack_slides(_write_list(tmp_path, rows))
    assert slides[0].slide_label == "tumor" and slides[0].split == "unassigned"
    assert slides[0].source == "tcga" and slides[0].mpp_override == 0.5
    assert slides[1].split == "train" and slides[1].path == tmp_path / "t.tif"
    with pytest.raises(ValueError, match="slide_label"):
        read_pack_slides(_write_list(tmp_path, "abc-2,s.svs,tcga,unassigned,cancer,,,\n"))
    with pytest.raises(ValueError, match="split"):
        read_pack_slides(_write_list(tmp_path, "abc-3,s.svs,tcga,holdout,tumor,,,\n"))


def test_pack_list_enforces_the_locked_camelyon_test_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="official split"):
        read_pack_slides(_write_list(tmp_path, "test_001,t.tif,camelyon16,train,tumor,,,\n"))
    with pytest.raises(ValueError, match="official split"):
        read_pack_slides(_write_list(tmp_path, "normal_002,n.tif,camelyon16,test,normal,,,\n"))
