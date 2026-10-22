"""CAMELYON annotations -> masks -> patch labels, official splits, resumable download (1.5)."""

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from models.histo.camelyon.annotations import (  # noqa: E402
    label_patches,
    overlay_mask,
    parse_asap_xml,
    patch_tumor_fractions,
    polygons_to_mask,
    slide_has_tumor,
    unknown_groups,
)
from models.histo.camelyon.splits import (  # noqa: E402
    PARTIALLY_ANNOTATED,
    build_splits,
    load_splits,
    official_split,
    read_reference_csv,
    slide_id_from_path,
    split_of,
    write_splits,
)
from tests.histo_fixtures import make_xml  # noqa: E402


def test_parse_xml_reads_polygons_in_order(tmp_path: Path) -> None:
    xml = make_xml(tmp_path, "tumor_001", exclusion_rect=(1000, 400, 1100, 500))
    polygons = parse_asap_xml(xml)
    assert [p.group for p in polygons] == ["Tumor", "Exclusion"]
    assert polygons[0].points.shape == (4, 2) and polygons[0].points[0].tolist() == [900.0, 300.0]
    assert polygons[0].area_level0 == pytest.approx(400 * 400)
    assert slide_has_tumor(polygons) and unknown_groups(polygons) == set()


def test_mask_area_matches_polygon_and_exclusion_is_cut(tmp_path: Path) -> None:
    xml = make_xml(tmp_path, "tumor_002", exclusion_rect=(1000, 400, 1100, 500))
    polygons = parse_asap_xml(xml)
    mask = polygons_to_mask(polygons, 1536, 1024, downsample=4.0)
    assert mask.shape == (256, 384) and mask.dtype == np.uint8
    expected = (400 * 400 - 100 * 100) / 16.0
    assert mask.sum() == pytest.approx(expected, rel=0.03)
    assert mask[110, 260] == 0  # inside the exclusion hole
    legacy = parse_asap_xml(make_xml(tmp_path, "legacy", group="_0"))
    # fillPoly includes the boundary row/column, hence the small relative tolerance
    assert polygons_to_mask(legacy, 1536, 1024, 4.0).sum() == pytest.approx(
        400 * 400 / 16, rel=0.03
    )
    assert unknown_groups(parse_asap_xml(make_xml(tmp_path, "odd", group="Other"))) == {"Other"}


def test_patch_labels_from_mask(tmp_path: Path) -> None:
    polygons = parse_asap_xml(make_xml(tmp_path, "tumor_003"))
    mask = polygons_to_mask(polygons, 1536, 1024, downsample=2.0)
    coords = np.array([[900, 300], [1100, 500], [0, 0], [1250, 650]], dtype=np.int32)
    fractions = patch_tumor_fractions(mask, 2.0, coords, 128)
    assert fractions[0] == pytest.approx(1.0) and fractions[1] == pytest.approx(1.0)
    assert fractions[2] == 0.0 and 0.0 < fractions[3] < 1.0
    labels = label_patches(fractions, tumor_min=0.5, normal_max=0.0)
    assert labels.tolist()[:3] == [1, 1, 0]
    assert labels[3] in (-1, 1)
    rgb = np.full((128, 192, 3), 230, dtype=np.uint8)
    over = overlay_mask(rgb, mask)
    assert over.shape == rgb.shape and over[75, 140].tolist() != [230, 230, 230]


def test_official_split_and_reference_csv(tmp_path: Path) -> None:
    assert official_split("tumor_001") == "train" and official_split("test_001") == "test"
    assert slide_id_from_path("/x/Test_001.tif") == "test_001"
    with pytest.raises(ValueError):
        official_split("patient_001")
    ref = tmp_path / "reference.csv"
    ref.write_text(
        "image,type,class,center\n"
        "test_001,Tumor,macro,RUMC\n"
        "test_002,Normal,negative,UMCU\n"
        "normal_001,Normal,negative,RUMC\n"
        "tumor_001,Tumor,micro,RUMC\n",
        encoding="utf-8",
    )
    rows = read_reference_csv(ref)
    assert rows["test_001"].label == "tumor" and rows["test_001"].centre == "RUMC"
    assert rows["test_002"].label == "normal" and rows["test_002"].lesion_class == "negative"
    headerless = tmp_path / "ref2.csv"
    headerless.write_text("test_003,Tumor,micro,UMCU\n", encoding="utf-8")
    assert read_reference_csv(headerless)["test_003"].centre == "UMCU"


def test_build_splits_counts_overlap_and_flags(tmp_path: Path) -> None:
    ref = tmp_path / "reference.csv"
    ref.write_text(
        "image,type,class,center\ntest_001,Tumor,macro,RUMC\ntest_002,Normal,negative,UMCU\n"
        "tumor_010,Tumor,macro,RUMC\n",
        encoding="utf-8",
    )
    ids = ["normal_001", "normal_002", "tumor_010", "test_001", "test_002", "test_003"]
    doc = build_splits(ids, read_reference_csv(ref))
    assert doc["counts"] == {
        "train": 3,
        "train_tumor": 1,
        "train_normal": 2,
        "test": 3,
        "test_tumor": 1,
        "test_normal": 1,
        "test_unlabelled": 1,
    }
    assert doc["centres"] == ["RUMC", "UMCU"]
    flagged = {e["slide_id"] for e in doc["train"] if e["partially_annotated"]}
    assert flagged == {"tumor_010"} and "tumor_010" in PARTIALLY_ANNOTATED
    lookup = split_of(doc)
    assert lookup["normal_001"] == "train" and lookup["test_003"] == "test"
    assert not (set(lookup) & set()) and len(set(lookup.values())) == 2
    write_splits(doc, tmp_path / "camelyon_splits.json")
    assert load_splits(tmp_path / "camelyon_splits.json")["counts"]["train"] == 3
    bad = tmp_path / "bad.csv"
    bad.write_text("image,type,class,center\ntumor_010,Normal,negative,RUMC\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reference says"):
        build_splits(ids, read_reference_csv(bad))


# ----------------------------------------------------------------------- download (HTTP double)
PAYLOAD = bytes(range(256)) * 40  # 10 240 bytes


@pytest.fixture
def bucket() -> Iterator[tuple[str, dict[str, Any]]]:
    pytest.importorskip("requests")
    state: dict[str, Any] = {"hits": 0, "ranges": [], "fail_first": 0}
    listing = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        "<IsTruncated>false</IsTruncated>"
        f"<Contents><Key>CAMELYON16/images/tumor_001.tif</Key><Size>{len(PAYLOAD)}</Size>"
        '<ETag>"0123456789abcdef0123456789abcdef-3"</ETag></Contents>'
        "<Contents><Key>CAMELYON16/images/</Key><Size>0</Size></Contents>"
        "</ListBucketResult>"
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            state["hits"] += 1
            if self.path.startswith("/?"):
                self.send_response(200)
                self.send_header("Content-Length", str(len(listing)))
                self.end_headers()
                self.wfile.write(listing)
                return
            if state["fail_first"] > 0:
                state["fail_first"] -= 1
                self.send_response(503)
                self.end_headers()
                return
            start = 0
            header = self.headers.get("Range")
            if header:
                start = int(header.split("=")[1].split("-")[0])
                state["ranges"].append(start)
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


def test_listing_manifest_and_resumable_download(
    tmp_path: Path, bucket: tuple[str, dict[str, Any]]
) -> None:
    from models.histo.camelyon.download import (
        download_object,
        list_bucket,
        read_manifest,
        write_manifest,
    )

    base, state = bucket
    objects = list_bucket(base, "CAMELYON16/images/")
    assert [o.key for o in objects] == ["CAMELYON16/images/tumor_001.tif"]
    assert objects[0].md5 is None  # multipart ETag is not an md5
    write_manifest(objects, tmp_path / "manifest.csv")
    assert read_manifest(tmp_path / "manifest.csv") == objects

    obj = objects[0]
    part = tmp_path / "data" / obj.key
    part.parent.mkdir(parents=True)
    part.with_suffix(".tif.part").write_bytes(PAYLOAD[:4000])  # simulate an interrupted run
    assert download_object(base, obj, tmp_path / "data", sleep=lambda _: None) == "downloaded"
    assert state["ranges"] == [4000] and part.read_bytes() == PAYLOAD
    assert download_object(base, obj, tmp_path / "data", sleep=lambda _: None) == "skipped"

    state["fail_first"] = 1
    part.unlink()
    assert download_object(base, obj, tmp_path / "data", sleep=lambda _: None) == "downloaded"
    assert part.read_bytes() == PAYLOAD
