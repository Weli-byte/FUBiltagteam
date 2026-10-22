"""TCGA barcode parsing, slide inventory, selection rule and patient map (mini sprint 1.4)."""

import pytest

pd = pytest.importorskip("pandas")

from models.histo.tcga.barcode import parse_slide_barcode, try_parse_slide_barcode  # noqa: E402
from models.histo.tcga.inventory import (  # noqa: E402
    build_slide_inventory,
    inventory_summary,
    patient_slide_map,
    select_slides,
    summary_markdown,
)


def test_parse_diagnostic_and_frozen_barcodes() -> None:
    dx = parse_slide_barcode("TCGA-A1-A0SB-01Z-00-DX1.0A1B2C3D-1234-5678-9ABC-DEF012345678.svs")
    assert dx.patient_barcode == "TCGA-A1-A0SB" and dx.tss == "A1"
    assert dx.sample_barcode == "TCGA-A1-A0SB-01Z" and dx.slide_barcode == "TCGA-A1-A0SB-01Z-00-DX1"
    assert dx.is_diagnostic and dx.is_primary_tumor and not dx.is_normal
    assert dx.sample_type == "Primary Solid Tumor" and dx.slide_number == 1

    ts = parse_slide_barcode("TCGA-A1-A0SB-11A-01-TS1.abc.svs")
    assert not ts.is_diagnostic and ts.is_normal and ts.slide_type == "TS"
    assert parse_slide_barcode("tcga-a1-a0sb-01a-01-bs2").slide_number == 2

    with pytest.raises(ValueError):
        parse_slide_barcode("slide_0001.svs")
    assert try_parse_slide_barcode("nope") is None


def _records() -> list[dict[str, object]]:
    def rec(
        fid: str, name: str, size: int, patient: str, project: str = "TCGA-LUAD"
    ) -> dict[str, object]:
        return {
            "file_id": fid,
            "file_name": name,
            "size": size,
            "md5": "0" * 32,
            "project_id": project,
            "patient_barcode": patient,
        }

    return [
        rec("f1", "TCGA-AA-0001-01Z-00-DX1.x.svs", 1_000_000_000, "TCGA-AA-0001"),
        rec("f2", "TCGA-AA-0001-01Z-00-DX2.x.svs", 900_000_000, "TCGA-AA-0001"),
        rec("f3", "TCGA-AA-0001-01A-01-TS1.x.svs", 300_000_000, "TCGA-AA-0001"),
        rec("f4", "TCGA-AA-0002-11A-01-DX1.x.svs", 800_000_000, "TCGA-AA-0002"),  # normal tissue
        rec("f5", "TCGA-BB-0003-01Z-00-DX1.x.svs", 2_000_000_000, "TCGA-BB-0003", "TCGA-BRCA"),
        rec("f6", "weird_name.svs", 10, "TCGA-BB-0004", "TCGA-BRCA"),
    ]


def test_inventory_selection_rule_and_patient_map() -> None:
    inventory = build_slide_inventory(_records())
    assert len(inventory) == 6 and inventory["slide_id"].is_unique
    assert inventory.set_index("slide_id").loc["f6", "barcode_parsed"] == False  # noqa: E712
    selected = select_slides(inventory)
    chosen = selected[selected["selected"]].set_index("patient_barcode")
    assert chosen.loc["TCGA-AA-0001", "slide_id"] == "f1"  # DX1 beats DX2
    assert "lowest DX number among 2" in chosen.loc["TCGA-AA-0001", "selection_reason"]
    assert "TCGA-AA-0002" not in chosen.index  # normal tissue slide is never a candidate
    assert chosen.loc["TCGA-BB-0003", "selection_reason"] == "only primary-tumor DX slide"

    qc = pd.DataFrame({"slide_id": ["f1", "f2"], "decision": ["exclude", "suspect"]})
    with_qc = select_slides(inventory, qc)
    assert (
        with_qc[with_qc["selected"]].set_index("patient_barcode").loc["TCGA-AA-0001", "slide_id"]
        == "f2"
    )

    table = patient_slide_map(selected, ["TCGA-AA-0001", "TCGA-CC-9999"])
    assert table["patient_barcode"].is_unique and len(table) == 4
    row = table.set_index("patient_barcode").loc["TCGA-AA-0001"]
    assert row["n_slides"] == 3 and row["n_dx"] == 2 and row["selected_slide_id"] == "f1"
    assert bool(row["in_leader_list"]) and not bool(
        table.set_index("patient_barcode").loc["TCGA-BB-0003", "in_leader_list"]
    )

    summary = inventory_summary(selected, table)
    assert summary.loc["TOTAL", "slides"] == 6 and summary.loc["TOTAL", "selected"] == 2
    assert summary.loc["TCGA-LUAD", "selected_gb"] == pytest.approx(1.0)
    assert summary.loc["TCGA-LUAD", "patients_in_leader_list"] == 1
    text = summary_markdown(summary, "46.0")
    assert "| TOTAL |" in text and "hasta düzeyi" in text


def test_duplicate_slide_ids_are_rejected() -> None:
    records = _records()
    records.append(dict(records[0]))
    with pytest.raises(ValueError, match="duplicate"):
        build_slide_inventory(records)
