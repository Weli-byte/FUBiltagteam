"""TCGA slide inventory, selection rule and patient map (mini sprint 1.4).

Bu dosya ne işe yarar? GDC'den listelenen slayt kayıtlarını (dosya kimliği, ad, boyut, hasta)
barkod bilgisiyle zenginleştirip bir envanter tablosu yapar; her hasta için hangi slaytın
kullanılacağını yazılı bir kurala göre seçer (birincil tümörün tanı slaydı; birden fazlaysa en
düşük DX numarası, varsa QC skoru yüksek olan) ve proje liderinin hasta listesiyle kesişimi
çıkarır. ``slide_id`` olarak GDC dosya kimliği (UUID) kullanılır: anonimdir ve proje liderinin
``slide_splits.parquet`` dosyasındaki ``slide_file_id`` ile birebir aynıdır.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from models.histo.tcga.barcode import try_parse_slide_barcode

INVENTORY_COLUMNS = (
    "slide_id",
    "file_name",
    "size_bytes",
    "md5",
    "project_id",
    "patient_barcode",
    "sample_type_code",
    "sample_type",
    "slide_type",
    "slide_number",
    "is_diagnostic",
    "is_primary_tumor",
    "barcode_parsed",
)


def build_slide_inventory(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """One row per GDC slide file with parsed barcode fields.

    ``records`` need ``file_id``, ``file_name``, ``size``, ``md5``, ``project_id``,
    ``patient_barcode`` (the fields ``models.omics.gdc.GdcFile`` carries).
    """
    rows: list[dict[str, Any]] = []
    for rec in records:
        parsed = try_parse_slide_barcode(str(rec["file_name"]))
        rows.append(
            {
                "slide_id": str(rec["file_id"]),
                "file_name": str(rec["file_name"]),
                "size_bytes": int(rec["size"]),
                "md5": rec.get("md5"),
                "project_id": rec["project_id"],
                "patient_barcode": rec["patient_barcode"],
                "sample_type_code": parsed.sample_type_code if parsed else None,
                "sample_type": parsed.sample_type if parsed else None,
                "slide_type": parsed.slide_type if parsed else None,
                "slide_number": parsed.slide_number if parsed else None,
                "is_diagnostic": bool(parsed.is_diagnostic) if parsed else False,
                "is_primary_tumor": bool(parsed.is_primary_tumor) if parsed else False,
                "barcode_parsed": parsed is not None,
            }
        )
    frame = pd.DataFrame(rows, columns=list(INVENTORY_COLUMNS))
    if frame["slide_id"].duplicated().any():
        raise ValueError("duplicate slide_id in GDC records")
    return frame


def select_slides(inventory: pd.DataFrame, qc: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mark one slide per patient as ``selected`` with a written rule (``selection_reason``).

    Rule: (1) only diagnostic (DX) slides of primary tumor samples (code 01) are candidates;
    (2) if ``qc`` (columns ``slide_id``, ``decision``) is given, excluded slides are dropped and
    ``use`` beats ``suspect``; (3) ties: lowest slide number, then lowest slide_id (stable).
    """
    frame = inventory.copy()
    frame["selected"] = False
    frame["selection_reason"] = ""
    candidates = frame[frame["is_diagnostic"] & frame["is_primary_tumor"]].copy()
    candidates["qc_rank"] = 0
    if qc is not None:
        decision = dict(zip(qc["slide_id"], qc["decision"], strict=True))
        candidates["qc_decision"] = candidates["slide_id"].map(decision).fillna("unknown")
        candidates = candidates[candidates["qc_decision"] != "exclude"]
        rank = {"use": 0, "unknown": 1, "suspect": 2}
        candidates["qc_rank"] = candidates["qc_decision"].map(rank)
    candidates = candidates.sort_values(["patient_barcode", "qc_rank", "slide_number", "slide_id"])
    chosen = candidates.drop_duplicates("patient_barcode", keep="first")
    n_per_patient = candidates.groupby("patient_barcode").size()
    frame.loc[chosen.index, "selected"] = True
    for idx, row in chosen.iterrows():
        n = int(n_per_patient[row["patient_barcode"]])
        reason = "only primary-tumor DX slide" if n == 1 else f"lowest DX number among {n}"
        if qc is not None:
            reason += f"; qc={row['qc_decision']}"
        frame.loc[idx, "selection_reason"] = reason
    return frame


def patient_slide_map(inventory: pd.DataFrame, patients: Sequence[str]) -> pd.DataFrame:
    """Intersection with the project leader's patient list (one row per patient).

    Columns: ``patient_barcode``, ``project_id``, ``n_slides`` (DX+frozen), ``n_dx``,
    ``selected_slide_id``, ``in_leader_list``.
    """
    inv = inventory.copy()
    if "selected" not in inv.columns:
        inv = select_slides(inv)
    grouped = inv.groupby("patient_barcode")
    table = pd.DataFrame(
        {
            "project_id": grouped["project_id"].first(),
            "n_slides": grouped.size(),
            "n_dx": grouped["is_diagnostic"].sum().astype(int),
        }
    )
    selected = inv[inv["selected"]].set_index("patient_barcode")["slide_id"]
    table["selected_slide_id"] = selected.reindex(table.index)
    table["in_leader_list"] = table.index.isin(set(patients))
    table = table.reset_index()
    if table["patient_barcode"].duplicated().any():  # pragma: no cover - groupby guarantees this
        raise AssertionError("a patient appears twice in the patient/slide map")
    return table


def inventory_summary(
    inventory: pd.DataFrame, patient_map: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-project aggregates (no patient ids): slides, DX slides, patients, selected, GB."""
    inv = inventory.copy()
    if "selected" not in inv.columns:
        inv = select_slides(inv)
    summary = inv.groupby("project_id").agg(
        slides=("slide_id", "size"),
        dx_slides=("is_diagnostic", "sum"),
        patients=("patient_barcode", "nunique"),
        selected=("selected", "sum"),
        total_gb=("size_bytes", lambda s: float(s.sum()) / 1e9),
        selected_gb=("size_bytes", lambda s: float(s[inv.loc[s.index, "selected"]].sum()) / 1e9),
    )
    if patient_map is not None:
        in_list = patient_map[patient_map["in_leader_list"]].groupby("project_id")
        summary["patients_in_leader_list"] = in_list.size().reindex(summary.index).fillna(0)
        summary["selected_in_leader_list"] = (
            in_list["selected_slide_id"].count().reindex(summary.index).fillna(0)
        )
    summary.loc["TOTAL"] = summary.sum(numeric_only=True)
    for column in ("slides", "dx_slides", "patients", "selected"):
        summary[column] = summary[column].astype(int)
    return summary.round(2)


def summary_markdown(summary: pd.DataFrame, release: str, note: str = "") -> str:
    header = "| Proje | " + " | ".join(summary.columns) + " |"
    sep = "|---|" + "---|" * len(summary.columns)

    def _fmt(value: Any) -> str:  # keep ints as ints; iterrows() would upcast them to float
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    body = [
        f"| {idx} | " + " | ".join(_fmt(v) for v in record.values()) + " |"
        for idx, record in summary.to_dict("index").items()
    ]
    lines = [
        "# TCGA Tanı Slaytı Envanteri",
        "",
        "> `scripts/download_tcga_slides.py` tarafından üretilir. Yalnızca toplu sayılar içerir;",
        "> hasta düzeyi tablolar (`slide_inventory.parquet`, `patient_slide_map.parquet`)",
        "> `data/interim/` altında kalır ve git'e girmez.",
        "",
        f"- GDC veri sürümü: `{release}`",
        "- Seçim kuralı: hasta başına bir birincil tümör (01) tanı slaydı (DX); birden fazlaysa en",
        "  düşük DX numarası; QC sonucu varsa `exclude` elenir, `use` > `suspect`.",
        "- `selected_gb`: yalnızca seçilen slaytların toplam boyutu (indirme planı için).",
        "",
        header,
        sep,
        *body,
        "",
    ]
    if note:
        lines += [note, ""]
    return "\n".join(lines)


def slide_metadata(path: Path, mpp_override: float | None = None) -> dict[str, Any]:
    """OpenSlide properties of a downloaded slide (mpp, magnification, size, levels, vendor)."""
    from models.histo.wsi.reader import SlideReader

    with SlideReader(path, mpp_override=mpp_override) as reader:
        return reader.info.to_dict()
