"""GDC (NCI Genomic Data Commons) access for TCGA open-access data (mini sprint 1.3).

What it does: query file metadata, write gdc-client compatible manifests, download with md5
verification and retry, and build the patient/sample/file mapping and an inventory.

IMPORTANT: the GDC field names and response shapes below were written from documentation
knowledge, not tested against the live API (the build environment had no access to it). Run
``scripts/download_gdc.py --dry-run`` first; any mismatch fails loudly with the offending field.

Only open-access files are requested (``access = open``); controlled-access data is never touched.
"""

import hashlib
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

GDC_API = "https://api.gdc.cancer.gov"
DEFAULT_PROJECTS = ("TCGA-LUAD", "TCGA-BRCA", "TCGA-COAD")
PRIMARY_TUMOR = "Primary Tumor"
MANIFEST_COLUMNS = ("id", "filename", "md5", "size", "state")

# kind -> equality filters on the GDC /files endpoint.
FILE_KINDS: dict[str, dict[str, str]] = {
    "rnaseq": {
        "data_category": "Transcriptome Profiling",
        "data_type": "Gene Expression Quantification",
        "analysis.workflow_type": "STAR - Counts",
    },
    "maf": {
        "data_category": "Simple Nucleotide Variation",
        "data_type": "Masked Somatic Mutation",
    },
    # Slides are only LISTED (patient mapping, Nisa's module); images are never downloaded here.
    "slides": {
        "data_type": "Slide Image",
        "experimental_strategy": "Diagnostic Slide",
    },
}
FILE_FIELDS = (
    "file_id",
    "file_name",
    "file_size",
    "md5sum",
    "access",
    "cases.submitter_id",
    "cases.project.project_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
)
CASE_FIELDS = (
    "submitter_id",
    "project.project_id",
    "demographic.gender",
    "demographic.vital_status",
    "demographic.age_at_index",
    "diagnoses.primary_diagnosis",
    "diagnoses.ajcc_pathologic_stage",
)


class Md5MismatchError(RuntimeError):
    """Raised when a downloaded file's md5 never matched the GDC-reported checksum."""


@dataclass(frozen=True)
class GdcFile:
    """One GDC file record, reduced to what the pipeline needs."""

    kind: str
    file_id: str
    file_name: str
    size: int
    md5: str
    project_id: str
    patient_barcode: str
    sample_barcode: str | None
    sample_type: str | None

    @property
    def is_primary_tumor(self) -> bool:
        return self.sample_type == PRIMARY_TUMOR


# ----------------------------------------------------------------------------- query building
def build_filters(kind: str, project: str) -> dict[str, Any]:
    """GDC filter JSON: open access + project + the kind's equality filters."""
    if kind not in FILE_KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {sorted(FILE_KINDS)}")
    equalities = {"access": "open", "cases.project.project_id": project, **FILE_KINDS[kind]}
    return {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": field, "value": value}}
            for field, value in equalities.items()
        ],
    }


def parse_file_hit(hit: dict[str, Any], kind: str) -> GdcFile:
    """Convert one ``/files`` hit into a :class:`GdcFile`; raise ``ValueError`` if malformed."""
    try:
        cases = hit["cases"]
        if len(cases) != 1:
            raise ValueError(f"expected exactly 1 case, found {len(cases)}")
        case = cases[0]
        samples = case.get("samples") or []
        single = samples[0] if len(samples) == 1 else {}
        return GdcFile(
            kind=kind,
            file_id=hit["file_id"],
            file_name=Path(hit["file_name"]).name,
            size=int(hit["file_size"]),
            md5=hit["md5sum"],
            project_id=case["project"]["project_id"],
            patient_barcode=case["submitter_id"],
            sample_barcode=single.get("submitter_id"),
            sample_type=single.get("sample_type"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unexpected GDC record for file {hit.get('file_id')}: {exc!r}") from exc


# ----------------------------------------------------------------------------------- HTTP
def _with_retry(
    action: Callable[[], Any],
    attempts: int = 5,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return action()
        except requests.RequestException as exc:
            last = exc
            if attempt < attempts - 1:
                sleep(base_delay * 2**attempt)
    raise RuntimeError(f"request failed after {attempts} attempts: {last!r}") from last


def _post_json(session: requests.Session, url: str, payload: dict[str, Any], timeout: float) -> Any:
    def call() -> Any:
        response = session.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    return _with_retry(call)


def query_files(
    kind: str,
    project: str,
    session: requests.Session | None = None,
    api: str = GDC_API,
    page_size: int = 500,
    timeout: float = 60.0,
) -> list[GdcFile]:
    """All open-access files of ``kind`` in ``project`` (paginated)."""
    session = session or requests.Session()
    files: list[GdcFile] = []
    offset = 0
    while True:
        payload = {
            "filters": build_filters(kind, project),
            "fields": ",".join(FILE_FIELDS),
            "format": "JSON",
            "size": page_size,
            "from": offset,
            "sort": "file_id",
        }
        data = _post_json(session, f"{api}/files", payload, timeout)["data"]
        hits = data["hits"]
        files.extend(parse_file_hit(hit, kind) for hit in hits)
        offset += len(hits)
        if not hits or offset >= int(data["pagination"]["total"]):
            return files


def fetch_clinical(
    project: str,
    session: requests.Session | None = None,
    api: str = GDC_API,
    page_size: int = 500,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Per-patient clinical rows for ``project`` straight from the /cases endpoint."""
    session = session or requests.Session()
    rows: list[dict[str, Any]] = []
    offset = 0
    flt = {"op": "=", "content": {"field": "project.project_id", "value": project}}
    while True:
        payload = {
            "filters": flt,
            "fields": ",".join(CASE_FIELDS),
            "format": "JSON",
            "size": page_size,
            "from": offset,
            "sort": "submitter_id",
        }
        data = _post_json(session, f"{api}/cases", payload, timeout)["data"]
        for hit in data["hits"]:
            diagnosis = (hit.get("diagnoses") or [{}])[0]
            demographic = hit.get("demographic") or {}
            rows.append(
                {
                    "patient_barcode": hit["submitter_id"],
                    "project_id": project,
                    "gender": demographic.get("gender"),
                    "vital_status": demographic.get("vital_status"),
                    "age_at_index": demographic.get("age_at_index"),
                    "primary_diagnosis": diagnosis.get("primary_diagnosis"),
                    "ajcc_pathologic_stage": diagnosis.get("ajcc_pathologic_stage"),
                }
            )
        offset += len(data["hits"])
        if not data["hits"] or offset >= int(data["pagination"]["total"]):
            return rows


def fetch_data_release(
    session: requests.Session | None = None, api: str = GDC_API, timeout: float = 30.0
) -> str:
    """GDC data release string (for provenance); ``'unknown'`` if the field is absent."""
    session = session or requests.Session()
    response = _with_retry(lambda: session.get(f"{api}/status", timeout=timeout))
    response.raise_for_status()
    return str(response.json().get("data_release", "unknown"))


# ------------------------------------------------------------------------- manifests, files
def write_manifest(files: Sequence[GdcFile], path: Path) -> None:
    """Write a gdc-client compatible manifest (tab separated)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(MANIFEST_COLUMNS)]
    lines += [f"{f.file_id}\t{f.file_name}\t{f.md5}\t{f.size}\treleased" for f in files]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def target_path(f: GdcFile, dest_root: Path) -> Path:
    return dest_root / f.project_id / f.kind / f.file_id / f.file_name


def download_file(
    f: GdcFile,
    dest_root: Path,
    session: requests.Session | None = None,
    api: str = GDC_API,
    attempts: int = 5,
    timeout: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Download one file with md5 verification. Returns ``'skipped'`` or ``'downloaded'``.

    A complete, md5-verified file is skipped, so an interrupted run resumes where it stopped.
    Raises :class:`Md5MismatchError` if every attempt produced a wrong checksum.
    """
    session = session or requests.Session()
    final = target_path(f, dest_root)
    if final.exists() and final.stat().st_size == f.size and md5_of(final) == f.md5:
        return "skipped"
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_suffix(final.suffix + ".part")
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with session.get(f"{api}/data/{f.file_id}", stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with part.open("wb") as out:
                    for block in response.iter_content(chunk_size=1 << 20):
                        out.write(block)
            if md5_of(part) == f.md5:
                os.replace(part, final)
                return "downloaded"
            last = Md5MismatchError(f"md5 mismatch for {f.file_id} ({f.file_name})")
        except requests.RequestException as exc:
            last = exc
        part.unlink(missing_ok=True)
        if attempt < attempts - 1:
            sleep(2**attempt)
    if isinstance(last, Md5MismatchError):
        raise last
    raise RuntimeError(f"download failed for {f.file_id}: {last!r}") from last


# ---------------------------------------------------------------- patient map and inventory
def select_primary_rnaseq(files: Sequence[GdcFile]) -> dict[str, GdcFile]:
    """One primary-tumor RNA-seq file per patient.

    Rule when a patient has several: lowest sample barcode (``...-01A`` before ``...-01B``), then
    lowest file id. The number of candidates is kept in the patient map so duplicates stay visible.
    """
    chosen: dict[str, GdcFile] = {}
    for f in sorted(files, key=lambda x: (x.sample_barcode or "", x.file_id)):
        if f.kind == "rnaseq" and f.is_primary_tumor:
            chosen.setdefault(f.patient_barcode, f)
    return chosen


def build_patient_map(clinical: Sequence[dict[str, Any]], files: Sequence[GdcFile]) -> pd.DataFrame:
    """One row per patient: clinical flag, chosen RNA-seq file, MAF and slide file ids."""
    chosen = select_primary_rnaseq(files)
    candidates: dict[str, int] = {}
    maf: dict[str, list[str]] = {}
    slides: dict[str, list[str]] = {}
    project_of: dict[str, str] = {c["patient_barcode"]: c["project_id"] for c in clinical}
    for f in files:
        project_of.setdefault(f.patient_barcode, f.project_id)
        if f.kind == "rnaseq" and f.is_primary_tumor:
            candidates[f.patient_barcode] = candidates.get(f.patient_barcode, 0) + 1
        elif f.kind == "maf":
            maf.setdefault(f.patient_barcode, []).append(f.file_id)
        elif f.kind == "slides":
            slides.setdefault(f.patient_barcode, []).append(f.file_id)
    clinical_ids = {c["patient_barcode"] for c in clinical}
    rows = []
    for patient in sorted(project_of):
        rna = chosen.get(patient)
        rows.append(
            {
                "patient_barcode": patient,
                "project_id": project_of[patient],
                "has_clinical": patient in clinical_ids,
                "rnaseq_file_id": rna.file_id if rna else None,
                "rnaseq_file_name": rna.file_name if rna else None,
                "rnaseq_sample_barcode": rna.sample_barcode if rna else None,
                "n_rnaseq_primary_candidates": candidates.get(patient, 0),
                "maf_file_ids": sorted(maf.get(patient, [])),
                "n_maf": len(maf.get(patient, [])),
                "slide_file_ids": sorted(slides.get(patient, [])),
                "n_slides": len(slides.get(patient, [])),
            }
        )
    return pd.DataFrame(rows)


def inventory(patient_map: pd.DataFrame) -> pd.DataFrame:
    """Per-project counts of patients with each data type and their intersections."""
    frame = patient_map.assign(
        has_rnaseq=patient_map["rnaseq_file_id"].notna(),
        has_maf=patient_map["n_maf"] > 0,
        has_slide=patient_map["n_slides"] > 0,
    )
    frame["rnaseq_and_maf"] = frame["has_rnaseq"] & frame["has_maf"]
    frame["all_three"] = frame["rnaseq_and_maf"] & frame["has_slide"]
    columns = [
        "has_clinical",
        "has_rnaseq",
        "has_maf",
        "has_slide",
        "rnaseq_and_maf",
        "all_three",
    ]
    table = frame.groupby("project_id")[columns].sum().astype(int)
    table.insert(0, "patients", frame.groupby("project_id").size())
    table.loc["TOTAL"] = table.sum()
    return table


def inventory_markdown(table: pd.DataFrame, release: str, projects: Sequence[str]) -> str:
    """Aggregated (no patient-level) inventory report for ``docs/veri_envanteri.md``."""
    header = "| Proje | " + " | ".join(table.columns) + " |"
    sep = "|---|" + "---|" * len(table.columns)
    body = [
        f"| {idx} | " + " | ".join(str(int(v)) for v in row) + " |" for idx, row in table.iterrows()
    ]
    return "\n".join(
        [
            "# Veri Envanteri (TCGA / GDC)",
            "",
            "> `scripts/download_gdc.py` tarafından üretilir. Yalnızca toplu sayılar içerir.",
            "",
            f"- GDC veri sürümü: `{release}`",
            f"- Projeler: {', '.join(projects)}",
            "- `has_rnaseq`: birincil tümör (Primary Tumor) RNA-seq dosyası olan hasta",
            "- `rnaseq_and_maf`, `all_three`: kesişimler (RNA-seq + MAF; + tanı slaydı)",
            "",
            header,
            sep,
            *body,
            "",
        ]
    )
