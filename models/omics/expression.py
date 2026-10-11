"""GDC STAR-Counts parsing and sample x gene matrix assembly (mini sprint 1.4).

Expected file layout (GDC "STAR - Counts", written from documentation knowledge; verify on a real
file with ``scripts/preprocess_omics.py inspect <file>``): leading ``#`` comment lines, one header
line with at least ``gene_id, gene_name, gene_type, unstranded, tpm_unstranded``, four summary rows
whose ``gene_id`` starts with ``N_`` (unmapped, multimapping, ...), then one row per gene.

Nothing here learns parameters from data, so nothing here can leak information between splits.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
REQUIRED_COLUMNS = ("gene_id", "gene_name", "gene_type", "unstranded", "tpm_unstranded")
SUMMARY_ROW_PREFIX = "N_"


@dataclass(frozen=True)
class StarCounts:
    """One parsed STAR-Counts file (gene rows only)."""

    gene_id: np.ndarray[Any, Any]
    gene_name: np.ndarray[Any, Any]
    gene_type: np.ndarray[Any, Any]
    counts: np.ndarray[Any, Any]
    tpm: np.ndarray[Any, Any]


def parse_star_counts(path: Path | str) -> StarCounts:
    """Parse one STAR-Counts TSV; raise ``ValueError`` naming the file if the layout differs."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        skip = 0
        for line in handle:
            if not line.startswith("#"):
                break
            skip += 1
    frame = pd.read_csv(path, sep="\t", skiprows=skip, dtype={"gene_id": str, "gene_name": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; found {list(frame.columns)}")
    frame = frame[~frame["gene_id"].str.startswith(SUMMARY_ROW_PREFIX)]
    if frame.empty:
        raise ValueError(f"{path.name}: no gene rows after dropping '{SUMMARY_ROW_PREFIX}*' rows")
    if frame["gene_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicated gene_id values")
    return StarCounts(
        gene_id=frame["gene_id"].to_numpy(),
        gene_name=frame["gene_name"].fillna("").to_numpy(),
        gene_type=frame["gene_type"].fillna("unknown").to_numpy(),
        counts=frame["unstranded"].to_numpy(dtype=np.float64),
        tpm=frame["tpm_unstranded"].to_numpy(dtype=np.float32),
    )


def strip_version(gene_id: str) -> str:
    """``ENSG00000000003.15`` -> ``ENSG00000000003`` (also handles ``..._PAR_Y`` suffixes)."""
    return gene_id.split(".")[0]


def gene_dictionary(parsed: StarCounts) -> pd.DataFrame:
    """Gene table: versioned id, stripped Ensembl id, symbol, type, duplicate flags.

    Duplicate symbols or stripped ids are never dropped silently: they are flagged here and
    logged (``symbol_is_duplicate``, ``ensembl_is_duplicate``). The versioned ``gene_id`` stays the
    unique key (pseudo-autosomal ``_PAR_Y`` genes share a stripped id with their X copy).
    """
    table = pd.DataFrame(
        {
            "gene_id": parsed.gene_id,
            "ensembl_id": [strip_version(g) for g in parsed.gene_id],
            "gene_name": parsed.gene_name,
            "gene_type": parsed.gene_type,
        }
    )
    named = table["gene_name"] != ""
    table["symbol_is_duplicate"] = named & table["gene_name"].duplicated(keep=False)
    table["ensembl_is_duplicate"] = table["ensembl_id"].duplicated(keep=False)
    n_sym = int(table.loc[table["symbol_is_duplicate"], "gene_name"].nunique())
    n_ens = int(table.loc[table["ensembl_is_duplicate"], "ensembl_id"].nunique())
    if n_sym or n_ens:
        LOG.warning(
            "gene dictionary: %d duplicated symbols, %d duplicated Ensembl ids (flagged, kept)",
            n_sym,
            n_ens,
        )
    return table.set_index("gene_id", drop=False)


@dataclass(frozen=True)
class ExpressionData:
    """Assembled cohort matrix."""

    tpm: np.ndarray[Any, Any]  # (samples, genes) float32
    obs: pd.DataFrame  # one row per sample (index = sample barcode)
    genes: pd.DataFrame  # gene dictionary, row order == tpm columns


def sample_file_path(row: pd.Series, raw_root: Path) -> Path:
    return raw_root / row["project_id"] / "rnaseq" / row["rnaseq_file_id"] / row["rnaseq_file_name"]


def tss_of(sample_barcode: str) -> str:
    """Tissue source site code of a TCGA barcode (``TCGA-AB-1234-01A`` -> ``AB``)."""
    parts = sample_barcode.split("-")
    return parts[1] if len(parts) > 1 else "unknown"


def assemble_expression(
    patient_map: pd.DataFrame, raw_root: Path | str, log_every: int = 200
) -> ExpressionData:
    """Read the chosen RNA-seq file of every patient that has one into one TPM matrix.

    Library size (sum of raw counts) and the number of detected genes are kept as sample
    attributes for QC. All files must list the same genes in the same order, otherwise this raises.
    """
    raw_root = Path(raw_root)
    rows = patient_map[patient_map["rnaseq_file_id"].notna()].sort_values("patient_barcode")
    if rows.empty:
        raise ValueError("patient_map has no rows with an RNA-seq file")
    paths = [sample_file_path(r, raw_root) for _, r in rows.iterrows()]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} RNA-seq files not found, e.g. {missing[:3]}")

    first = parse_star_counts(paths[0])
    genes = gene_dictionary(first)
    tpm = np.empty((len(paths), len(first.gene_id)), dtype=np.float32)
    library = np.empty(len(paths), dtype=np.float64)
    for i, path in enumerate(paths):
        parsed = first if i == 0 else parse_star_counts(path)
        if not np.array_equal(parsed.gene_id, first.gene_id):
            raise ValueError(f"{path.name}: gene order/content differs from the first file")
        tpm[i] = parsed.tpm
        library[i] = parsed.counts.sum()
        if log_every and (i + 1) % log_every == 0:
            LOG.info("parsed %d/%d files", i + 1, len(paths))

    obs = pd.DataFrame(
        {
            "patient_barcode": rows["patient_barcode"].to_numpy(),
            "project_id": rows["project_id"].to_numpy(),
            "tss": [tss_of(b) for b in rows["rnaseq_sample_barcode"]],
            "library_size": library,
            "n_detected_genes": (tpm > 0).sum(axis=1),
        },
        index=pd.Index(rows["rnaseq_sample_barcode"].to_numpy(), name="sample_barcode"),
    )
    if obs.index.duplicated().any():
        raise ValueError("duplicated sample barcodes in the assembled cohort")
    return ExpressionData(tpm=tpm, obs=obs, genes=genes)


def read_id_list(path: Path | str) -> list[str]:
    """One barcode per line; blank lines and ``#`` comments are ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def select_samples(obs: pd.DataFrame, ids: Sequence[str]) -> np.ndarray[Any, Any]:
    """Positions of ``ids`` in ``obs``; ids may be sample or patient barcodes. Unknown ids raise."""
    wanted = set(ids)
    mask = obs.index.isin(wanted) | obs["patient_barcode"].isin(wanted)
    known = set(obs.index[mask]) | set(obs.loc[mask, "patient_barcode"])
    unknown = wanted - known
    if unknown:
        raise ValueError(f"{len(unknown)} ids not in the cohort, e.g. {sorted(unknown)[:3]}")
    return np.flatnonzero(mask)
