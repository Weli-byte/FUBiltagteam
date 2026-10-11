"""Patient-level binary mutation labels from GDC masked somatic MAF files (mini sprint 1.5).

Definition (details in docs/etiket_tanimi.md):
* A patient is LABELED if at least one MAF file was downloaded for them AND the MAF contains a
  primary-tumour sample (TCGA sample type 01). Only that sample's variants are used.
* label = 1 if the gene has >= 1 variant whose Variant_Classification is in the configured list,
  else 0. A patient without a usable MAF has an UNKNOWN label (NA), never 0: absence of evidence of
  sequencing is not evidence of a wild-type gene.
* If a patient has several primary-tumour samples, the sample that was used for RNA-seq is
  preferred (so label and expression describe the same specimen); otherwise the lowest barcode.

MAF layout assumed (GDC masked somatic mutation, from documentation; verify with the real files):
leading ``#`` lines, then columns incl. ``Hugo_Symbol, Variant_Classification,
Tumor_Sample_Barcode`` and optionally ``HGVSp_Short, Exon_Number, Chromosome, Start_Position,
Reference_Allele, Tumor_Seq_Allele2, t_depth, t_alt_count, FILTER``.

These labels are research annotations, not clinical determinations.
"""

import gzip
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

LOG = logging.getLogger(__name__)
REQUIRED_MAF_COLUMNS = ("Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode")
OPTIONAL_MAF_COLUMNS = (
    "HGVSp_Short",
    "Exon_Number",
    "Chromosome",
    "Start_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
    "t_depth",
    "t_alt_count",
    "FILTER",
)
STATUS_LABELED = "labeled"
STATUS_NO_MAF = "no_maf"
STATUS_NO_PRIMARY = "no_primary_tumor_maf"


@dataclass(frozen=True)
class LabelRules:
    """Label rules, read from ``configs/omics/labels.yaml``."""

    genes: tuple[str, ...]
    variant_classifications: frozenset[str]
    primary_sample_type_code: str = "01"
    min_positives: int = 30
    imbalance_ratio: float = 0.15
    annotations: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def rules_from_config(cfg: DictConfig) -> LabelRules:
    genes = tuple(str(g) for g in cfg.genes)
    classes = frozenset(str(c) for c in cfg.variant_classifications)
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("labels config: `genes` must be a non-empty list without duplicates")
    if not classes:
        raise ValueError("labels config: `variant_classifications` must not be empty")
    kept: list[dict[str, Any]] = []
    for ann in (dict(a) for a in (cfg.get("annotations") or [])):
        if "name" not in ann or "gene" not in ann:
            raise ValueError(f"annotation {ann} needs `name` and `gene`")
        if ann["gene"] in genes:
            kept.append(ann)
        else:
            LOG.info("annotation %s skipped: gene %s is not in `genes`", ann["name"], ann["gene"])
    annotations = tuple(kept)
    return LabelRules(
        genes=genes,
        variant_classifications=classes,
        primary_sample_type_code=str(cfg.get("primary_sample_type_code", "01")),
        min_positives=int(cfg.get("min_positives", 30)),
        imbalance_ratio=float(cfg.get("imbalance_ratio", 0.15)),
        annotations=annotations,
    )


# ------------------------------------------------------------------------------- reading
def sample_of(barcode: str) -> str:
    """``TCGA-AB-1234-01A-11D-...`` -> sample barcode ``TCGA-AB-1234-01A`` (16 characters)."""
    return barcode[:16]


def patient_of(barcode: str) -> str:
    return barcode[:12]


def sample_type_code(barcode: str) -> str:
    return barcode[13:15]


def _leading_comment_lines(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    skip = 0
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            skip += 1
    return skip


@dataclass(frozen=True)
class MafRead:
    """Rows of the target genes plus every tumour sample barcode seen in the file."""

    rows: pd.DataFrame
    samples: frozenset[str]
    n_rows_total: int


def read_maf_file(path: Path | str, genes: Sequence[str]) -> MafRead:
    """Read one MAF (plain or gzip); keep rows of ``genes`` only, remember all sample barcodes."""
    path = Path(path)
    skip = _leading_comment_lines(path)
    header = pd.read_csv(path, sep="\t", skiprows=skip, nrows=0)
    missing = [c for c in REQUIRED_MAF_COLUMNS if c not in header.columns]
    if missing:
        raise ValueError(
            f"{path.name}: missing MAF columns {missing}; found {list(header.columns)}"
        )
    use = [c for c in (*REQUIRED_MAF_COLUMNS, *OPTIONAL_MAF_COLUMNS) if c in header.columns]
    frame = pd.read_csv(path, sep="\t", skiprows=skip, usecols=use, dtype=str)
    samples = frozenset(sample_of(b) for b in frame["Tumor_Sample_Barcode"].dropna().unique())
    rows = frame[frame["Hugo_Symbol"].isin(list(genes))].copy()
    return MafRead(rows=rows, samples=samples, n_rows_total=len(frame))


def collect_maf(
    patient_map: pd.DataFrame, raw_root: Path | str, genes: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every downloaded MAF of every patient.

    Returns ``(variants, samples)``: target-gene rows and a (patient, sample, file) table. A MAF
    whose barcodes belong to a different patient than the patient map says raises ``ValueError``.
    """
    raw_root = Path(raw_root)
    variant_parts: list[pd.DataFrame] = []
    sample_rows: list[dict[str, str]] = []
    missing: list[str] = []
    for _, row in patient_map[patient_map["n_maf"] > 0].iterrows():
        for file_id in row["maf_file_ids"]:
            folder = raw_root / row["project_id"] / "maf" / file_id
            files = sorted(p for p in folder.glob("*") if p.is_file()) if folder.exists() else []
            if not files:
                missing.append(str(folder))
                continue
            read = read_maf_file(files[0], genes)
            wrong = [s for s in read.samples if patient_of(s) != row["patient_barcode"]]
            if wrong:
                raise ValueError(
                    f"{files[0].name}: barcodes {wrong[:2]} do not belong to patient "
                    f"{row['patient_barcode']}"
                )
            sample_rows += [
                {"patient_barcode": row["patient_barcode"], "sample_barcode": s, "file_id": file_id}
                for s in read.samples
            ]
            if not read.rows.empty:
                variant_parts.append(
                    read.rows.assign(patient_barcode=row["patient_barcode"], file_id=file_id)
                )
    if missing:
        raise FileNotFoundError(f"{len(missing)} MAF files not found, e.g. {missing[:3]}")
    variants = pd.concat(variant_parts, ignore_index=True) if variant_parts else pd.DataFrame()
    samples = pd.DataFrame(sample_rows, columns=["patient_barcode", "sample_barcode", "file_id"])
    return variants, samples


# ------------------------------------------------------------------------------- labelling
@dataclass(frozen=True)
class LabelResult:
    labels: pd.DataFrame  # one row per patient in the patient map
    detail: pd.DataFrame  # every target-gene variant of the labelling sample (counted or not)
    n_excluded_by_class: dict[str, int]


def _choose_samples(
    samples: pd.DataFrame, patients: pd.DataFrame, primary_code: str
) -> dict[str, str]:
    """Patient -> labelling sample (primary tumour; RNA-seq sample preferred)."""
    rna = dict(zip(patients["patient_barcode"], patients["rnaseq_sample_barcode"], strict=True))
    chosen: dict[str, str] = {}
    primary = samples[samples["sample_barcode"].map(sample_type_code) == primary_code]
    for patient, group in primary.groupby("patient_barcode"):
        candidates = sorted(set(group["sample_barcode"]))
        raw = rna.get(patient)
        preferred = sample_of(raw) if isinstance(raw, str) else None
        pick = candidates[0]
        if preferred is not None and preferred in candidates:
            pick = preferred
        chosen[str(patient)] = pick
    return chosen


def annotate_variants(detail: pd.DataFrame, annotations: Sequence[dict[str, Any]]) -> pd.Series:
    """Semicolon-joined annotation names per variant row (``""`` if none)."""
    names = [[] for _ in range(len(detail))]  # type: list[list[str]]
    for ann in annotations:
        mask = detail["Hugo_Symbol"] == ann["gene"]
        if "hgvsp_short" in ann:
            mask &= detail["HGVSp_Short"].isin(list(ann["hgvsp_short"]))
        if "variant_classification" in ann:
            mask &= detail["Variant_Classification"].isin(list(ann["variant_classification"]))
        if "exon" in ann:
            exon = detail["Exon_Number"].fillna("").str.split("/").str[0]
            mask &= exon == str(ann["exon"])
        for position in np.flatnonzero(mask.to_numpy()):
            names[position].append(str(ann["name"]))
    return pd.Series([";".join(n) for n in names], index=detail.index, dtype="object")


def build_labels(
    variants: pd.DataFrame, samples: pd.DataFrame, patient_map: pd.DataFrame, rules: LabelRules
) -> LabelResult:
    """Binary label per patient and gene, plus the variant-level detail table."""
    patients = patient_map.sort_values("patient_barcode").reset_index(drop=True)
    chosen = _choose_samples(samples, patients, rules.primary_sample_type_code)

    status: list[str] = []
    label_sample: list[str | None] = []
    for _, row in patients.iterrows():
        patient = row["patient_barcode"]
        if int(row["n_maf"]) == 0:
            status.append(STATUS_NO_MAF)
        elif patient not in chosen:
            status.append(STATUS_NO_PRIMARY)
        else:
            status.append(STATUS_LABELED)
        label_sample.append(chosen.get(patient))
    labels = pd.DataFrame(
        {
            "patient_barcode": patients["patient_barcode"],
            "project_id": patients["project_id"],
            "label_status": status,
            "label_sample_barcode": label_sample,
        }
    )

    if variants.empty:
        detail = pd.DataFrame(columns=["patient_barcode", "Hugo_Symbol", "Variant_Classification"])
        detail["counted"] = pd.Series(dtype=bool)
        detail["annotations"] = pd.Series(dtype=object)
    else:
        variants = variants.assign(sample_barcode=variants["Tumor_Sample_Barcode"].map(sample_of))
        keep = variants["sample_barcode"] == variants["patient_barcode"].map(chosen)
        detail = variants[keep].copy()
        detail["counted"] = detail["Variant_Classification"].isin(rules.variant_classifications)
        # The same variant can appear in several MAF files of one patient; count it once.
        subset = [
            c
            for c in (
                "patient_barcode",
                "Hugo_Symbol",
                "Chromosome",
                "Start_Position",
                "Reference_Allele",
                "Tumor_Seq_Allele2",
                "Variant_Classification",
            )
            if c in detail.columns
        ]
        detail = detail.drop_duplicates(subset=subset)
        if "HGVSp_Short" not in detail.columns:
            detail["HGVSp_Short"] = np.nan
        if "Exon_Number" not in detail.columns:
            detail["Exon_Number"] = np.nan
        detail["annotations"] = annotate_variants(detail, rules.annotations)
        sort_cols = [
            c
            for c in ("patient_barcode", "Hugo_Symbol", "Chromosome", "Start_Position")
            if c in detail.columns
        ]
        detail = detail.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    excluded = detail[~detail["counted"].astype(bool)] if not detail.empty else detail
    n_excluded = (
        excluded.groupby("Variant_Classification").size().sort_index().to_dict()
        if not excluded.empty
        else {}
    )
    positive = (
        detail[detail["counted"].astype(bool)].groupby("patient_barcode")["Hugo_Symbol"].agg(set)
        if not detail.empty
        else pd.Series(dtype=object)
    )
    is_labeled = labels["label_status"] == STATUS_LABELED
    for gene in rules.genes:
        hit = labels["patient_barcode"].map(lambda p, g=gene: g in positive.get(p, set()))
        values = np.where(is_labeled, hit.astype(int), pd.NA)
        labels[gene] = pd.array(values, dtype="Int8")
    return LabelResult(labels=labels, detail=detail, n_excluded_by_class=n_excluded)


# --------------------------------------------------------------------------- class balance
def class_balance(labels: pd.DataFrame, rules: LabelRules) -> pd.DataFrame:
    """Per (project, gene): labeled/positive/negative/unknown counts and positive fraction."""
    rows: list[dict[str, Any]] = []
    groups = [*sorted(labels["project_id"].unique()), "ALL"]
    for project in groups:
        sub = labels if project == "ALL" else labels[labels["project_id"] == project]
        for gene in rules.genes:
            known = sub[gene].dropna()
            positives = int((known == 1).sum())
            negatives = int((known == 0).sum())
            rows.append(
                {
                    "project_id": project,
                    "gene": gene,
                    "patients": len(sub),
                    "labeled": positives + negatives,
                    "positive": positives,
                    "negative": negatives,
                    "unknown": len(sub) - positives - negatives,
                    "positive_fraction": (
                        positives / (positives + negatives)
                        if positives + negatives
                        else float("nan")
                    ),
                    "low_positives": positives < rules.min_positives,
                }
            )
    return pd.DataFrame(rows)


def recommend(balance: pd.DataFrame, rules: LabelRules) -> pd.DataFrame:
    """Rule-based suggestion per (project, gene). Thresholds come from the config; not tuned."""

    def text(row: pd.Series) -> str:
        if row["positive"] < rules.min_positives:
            return (
                f"yetersiz pozitif (<{rules.min_positives}): bu grupta ayrı model eğitme; "
                "havuzlu (pan-cancer) eğitimde kullan ve sonucu AUPRC ile dürüstçe raporla"
            )
        if row["negative"] < rules.min_positives:
            return "negatif örnek az: değerlendirme güvenilmez, grup tanımını gözden geçir"
        if row["positive_fraction"] < rules.imbalance_ratio:
            return (
                "dengesiz: ağırlıklı BCE (pos_weight) veya focal loss; AUROC yanında AUPRC raporla"
            )
        return "yeterince dengeli: standart BCE"

    out = balance[["project_id", "gene", "positive", "negative", "positive_fraction"]].copy()
    out["recommendation"] = balance.apply(text, axis=1)
    return out


def balance_markdown(
    balance: pd.DataFrame, rec: pd.DataFrame, rules: LabelRules, excluded: dict[str, int]
) -> str:
    lines = [
        "# Mutasyon Etiketi Sınıf Dengesi",
        "",
        "> `scripts/build_mutation_labels.py` tarafından üretilir. Yalnızca toplu sayılar içerir.",
        "",
        f"- Genler: {', '.join(rules.genes)}",
        f"- Mutasyon sayılan sınıflar: {', '.join(sorted(rules.variant_classifications))}",
        f"- Pozitif sayısı < {rules.min_positives} olan gruplar `low_positives` ile işaretlidir.",
        f"- 'dengesiz' eşiği: pozitif oranı < {rules.imbalance_ratio} (yapılandırmadan).",
        "- `unknown`: MAF'ı olmayan veya birincil tümör örneği bulunmayan hastalar.",
        "",
        "| proje | gen | hasta | etiketli | pozitif | negatif | bilinmeyen | oran | düşük |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in balance.iterrows():
        frac = "n/a" if pd.isna(r["positive_fraction"]) else f"{r['positive_fraction']:.3f}"
        low = "evet" if r["low_positives"] else "hayır"
        lines.append(
            f"| {r['project_id']} | {r['gene']} | {r['patients']} | {r['labeled']} "
            f"| {r['positive']} | {r['negative']} | {r['unknown']} | {frac} | {low} |"
        )
    lines += [
        "",
        "## Öneri (kural tabanlı; veriye bakılarak ayarlanmamış eşiklerle)",
        "",
        "| proje | gen | öneri |",
        "|---|---|---|",
    ]
    lines += [
        f"| {r['project_id']} | {r['gene']} | {r['recommendation']} |" for _, r in rec.iterrows()
    ]
    if excluded:
        lines += [
            "",
            "## Sayılmayan varyantlar (etiket örneğinde, hedef genlerde)",
            "",
            "| Variant_Classification | adet |",
            "|---|---|",
        ]
        lines += [f"| {k} | {v} |" for k, v in excluded.items()]
    lines.append("")
    return "\n".join(lines)


def save_balance_plot(balance: pd.DataFrame, rules: LabelRules, path: Path) -> Path:
    """Grouped bars of positive vs negative patients per gene and project."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    projects = list(balance["project_id"].unique())
    fig, axes = plt.subplots(1, len(rules.genes), figsize=(3.2 * len(rules.genes), 4), sharey=False)
    for ax, gene in zip(np.atleast_1d(axes), rules.genes, strict=True):
        sub = balance[balance["gene"] == gene].set_index("project_id").loc[projects]
        x = np.arange(len(projects))
        ax.bar(x - 0.2, sub["positive"], 0.4, label="pozitif")
        ax.bar(x + 0.2, sub["negative"], 0.4, label="negatif")
        ax.axhline(rules.min_positives, color="grey", ls="--", lw=0.8)
        ax.set_xticks(x, [p.replace("TCGA-", "") for p in projects], rotation=45)
        ax.set_title(gene)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
