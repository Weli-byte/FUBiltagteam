"""Patient-level splits, leakage checks and test-set locking (mini sprint 1.6).

Design
------
* The unit of splitting is the PATIENT. Samples, slides and patches inherit their patient's split,
  so one patient can never sit in two subsets.
* Multi-label stratification (iterative stratification, Sechidis et al. 2011) balances cancer type,
  each target gene's positives, label availability and modality availability across subsets.
* ``split``       : train / val / test (default 70/15/15).
* ``cv_fold``     : 0..K-1 inside the train+val pool (hyper-parameter search); -1 for test patients.
* ``split_site``  : alternative scenario where whole tissue-source sites (per project) are held out
                    as test, to measure robustness to unseen institutions.
* The test split is created ONCE per version, recorded by a SHA-256 hash in a manifest, never
  overwritten, and only handed out with an explicit ``final_evaluation=True``.
"""

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.omics.expression import tss_of

LOG = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "val", "test")


# ------------------------------------------------------------------------ stratification
def iterative_stratify(
    indicators: np.ndarray[Any, Any], ratios: Sequence[float], seed: int
) -> np.ndarray[Any, Any]:
    """Assign each row to one of ``len(ratios)`` subsets, balancing every indicator column.

    Implements iterative stratification: repeatedly take the rarest remaining label and give its
    examples, one by one, to the subset that still needs that label most (ties: subset that still
    needs most examples overall, then random). Rows without any positive label fill the subsets
    by remaining capacity.

    Args:
        indicators: ``(n, L)`` 0/1 matrix (a row may have several 1s).
        ratios: Target subset proportions (normalised to sum to 1).
        seed: Seed for the random tie-breaks (deterministic).

    Returns:
        Integer array of subset indices, one per row.
    """
    y = np.asarray(indicators, dtype=np.int64)
    if y.ndim != 2:
        raise ValueError("indicators must be 2-D (n_rows, n_labels)")
    n, n_labels = y.shape
    weights = np.asarray(ratios, dtype=np.float64)
    if (weights <= 0).any() or len(weights) < 2:
        raise ValueError("ratios must be >= 2 positive numbers")
    weights = weights / weights.sum()
    k = len(weights)
    rng = np.random.default_rng(seed)

    assigned = np.full(n, -1, dtype=np.int64)
    remaining = np.ones(n, dtype=bool)
    wanted = weights * n  # examples each subset still needs
    wanted_label = weights[:, None] * y.sum(axis=0)[None, :]  # per subset and label

    def pick(candidates: np.ndarray[Any, Any], scores: np.ndarray[Any, Any]) -> int:
        best = candidates[np.isclose(scores[candidates], scores[candidates].max())]
        return int(best[0] if best.size == 1 else rng.choice(best))

    while True:
        counts = y[remaining].sum(axis=0)
        active = np.flatnonzero(counts > 0)
        if active.size == 0:
            break
        rarest = active[counts[active] == counts[active].min()]
        label = int(rarest[0] if rarest.size == 1 else rng.choice(rarest))
        rows = np.flatnonzero(remaining & (y[:, label] == 1))
        rng.shuffle(rows)
        for row in rows:
            everyone = np.arange(k)
            by_label = everyone[np.isclose(wanted_label[:, label], wanted_label[:, label].max())]
            subset = pick(by_label, wanted) if by_label.size > 1 else int(by_label[0])
            assigned[row] = subset
            remaining[row] = False
            wanted_label[subset, y[row] == 1] -= 1
            wanted[subset] -= 1

    leftovers = np.flatnonzero(remaining)
    rng.shuffle(leftovers)
    for row in leftovers:
        subset = pick(np.arange(k), wanted)
        assigned[row] = subset
        wanted[subset] -= 1
    return assigned


def build_split_table(
    patient_map: pd.DataFrame, labels: pd.DataFrame, genes: Sequence[str]
) -> pd.DataFrame:
    """One row per patient: project, tissue source site, modality flags and gene labels."""
    missing = [g for g in genes if g not in labels.columns]
    if missing:
        raise ValueError(f"labels table lacks gene columns {missing}")
    base = patient_map[["patient_barcode", "project_id", "rnaseq_file_id", "n_slides"]]
    table = base.merge(
        labels[["patient_barcode", "label_status", *genes]],
        on="patient_barcode",
        how="left",
        validate="one_to_one",
    )
    if table["label_status"].isna().any():
        raise ValueError("some patients of the patient map are missing from the labels table")
    out = pd.DataFrame(
        {
            "patient_barcode": table["patient_barcode"],
            "project_id": table["project_id"],
            "tss": [tss_of(p) for p in table["patient_barcode"]],
            "has_rnaseq": table["rnaseq_file_id"].notna().to_numpy(),
            "has_slide": (table["n_slides"] > 0).to_numpy(),
            "labeled": (table["label_status"] == "labeled").to_numpy(),
        }
    )
    for gene in genes:
        out[gene] = table[gene].astype("Int8")
    out = out.sort_values("patient_barcode").reset_index(drop=True)
    if out["patient_barcode"].duplicated().any():
        raise ValueError("duplicated patients in the split table")
    return out


def indicator_matrix(
    table: pd.DataFrame, genes: Sequence[str]
) -> tuple[np.ndarray[Any, Any], list[str]]:
    """0/1 stratification matrix: project one-hot, gene positives, label and modality flags."""
    columns: dict[str, np.ndarray[Any, Any]] = {}
    for project in sorted(table["project_id"].unique()):
        columns[f"project={project}"] = (table["project_id"] == project).to_numpy()
    for gene in genes:
        columns[f"pos={gene}"] = (table[gene] == 1).fillna(False).to_numpy(dtype=bool)
    for flag in ("labeled", "has_slide", "has_rnaseq"):
        columns[flag] = table[flag].to_numpy(dtype=bool)
    matrix = np.stack([v.astype(np.int8) for v in columns.values()], axis=1)
    return matrix, list(columns)


def site_heldout_mask(table: pd.DataFrame, fraction: float, seed: int) -> np.ndarray[Any, Any]:
    """Test mask holding out WHOLE (project, tissue site) groups, ~``fraction`` per project."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(table), dtype=bool)
    for project in sorted(table["project_id"].unique()):
        in_project = (table["project_id"] == project).to_numpy()
        sizes = table[in_project].groupby("tss").size().sort_index()
        order = list(sizes.index)
        rng.shuffle(order)
        target = fraction * int(in_project.sum())
        held, total = [], 0
        for site in order:
            if total >= target:
                break
            held.append(site)
            total += int(sizes[site])
        test |= in_project & table["tss"].isin(held).to_numpy()
    return test


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    n_folds: int = 5
    site_holdout_fraction: float = 0.15
    min_positives_per_split: int = 5
    max_ratio_deviation: float = 0.03


@dataclass
class SplitResult:
    splits: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def make_splits(table: pd.DataFrame, genes: Sequence[str], cfg: SplitConfig) -> SplitResult:
    """Create ``split``, ``cv_fold`` and ``split_site`` for every patient (deterministic)."""
    table = table.sort_values("patient_barcode").reset_index(drop=True)
    matrix, _ = indicator_matrix(table, genes)
    names = np.array(SPLIT_NAMES)

    primary = iterative_stratify(matrix, cfg.ratios, cfg.seed)
    split = names[primary]
    pool = np.flatnonzero(primary < 2)
    cv_fold = np.full(len(table), -1, dtype=np.int64)
    cv_fold[pool] = iterative_stratify(matrix[pool], [1.0] * cfg.n_folds, cfg.seed + 1)

    test_site = site_heldout_mask(table, cfg.site_holdout_fraction, cfg.seed + 2)
    rest = np.flatnonzero(~test_site)
    train_share = cfg.ratios[0] / (cfg.ratios[0] + cfg.ratios[1])
    inner = iterative_stratify(matrix[rest], [train_share, 1 - train_share], cfg.seed + 3)
    split_site = np.full(len(table), "test", dtype=object)
    split_site[rest] = names[inner]

    result = table[
        ["patient_barcode", "project_id", "tss", "has_rnaseq", "has_slide", "labeled"]
    ].copy()
    result["split"] = split
    result["cv_fold"] = cv_fold
    result["split_site"] = split_site.astype(str)
    joined = result.join(table[list(genes)])
    return SplitResult(splits=joined, warnings=balance_warnings(joined, genes, cfg))


# -------------------------------------------------------------------------------- reports
def class_distribution(
    splits: pd.DataFrame, genes: Sequence[str], split_col: str = "split"
) -> pd.DataFrame:
    """Per subset and gene: labeled, positives, positive fraction and deviation from overall."""
    labeled = splits[splits["labeled"]]
    rows: list[dict[str, Any]] = []
    for gene in genes:
        overall = float((labeled[gene] == 1).mean()) if len(labeled) else float("nan")
        for name in SPLIT_NAMES:
            part = labeled[labeled[split_col] == name]
            positives = int((part[gene] == 1).sum())
            fraction = positives / len(part) if len(part) else float("nan")
            rows.append(
                {
                    "split": name,
                    "gene": gene,
                    "labeled": len(part),
                    "positive": positives,
                    "positive_fraction": fraction,
                    "overall_fraction": overall,
                    "deviation_points": 100.0 * (fraction - overall),
                }
            )
    return pd.DataFrame(rows)


def balance_warnings(splits: pd.DataFrame, genes: Sequence[str], cfg: SplitConfig) -> list[str]:
    """Human-readable problems; the caller decides whether they are fatal (never hidden)."""
    warnings: list[str] = []
    for column in ("split", "split_site"):
        table = class_distribution(splits, genes, column)
        for _, row in table.iterrows():
            if row["positive"] < cfg.min_positives_per_split:
                warnings.append(
                    f"{column}/{row['split']}/{row['gene']}: only {row['positive']} positives "
                    f"(< {cfg.min_positives_per_split}); the subset got too few of them"
                )
            if (
                column == "split"
                and row["labeled"] > 0
                and abs(row["deviation_points"]) > 100 * cfg.max_ratio_deviation
            ):
                warnings.append(
                    f"split/{row['split']}/{row['gene']}: positive fraction deviates "
                    f"{row['deviation_points']:+.1f} points from overall "
                    f"(> {100 * cfg.max_ratio_deviation:.0f})"
                )
    return warnings


def distribution_markdown(
    splits: pd.DataFrame,
    genes: Sequence[str],
    version: str,
    warnings: Sequence[str],
    cfg: SplitConfig,
) -> str:
    """Aggregate-only report (no patient identifiers)."""
    lines = [
        f"# Bölme Dağılım Raporu ({version})",
        "",
        "> `scripts/make_splits.py` tarafından üretilir. Yalnızca toplu sayılar içerir.",
        "",
        f"- Seed: {cfg.seed}; oranlar (train/val/test): {cfg.ratios}; CV katmanı: {cfg.n_folds}",
        f"- Hasta sayısı: {len(splits)}",
        f"- Uyarı eşikleri: alt kümede < {cfg.min_positives_per_split} pozitif; oran genelden "
        f"> {100 * cfg.max_ratio_deviation:.0f} puan sapma",
        "",
        "## Hasta sayısı (proje x küme)",
        "",
        "| proje | train | val | test | toplam |",
        "|---|---|---|---|---|",
    ]
    counts = splits.groupby(["project_id", "split"]).size().unstack(fill_value=0)
    for project, row in counts.iterrows():
        lines.append(
            f"| {project} | "
            + " | ".join(str(int(row.get(n, 0))) for n in SPLIT_NAMES)
            + f" | {int(row.sum())} |"
        )
    for column, title in (
        ("split", "Ana bölme"),
        ("split_site", "Kurum tamamen test (site held-out)"),
    ):
        lines += [
            "",
            f"## Sınıf dağılımı: {title}",
            "",
            "| küme | gen | etiketli | pozitif | pozitif oranı | genel oran | sapma (puan) |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in class_distribution(splits, genes, column).iterrows():
            lines.append(
                f"| {r['split']} | {r['gene']} | {r['labeled']} | {r['positive']} | "
                f"{r['positive_fraction']:.3f} | {r['overall_fraction']:.3f} | "
                f"{r['deviation_points']:+.1f} |"
            )
    folds = splits[splits["cv_fold"] >= 0].groupby("cv_fold").size()
    lines += [
        "",
        "## CV katmanları (train+val havuzu)",
        "",
        "| katman | hasta |",
        "|---|---|",
        *[f"| {i} | {n} |" for i, n in folds.items()],
    ]
    site = splits.groupby("split_site").size()
    lines += [
        "",
        "## Site held-out kümeleri",
        "",
        "| küme | hasta |",
        "|---|---|",
        *[f"| {i} | {n} |" for i, n in site.items()],
    ]
    lines += ["", "## Uyarılar", ""]
    lines += [f"- {w}" for w in warnings] or ["- (yok)"]
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------ leakage checks
def slide_table(patient_map: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Every listed slide with its patient's split (identity contract for the ViT/patch data)."""
    rows = [
        {"slide_file_id": slide, "patient_barcode": row["patient_barcode"]}
        for _, row in patient_map.iterrows()
        for slide in row["slide_file_ids"]
    ]
    slides = pd.DataFrame(rows, columns=["slide_file_id", "patient_barcode"])
    return slides.merge(
        splits[["patient_barcode", "split", "cv_fold", "split_site"]],
        on="patient_barcode",
        how="left",
        validate="many_to_one",
    )


def check_leakage(splits: pd.DataFrame, slides: pd.DataFrame | None = None) -> list[str]:
    """Return every leakage/consistency problem found (empty list = clean)."""
    problems: list[str] = []
    if splits["patient_barcode"].duplicated().any():
        problems.append("a patient appears more than once in the split table")
    for column in ("split", "split_site"):
        bad = set(splits[column]) - set(SPLIT_NAMES)
        if bad:
            problems.append(f"{column}: unknown subset names {sorted(bad)}")
        groups = {n: set(splits.loc[splits[column] == n, "patient_barcode"]) for n in SPLIT_NAMES}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            both = groups[a] & groups[b]
            if both:
                problems.append(f"{column}: {len(both)} patients in both {a} and {b}")
    in_test = splits["split"] == "test"
    if (splits.loc[in_test, "cv_fold"] != -1).any():
        problems.append("cv_fold: test patients must have fold -1")
    if (splits.loc[~in_test, "cv_fold"] < 0).any():
        problems.append("cv_fold: train/val patients must have a fold >= 0")
    site_unit = splits["project_id"] + "|" + splits["tss"]
    test_sites = set(site_unit[splits["split_site"] == "test"])
    shared = test_sites & set(site_unit[splits["split_site"] != "test"])
    if shared:
        problems.append(f"split_site: {len(shared)} sites appear in test AND train/val")
    if slides is not None:
        if slides["slide_file_id"].duplicated().any():
            problems.append("a slide appears more than once")
        if slides["split"].isna().any():
            problems.append("some slides belong to patients without a split")
        mine = slides.merge(
            splits[["patient_barcode", "split"]], on="patient_barcode", suffixes=("", "_patient")
        )
        if (mine["split"] != mine["split_patient"]).any():
            problems.append("a slide's split differs from its patient's split")
    return problems


def assert_no_leakage(splits: pd.DataFrame, slides: pd.DataFrame | None = None) -> None:
    problems = check_leakage(splits, slides)
    if problems:
        raise AssertionError("leakage check failed:\n- " + "\n- ".join(problems))


# ------------------------------------------------------------------- freezing and access
def ids_hash(ids: Iterable[str]) -> str:
    """SHA-256 of the sorted patient ids (order independent)."""
    digest = hashlib.sha256()
    for item in sorted(ids):
        digest.update(item.encode("utf-8") + b"\n")
    return digest.hexdigest()


def build_manifest(
    splits: pd.DataFrame,
    version: str,
    cfg: SplitConfig,
    genes: Sequence[str],
    warnings: Sequence[str],
) -> dict[str, Any]:
    """Frozen description of a split version: counts, config and hashes (no patient ids)."""
    manifest: dict[str, Any] = {
        "version": version,
        "seed": cfg.seed,
        "ratios": list(cfg.ratios),
        "n_folds": cfg.n_folds,
        "genes": list(genes),
        "n_patients": int(len(splits)),
        "counts": {n: int((splits["split"] == n).sum()) for n in SPLIT_NAMES},
        "counts_site": {n: int((splits["split_site"] == n).sum()) for n in SPLIT_NAMES},
        "sha256": {
            n: ids_hash(splits.loc[splits["split"] == n, "patient_barcode"]) for n in SPLIT_NAMES
        },
        "sha256_site_test": ids_hash(splits.loc[splits["split_site"] == "test", "patient_barcode"]),
        "warnings": list(warnings),
        "test_set_rule": "final evaluation only; never for model or hyper-parameter selection",
    }
    return manifest


def verify_manifest(splits: pd.DataFrame, manifest: dict[str, Any]) -> list[str]:
    """Compare a split table with its frozen manifest; any difference means it was altered."""
    problems = []
    for name in SPLIT_NAMES:
        current = ids_hash(splits.loc[splits["split"] == name, "patient_barcode"])
        if current != manifest["sha256"][name]:
            problems.append(f"'{name}' ids differ from the frozen manifest ({manifest['version']})")
    site = ids_hash(splits.loc[splits["split_site"] == "test", "patient_barcode"])
    if site != manifest["sha256_site_test"]:
        problems.append("site held-out test ids differ from the frozen manifest")
    return problems


def get_ids(
    splits: pd.DataFrame,
    which: str,
    *,
    final_evaluation: bool = False,
    column: str = "split",
    modality: str | None = None,
) -> list[str]:
    """Patient ids of one subset; the test subset needs ``final_evaluation=True`` (opt-in)."""
    if which not in SPLIT_NAMES:
        raise ValueError(f"which must be one of {SPLIT_NAMES}")
    if which == "test" and not final_evaluation:
        raise PermissionError(
            "The test split is locked: open it only for the final evaluation "
            "(final_evaluation=True). Never use it for model or hyper-parameter selection."
        )
    part = splits[splits[column] == which]
    if modality is not None:
        if modality not in ("has_rnaseq", "has_slide", "labeled"):
            raise ValueError("modality must be has_rnaseq, has_slide or labeled")
        part = part[part[modality]]
    return sorted(part["patient_barcode"])


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} already exists; create a new version instead of overwriting")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
