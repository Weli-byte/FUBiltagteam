"""Quality control for the omics matrix: sample outliers, batch effects, plots, report.

QC here is descriptive (EDA). PCA/UMAP are fit on the whole cohort ONLY to look at the data; the
resulting coordinates are never used as model features.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
MAD_SCALE = 1.4826  # makes MAD comparable to a standard deviation for normal data


def robust_z(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """(x - median) / (1.4826 * MAD); all zeros if the MAD is 0."""
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * MAD_SCALE
    if mad == 0:
        return np.zeros_like(values)
    return (values - median) / mad


def log2_tpm(tpm: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.log2(np.asarray(tpm, dtype=np.float32) + 1.0)


BLOCK = 4000  # genes per block: keeps temporary arrays small (n_samples x BLOCK)


def _gene_blocks(n_genes: int, block: int) -> list[slice]:
    return [slice(start, min(start + block, n_genes)) for start in range(0, n_genes, block)]


def _log_block(raw: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """log2(TPM+1) of a gene block (non-finite -> 0) and the per-sample 'all finite' mask."""
    finite = np.isfinite(raw)
    return log2_tpm(np.where(finite, raw, 0.0)), finite.all(axis=1)


def sample_qc(
    tpm: np.ndarray[Any, Any],
    library_size: np.ndarray[Any, Any],
    n_detected: np.ndarray[Any, Any],
    mad_threshold: float = 4.0,
    block: int = BLOCK,
) -> pd.DataFrame:
    """Per-sample QC table with outlier flags (nothing is dropped here).

    Flags: missing/non-finite values, library size or detected-gene count far from the cohort
    median (robust z beyond ``mad_threshold``, either direction), and low correlation of the
    sample's log2(TPM+1) profile with the cohort median profile (one-sided: only low is bad).

    Memory: the matrix is processed in gene blocks (a single pass), so no full-size copy of the
    log-transformed matrix is created. Pearson correlation is assembled from running sums.
    """
    n_samples, n_genes = tpm.shape
    complete = np.ones(n_samples, dtype=bool)
    sum_x = np.zeros(n_samples)
    sum_xx = np.zeros(n_samples)
    sum_xp = np.zeros(n_samples)
    sum_p = 0.0
    sum_pp = 0.0
    for part in _gene_blocks(n_genes, block):
        x, ok = _log_block(tpm[:, part])
        complete &= ok
        profile = np.median(x, axis=0).astype(np.float64)
        x = x.astype(np.float64)
        sum_x += x.sum(axis=1)
        sum_xx += (x * x).sum(axis=1)
        sum_xp += x @ profile
        sum_p += profile.sum()
        sum_pp += float((profile * profile).sum())
    cov = sum_xp - sum_x * sum_p / n_genes
    var_x = np.maximum(sum_xx - sum_x**2 / n_genes, 0.0)
    var_p = max(sum_pp - sum_p**2 / n_genes, 0.0)
    denom = np.sqrt(var_x * var_p)
    corr = np.divide(cov, denom, out=np.zeros(n_samples), where=denom > 0)
    has_missing = ~complete

    z_lib = robust_z(np.log10(np.maximum(library_size, 1.0)))
    z_det = robust_z(n_detected)
    z_corr = robust_z(corr)
    table = pd.DataFrame(
        {
            "library_size": library_size,
            "n_detected_genes": n_detected,
            "corr_to_median": corr,
            "z_log_library_size": z_lib,
            "z_detected_genes": z_det,
            "z_corr_to_median": z_corr,
            "flag_missing": has_missing,
            "flag_library_size": np.abs(z_lib) > mad_threshold,
            "flag_detected_genes": np.abs(z_det) > mad_threshold,
            "flag_low_correlation": z_corr < -mad_threshold,
        }
    )
    flags = [c for c in table.columns if c.startswith("flag_")]
    table["flag_any"] = table[flags].any(axis=1)
    return table


def top_variance_log(
    tpm: np.ndarray[Any, Any], n_top: int, block: int = BLOCK
) -> np.ndarray[Any, Any]:
    """log2(TPM+1) of the ``n_top`` highest-variance genes, found block by block."""
    n_genes = tpm.shape[1]
    variance = np.empty(n_genes)
    for part in _gene_blocks(n_genes, block):
        variance[part] = _log_block(tpm[:, part])[0].var(axis=0)
    top = np.sort(np.argsort(-variance, kind="stable")[: min(n_top, n_genes)])
    return _log_block(tpm[:, top])[0]


def pca_embedding(
    logged_top: np.ndarray[Any, Any], n_components: int = 20, seed: int = 0
) -> np.ndarray[Any, Any]:
    """PCA of an already variance-selected log matrix (centred and scaled per gene)."""
    from sklearn.decomposition import PCA

    x = logged_top.astype(np.float64)
    std = x.std(axis=0)
    x = (x - x.mean(axis=0)) / np.where(std > 0, std, 1.0)
    n = min(n_components, x.shape[0] - 1, x.shape[1])
    return np.asarray(PCA(n_components=n, random_state=seed).fit_transform(x), dtype=np.float32)


def sample_correlation(
    tpm: np.ndarray[Any, Any], project: np.ndarray[Any, Any], max_samples: int = 300, seed: int = 0
) -> np.ndarray[Any, Any]:
    """Pearson correlation matrix of up to ``max_samples`` random samples, grouped by project."""
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(tpm), size=min(max_samples, len(tpm)), replace=False))
    order = pick[np.argsort(project[pick], kind="stable")]
    return np.corrcoef(_log_block(tpm[order])[0])


def silhouette(embedding: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float | None:
    """Silhouette score, or ``None`` if there are fewer than 2 usable groups."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    values, counts = np.unique(labels, return_counts=True)
    keep = np.isin(labels, values[counts >= 2])
    if len(np.unique(labels[keep])) < 2:
        return None
    return float(silhouette_score(embedding[keep], labels[keep]))


def batch_effect_summary(
    embedding: np.ndarray[Any, Any], project: np.ndarray[Any, Any], tss: np.ndarray[Any, Any]
) -> dict[str, Any]:
    """How strongly samples cluster by cancer type (wanted) vs tissue source site (batch proxy).

    Silhouette is in [-1, 1]; higher = tighter grouping. TSS is evaluated within each project so
    cancer type does not masquerade as batch. Values are reported as measured, with no threshold.
    """
    project = np.asarray(project)
    tss = np.asarray(tss)
    within: dict[str, float] = {}
    for name in np.unique(project):
        mask = project == name
        score = silhouette(embedding[mask], tss[mask])
        if score is not None:
            within[str(name)] = score
    return {
        "silhouette_project": silhouette(embedding, project),
        "silhouette_tss_within_project": within,
        "silhouette_tss_mean": float(np.mean(list(within.values()))) if within else None,
    }


def umap_embedding(embedding: np.ndarray[Any, Any], seed: int = 0) -> np.ndarray[Any, Any] | None:
    """2-D UMAP of the PCA embedding, or ``None`` if umap-learn is unavailable."""
    try:
        import umap
    except ImportError:
        LOG.warning("umap-learn not installed; UMAP plots skipped")
        return None
    return np.asarray(umap.UMAP(n_components=2, random_state=seed).fit_transform(embedding))


def _scatter(ax: Any, xy: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], title: str) -> None:
    for name in np.unique(labels):
        mask = labels == name
        ax.scatter(xy[mask, 0], xy[mask, 1], s=6, alpha=0.7, label=str(name))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def save_qc_plots(
    out_dir: Path,
    qc: pd.DataFrame,
    corr: np.ndarray[Any, Any],
    project: np.ndarray[Any, Any],
    tss: np.ndarray[Any, Any],
    pca: np.ndarray[Any, Any],
    umap_xy: np.ndarray[Any, Any] | None,
) -> list[Path]:
    """Write library-size, PCA, UMAP and sample-correlation figures; return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in np.unique(project):
        sizes = np.log10(np.maximum(qc.loc[project == name, "library_size"], 1))
        ax.hist(sizes, bins=40, alpha=0.5, label=str(name))
    ax.set_xlabel("log10(library size = sum of counts)")
    ax.set_ylabel("samples")
    ax.legend()
    written.append(_save(fig, out_dir / "library_size_hist.png"))

    plots = [("pca", pca[:, :2], "PCA")]
    if umap_xy is not None:
        plots.append(("umap", umap_xy, "UMAP"))
    for stem, xy, title in plots:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        _scatter(axes[0], xy, project, f"{title}: cancer type")
        axes[0].legend(markerscale=2, fontsize=8)
        _scatter(axes[1], xy, tss, f"{title}: tissue source site (batch proxy)")
        written.append(_save(fig, out_dir / f"{stem}_project_vs_tss.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr, cmap="viridis", vmin=max(-1.0, float(np.nanmin(corr))), vmax=1.0)
    fig.colorbar(image, ax=ax, label="Pearson r (log2 TPM+1)")
    ax.set_title(f"Sample correlation ({len(corr)} random samples, grouped by cancer type)")
    written.append(_save(fig, out_dir / "sample_correlation.png"))
    return written


def _save(fig: Any, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=100)  # keeps each figure well under the 500 KB repo limit
    plt.close(fig)
    return path


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def qc_report_markdown(
    obs: pd.DataFrame,
    qc: pd.DataFrame,
    batch: dict[str, Any],
    mad_threshold: float,
    figures: list[Path],
) -> str:
    """Markdown QC summary (counts only; no per-patient identifiers)."""
    flag_cols = [c for c in qc.columns if c.startswith("flag_")]
    per_project = (
        qc.assign(project_id=obs["project_id"].to_numpy())
        .groupby("project_id")[flag_cols]
        .sum()
        .astype(int)
        .assign(samples=obs.groupby("project_id").size())
    )
    lines = [
        "# Omics QC Raporu",
        "",
        f"- Örnek sayısı: {len(obs)}",
        f"- Uç değer eşiği: |robust z| > {mad_threshold} (medyan ve MAD tabanlı)",
        "- Bayrak (flag) taşıyan örnekler **silinmedi**; karar `--drop-flagged` ile verilir.",
        "",
        "## Bayraklı örnek sayıları (proje bazında)",
        "",
        "| proje | " + " | ".join(["samples", *flag_cols]) + " |",
        "|---|" + "---|" * (len(flag_cols) + 1),
    ]
    for project, row in per_project.iterrows():
        lines.append(
            f"| {project} | {int(row['samples'])} | "
            + " | ".join(str(int(row[c])) for c in flag_cols)
            + " |"
        )
    lines += ["", "## Batch etkisi (silhouette; -1..1, yüksek = daha sıkı gruplaşma)", ""]
    lines.append(f"- Kanser türüne göre: {_fmt(batch['silhouette_project'])}")
    lines.append(
        "- Doku kaynak merkezine (TSS) göre, proje içinde ortalama: "
        f"{_fmt(batch['silhouette_tss_mean'])}"
    )
    for name, value in batch["silhouette_tss_within_project"].items():
        lines.append(f"  - {name}: {value:.3f}")
    lines += [
        "",
        "> Yorum kuralı: kanser türü skoru TSS skorundan belirgin yüksekse biyolojik sinyal "
        "batch'ten baskındır. Sayılar olduğu gibi raporlanır; eşik uydurulmamıştır.",
        "",
        "## Şekiller",
        "",
        *[f"- `{p.name}`" for p in figures],
        "",
    ]
    return "\n".join(lines)
