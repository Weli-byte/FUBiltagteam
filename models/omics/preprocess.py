"""Fitted omics preprocessing: gene filter -> log2(TPM+1) -> variance selection -> z-score.

``ExpressionPreprocessor`` follows the scikit-learn ``fit``/``transform`` contract. Every learned
quantity (which genes pass the filter, which are the top-variance genes, per-gene mean/std) is
computed from the matrix given to ``fit`` ONLY. Fit it on the training split and call ``transform``
on validation/test samples; never fit on the full cohort for model training (data leakage).
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

LOG = logging.getLogger(__name__)


BLOCK = 4000  # genes per block; temporary arrays stay at n_samples x BLOCK
MIN_STD = 1e-8  # log2 units; below this a gene is constant (float rounding would give ~1e-16)


def _check_matrix(x: Any, name: str = "X", block: int = BLOCK) -> np.ndarray[Any, Any]:
    """Validate shape/numeric/NaN/negatives block by block (no full-size boolean copies)."""
    array = np.asarray(x)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D (samples, genes); got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    for start in range(0, array.shape[1], block):
        part = array[:, start : start + block]
        if np.isnan(part).any():
            raise ValueError(f"{name} contains NaN; remove or impute these samples first")
        if (part < 0).any():
            raise ValueError(f"{name} contains negative TPM values")
    return array


class ExpressionPreprocessor(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Learn a gene selection and scaling from training TPM and apply it to any TPM matrix.

    Args:
        min_tpm: A gene counts as expressed in a sample if TPM > ``min_tpm``.
        min_fraction: Keep genes expressed in at least this fraction of the fit samples.
        top_k: Keep only the ``top_k`` highest-variance genes (after filtering); ``None`` = all.
        allowed_genes: Optional boolean mask over input genes (e.g. protein-coding only).
        zscore: Standardise each kept gene with the fit mean/std.
        clip: If set, clip z-scores to ``[-clip, clip]`` (limits extreme-value influence).
    """

    def __init__(
        self,
        min_tpm: float = 1.0,
        min_fraction: float = 0.2,
        top_k: int | None = None,
        allowed_genes: np.ndarray[Any, Any] | None = None,
        zscore: bool = True,
        clip: float | None = None,
    ) -> None:
        self.min_tpm = min_tpm
        self.min_fraction = min_fraction
        self.top_k = top_k
        self.allowed_genes = allowed_genes
        self.zscore = zscore
        self.clip = clip

    def fit(
        self,
        X: Any,  # noqa: N803
        y: Any = None,
        sample_idx: np.ndarray[Any, Any] | None = None,
    ) -> "ExpressionPreprocessor":
        """Learn the gene selection and scaling.

        ``sample_idx`` restricts learning to those rows of ``X`` (e.g. the training split) without
        copying them; all statistics are computed in gene blocks to keep memory small.
        """
        tpm = _check_matrix(X)
        n_genes = tpm.shape[1]
        rows = np.arange(tpm.shape[0]) if sample_idx is None else np.asarray(sample_idx)
        n_samples = len(rows)
        if n_samples < 2:
            raise ValueError("need at least 2 samples to fit")
        if not 0.0 <= self.min_fraction <= 1.0:
            raise ValueError("min_fraction must be in [0, 1]")
        allowed = np.ones(n_genes, dtype=bool)
        if self.allowed_genes is not None:
            allowed = np.asarray(self.allowed_genes, dtype=bool)
            if allowed.shape != (n_genes,):
                raise ValueError("allowed_genes must have one entry per input gene")

        prevalence = np.empty(n_genes)
        for start in range(0, n_genes, BLOCK):
            part = tpm[rows, start : start + BLOCK]
            prevalence[start : start + BLOCK] = (part > self.min_tpm).mean(axis=0)
        candidates = np.flatnonzero(allowed & (prevalence >= self.min_fraction))

        mean = np.empty(candidates.size)
        std = np.empty(candidates.size)
        for start in range(0, candidates.size, BLOCK):
            cols = candidates[start : start + BLOCK]
            logged = np.log2(tpm[np.ix_(rows, cols)].astype(np.float64) + 1.0)
            mean[start : start + BLOCK] = logged.mean(axis=0)
            std[start : start + BLOCK] = logged.std(axis=0)
        varying = std > MIN_STD
        candidates, mean, std = candidates[varying], mean[varying], std[varying]
        if candidates.size == 0:
            raise ValueError("no gene passed the expression filter with non-zero variance")
        if self.top_k is not None and self.top_k < candidates.size:
            top = np.sort(np.argsort(-(std**2), kind="stable")[: self.top_k])
            candidates, mean, std = candidates[top], mean[top], std[top]
        elif self.top_k is not None and self.top_k > candidates.size:
            LOG.warning("top_k=%d > %d genes available; keeping all", self.top_k, candidates.size)

        self.n_features_in_ = n_genes
        self.selected_idx_ = candidates
        self.mean_ = mean
        self.scale_ = std
        self.n_fit_samples_ = n_samples
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "selected_idx_"):
            raise NotFittedError("ExpressionPreprocessor is not fitted; call fit() first")

    def transform(self, X: Any) -> np.ndarray[Any, Any]:  # noqa: N803
        self._check_fitted()
        tpm = _check_matrix(X)
        if tpm.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} genes, got {tpm.shape[1]}")
        out = np.log2(tpm[:, self.selected_idx_].astype(np.float64) + 1.0)
        if self.zscore:
            out = (out - self.mean_) / self.scale_
            if self.clip is not None:
                out = np.clip(out, -self.clip, self.clip)
        return out.astype(np.float32)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray[Any, Any]:
        self._check_fitted()
        if input_features is None:
            return np.array([f"x{i}" for i in self.selected_idx_])
        return np.asarray(input_features)[self.selected_idx_]


# ------------------------------------------------------------------------------ gene sets
def parse_gmt(path: Path | str) -> dict[str, list[str]]:
    """Read a GMT gene-set file (``name<TAB>description<TAB>gene1<TAB>gene2...``)."""
    sets: dict[str, list[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3 or not parts[0]:
            continue
        if parts[0] in sets:
            raise ValueError(f"duplicate gene set name {parts[0]!r} in {path}")
        sets[parts[0]] = [g for g in parts[2:] if g]
    return sets


def aggregate_gene_sets(
    z: np.ndarray[Any, Any],
    symbols: Sequence[str],
    gene_sets: dict[str, list[str]],
    min_overlap: int = 5,
) -> tuple[np.ndarray[Any, Any], list[str], dict[str, int]]:
    """Mean of (already standardised) gene values per gene set -> ``(samples, sets)`` matrix.

    Sets with fewer than ``min_overlap`` genes present are skipped and reported, not hidden.
    Symbols are matched case-insensitively; duplicate symbols in ``symbols`` all contribute.
    """
    index: dict[str, list[int]] = {}
    for i, symbol in enumerate(symbols):
        if symbol:
            index.setdefault(symbol.upper(), []).append(i)
    columns: list[np.ndarray[Any, Any]] = []
    names: list[str] = []
    overlaps: dict[str, int] = {}
    for name, members in gene_sets.items():
        positions = sorted({i for g in members for i in index.get(g.upper(), [])})
        overlaps[name] = len(positions)
        if len(positions) < min_overlap:
            LOG.warning(
                "gene set %s skipped: %d genes overlap (< %d)", name, len(positions), min_overlap
            )
            continue
        columns.append(z[:, positions].mean(axis=1))
        names.append(name)
    if not columns:
        raise ValueError("no gene set had enough overlapping genes")
    return np.stack(columns, axis=1).astype(np.float32), names, overlaps
