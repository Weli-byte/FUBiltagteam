"""Classification metrics with bootstrap confidence intervals (mini sprint 1.9).

Bu dosya ne işe yarar? Patch sınıflandırma sonuçlarını özetler: AUROC (eğri altı alan; rastgele
seçilen bir tümör patch'ine normalden daha yüksek skor verme olasılığı), AUPRC (hassasiyet-
duyarlılık eğrisi altı alan; dengesiz sınıflarda daha bilgilendirici), F1 ve doğruluk. Güven
aralıkları bootstrap (yeniden örnekleme) ile hesaplanır. Saf numpy yazıldı; scikit-learn ile
testte karşılaştırılır.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Metric = Callable[[NDArray[np.int64], NDArray[np.float64]], float]


def _check(y: NDArray[np.int64], p: NDArray[np.float64]) -> None:
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y and p must be 1-D arrays of the same length")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y must be binary (0/1)")


def auroc(y: NDArray[np.int64], p: NDArray[np.float64]) -> float:
    """Mann-Whitney formulation (ties count 1/2); NaN if one class is absent."""
    _check(y, p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = p[order]
    i = 0
    while i < len(p):  # average ranks for ties
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = ranks[y == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: NDArray[np.int64], p: NDArray[np.float64]) -> float:
    """Area under the precision-recall curve (step-wise, as scikit-learn's AP)."""
    _check(y, p)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    p_sorted, y_sorted = p[order], y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    # Tied scores form one threshold (as scikit-learn's precision_recall_curve does).
    last_of_group = np.concatenate([np.flatnonzero(np.diff(p_sorted)), [len(p) - 1]])
    precision = tp[last_of_group] / (tp[last_of_group] + fp[last_of_group])
    recall = tp[last_of_group] / y.sum()
    prev_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev_recall) * precision))


def f1_at(y: NDArray[np.int64], p: NDArray[np.float64], threshold: float = 0.5) -> float:
    _check(y, p)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0


def accuracy_at(y: NDArray[np.int64], p: NDArray[np.float64], threshold: float = 0.5) -> float:
    _check(y, p)
    return float(((p >= threshold).astype(np.int64) == y).mean())


@dataclass(frozen=True)
class Estimate:
    point: float
    lower: float
    upper: float
    level: float = 0.95

    def fmt(self, digits: int = 3) -> str:
        return f"{self.point:.{digits}f} [{self.lower:.{digits}f}, {self.upper:.{digits}f}]"


def bootstrap_ci(
    metric: Metric,
    y: NDArray[np.int64],
    p: NDArray[np.float64],
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> Estimate:
    """Percentile bootstrap CI over samples (resampling patches; see slide-level caveat in docs)."""
    rng = np.random.default_rng(seed)
    point = metric(y, p)
    values: list[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        value = metric(y[idx], p[idx])
        if not np.isnan(value):
            values.append(value)
    if not values:
        return Estimate(point, float("nan"), float("nan"), level)
    lo, hi = np.percentile(values, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return Estimate(point, float(lo), float(hi), level)


def all_metrics(
    y: NDArray[np.int64], p: NDArray[np.float64], n_boot: int = 1000, seed: int = 0
) -> dict[str, Estimate]:
    return {
        "auroc": bootstrap_ci(auroc, y, p, n_boot, seed),
        "auprc": bootstrap_ci(average_precision, y, p, n_boot, seed),
        "f1": bootstrap_ci(f1_at, y, p, n_boot, seed),
        "accuracy": bootstrap_ci(accuracy_at, y, p, n_boot, seed),
    }


def summarize_seeds(values: list[float]) -> dict[str, float]:
    """Mean, std and a normal-approximation 95 % interval of the mean across seeds."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {
        "n": float(n),
        "mean": float(arr.mean()),
        "std": std,
        "ci_lower": float(arr.mean() - half),
        "ci_upper": float(arr.mean() + half),
    }
