"""Evaluation metrics for rare binary targets (mini sprint 1.8).

Reusable and dependency-light: every function takes plain arrays ``y`` (0/1) and ``p``
(predicted probability of class 1). Metrics that are undefined for one-class input return ``nan``
instead of raising, so bootstrap loops can skip degenerate resamples and say how many they skipped.

AUPRC (average precision) is the primary metric for rare targets; AUROC alone can look good while
precision is poor. The no-skill AUPRC equals the prevalence, which is reported next to it.
"""

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Explicit TypeAlias: in CI (no numpy installed) ``np`` is ``Any`` and a bare assignment is not
# accepted as a type by mypy --strict.
Array: TypeAlias = np.ndarray[Any, Any]
MetricFn = Callable[[Array, Array], float]


def _check(y: Any, p: Any) -> tuple[Array, Array]:
    y_arr = np.asarray(y).astype(int)
    p_arr = np.asarray(p, dtype=np.float64)
    if y_arr.shape != p_arr.shape or y_arr.ndim != 1:
        raise ValueError("y and p must be 1-D arrays of equal length")
    if not np.isin(y_arr, (0, 1)).all():
        raise ValueError("y must contain only 0 and 1")
    if not np.isfinite(p_arr).all():
        raise ValueError("p contains NaN/inf")
    return y_arr, p_arr


def prevalence(y: Any) -> float:
    """Fraction of positives (= the AUPRC of a no-skill classifier)."""
    return float(np.mean(np.asarray(y)))


def auroc(y: Any, p: Any) -> float:
    y_arr, p_arr = _check(y, p)
    if y_arr.min() == y_arr.max():
        return float("nan")
    return float(roc_auc_score(y_arr, p_arr))


def auprc(y: Any, p: Any) -> float:
    y_arr, p_arr = _check(y, p)
    if y_arr.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_arr, p_arr))


def brier(y: Any, p: Any) -> float:
    """Mean squared error of the probabilities (lower is better; always-prevalence = p(1-p))."""
    y_arr, p_arr = _check(y, p)
    return float(np.mean((p_arr - y_arr) ** 2))


def reliability_curve(y: Any, p: Any, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width probability bins: mean prediction, observed positive fraction, count."""
    y_arr, p_arr = _check(y, p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(p_arr, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = index == b
        if mask.any():
            rows.append(
                {
                    "bin_low": edges[b],
                    "bin_high": edges[b + 1],
                    "mean_pred": float(p_arr[mask].mean()),
                    "frac_pos": float(y_arr[mask].mean()),
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows, columns=["bin_low", "bin_high", "mean_pred", "frac_pos", "n"])


def ece(y: Any, p: Any, n_bins: int = 10) -> float:
    """Expected calibration error: size-weighted mean |observed - predicted| (equal-width bins).

    Pure numpy (no tables) because it runs inside every bootstrap resample.
    """
    y_arr, p_arr = _check(y, p)
    index = np.clip(np.digitize(p_arr, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]), 0, n_bins - 1)
    counts = np.bincount(index, minlength=n_bins).astype(np.float64)
    sum_p = np.bincount(index, weights=p_arr, minlength=n_bins)
    sum_y = np.bincount(index, weights=y_arr.astype(np.float64), minlength=n_bins)
    used = counts > 0
    gap = np.abs(sum_y[used] - sum_p[used])  # |frac_pos - mean_pred| * count
    return float(gap.sum() / counts.sum())


def f1_at(y: Any, p: Any, threshold: float) -> float:
    """F1 of the rule ``p >= threshold``; 0.0 if nothing is predicted or nothing is positive."""
    y_arr, p_arr = _check(y, p)
    pred = p_arr >= threshold
    tp = float(np.sum(pred & (y_arr == 1)))
    denom = 2 * tp + float(np.sum(pred & (y_arr == 0))) + float(np.sum(~pred & (y_arr == 1)))
    return 0.0 if denom == 0 else 2 * tp / denom


def best_f1_threshold(y: Any, p: Any, max_candidates: int = 200) -> float:
    """Threshold maximising F1 on the given data (ties: the larger threshold).

    IMPORTANT: call this on TRAINING-side predictions only (e.g. inner cross-validation), never on
    the data the threshold will be evaluated on.
    """
    y_arr, p_arr = _check(y, p)
    candidates = np.unique(p_arr)
    if len(candidates) > max_candidates:
        candidates = np.unique(np.quantile(p_arr, np.linspace(0.0, 1.0, max_candidates)))
    scores = np.array([f1_at(y_arr, p_arr, t) for t in candidates])
    best = np.flatnonzero(np.isclose(scores, scores.max()))
    return float(candidates[best[-1]])


def bootstrap_ci(
    y: Any,
    p: Any,
    metrics: Mapping[str, MetricFn],
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap confidence intervals by resampling patients with replacement.

    Resamples where a metric is undefined (e.g. no positives drawn) are skipped for that metric;
    ``n_valid`` tells how many resamples were used. The project rule of >= 1000 resamples for
    reported results is enforced where results are produced (``baselines.config_from``).
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    y_arr, p_arr = _check(y, p)
    rng = np.random.default_rng(seed)
    n = len(y_arr)
    draws = {name: np.full(n_boot, np.nan) for name in metrics}
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for name, fn in metrics.items():
            draws[name][i] = fn(y_arr[idx], p_arr[idx])
    alpha = (1.0 - level) / 2.0
    out: dict[str, dict[str, float]] = {}
    for name, fn in metrics.items():
        valid = draws[name][~np.isnan(draws[name])]
        out[name] = {
            "point": fn(y_arr, p_arr),
            "lo": float(np.quantile(valid, alpha)) if len(valid) else float("nan"),
            "hi": float(np.quantile(valid, 1 - alpha)) if len(valid) else float("nan"),
            "n_valid": float(len(valid)),
        }
    return out


def paired_bootstrap_delta(
    y: Any,
    p_model: Any,
    p_reference: Any,
    metric: MetricFn,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """CI of ``metric(model) - metric(reference)`` on the SAME resampled patients.

    A CI entirely above 0 means the model beats the reference beyond resampling noise; comparing two
    separate CIs for overlap is weaker and not used for decisions.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    y_arr, a = _check(y, p_model)
    _, b = _check(y, p_reference)
    rng = np.random.default_rng(seed)
    n = len(y_arr)
    deltas = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = metric(y_arr[idx], a[idx]) - metric(y_arr[idx], b[idx])
    valid = deltas[~np.isnan(deltas)]
    alpha = (1.0 - level) / 2.0
    return {
        "delta": metric(y_arr, a) - metric(y_arr, b),
        "lo": float(np.quantile(valid, alpha)) if len(valid) else float("nan"),
        "hi": float(np.quantile(valid, 1 - alpha)) if len(valid) else float("nan"),
        "n_valid": float(len(valid)),
    }


def summarize(
    y: Any, p: Any, threshold: float, n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    """All reported metrics with 95% bootstrap CIs (``<metric>``, ``<metric>_lo``, ``_hi``)."""
    metrics: dict[str, MetricFn] = {
        "auroc": auroc,
        "auprc": auprc,
        "f1": lambda yy, pp: f1_at(yy, pp, threshold),
        "brier": brier,
        "ece": ece,
    }
    boot = bootstrap_ci(y, p, metrics, n_boot=n_boot, seed=seed)
    flat: dict[str, float] = {"prevalence": prevalence(y), "n": float(len(np.asarray(y)))}
    for name, values in boot.items():
        flat[name] = values["point"]
        flat[f"{name}_lo"] = values["lo"]
        flat[f"{name}_hi"] = values["hi"]
    flat["n_boot_valid_min"] = min(v["n_valid"] for v in boot.values())
    return flat
