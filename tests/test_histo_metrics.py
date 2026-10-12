"""Metrics and bootstrap intervals, cross-checked against scikit-learn (mini sprint 1.9)."""

import pytest

np = pytest.importorskip("numpy")

from models.histo.patch_classifier.metrics import (  # noqa: E402
    accuracy_at,
    all_metrics,
    auroc,
    average_precision,
    bootstrap_ci,
    f1_at,
    summarize_seeds,
)


def test_perfect_and_inverted_rankings() -> None:
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5  # all ties
    assert average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert np.isnan(auroc(np.array([1, 1]), np.array([0.1, 0.9])))


def test_matches_sklearn_on_random_data() -> None:
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(np.int64)
    p = np.clip(rng.normal(0.3 + 0.3 * y, 0.25), 0, 1)
    p[::7] = np.round(p[::7], 1)  # introduce ties
    assert auroc(y, p) == pytest.approx(sklearn_metrics.roc_auc_score(y, p), abs=1e-9)
    assert average_precision(y, p) == pytest.approx(
        sklearn_metrics.average_precision_score(y, p), abs=1e-9
    )
    assert f1_at(y, p) == pytest.approx(sklearn_metrics.f1_score(y, p >= 0.5), abs=1e-9)
    assert accuracy_at(y, p) == pytest.approx(sklearn_metrics.accuracy_score(y, p >= 0.5))


def test_bootstrap_interval_contains_point_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.4).astype(np.int64)
    p = np.clip(rng.normal(0.4 + 0.2 * y, 0.3), 0, 1)
    est = bootstrap_ci(auroc, y, p, n_boot=200, seed=3)
    assert est.lower <= est.point <= est.upper and 0 < est.upper - est.lower < 0.3
    assert bootstrap_ci(auroc, y, p, n_boot=200, seed=3) == est
    assert "[" in est.fmt()
    report = all_metrics(y, p, n_boot=50)
    assert set(report) == {"auroc", "auprc", "f1", "accuracy"}


def test_summarize_seeds() -> None:
    s = summarize_seeds([0.80, 0.82, 0.84])
    assert s["n"] == 3 and s["mean"] == pytest.approx(0.82) and s["ci_lower"] < 0.82 < s["ci_upper"]
    single = summarize_seeds([0.9])
    assert single["std"] == 0.0 and single["ci_lower"] == single["ci_upper"] == 0.9
