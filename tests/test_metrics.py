"""Metric tests against hand-computed values and known properties."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import numpy as np  # noqa: E402

from models.omics import metrics as M  # noqa: E402


def test_known_values_from_hand_computation() -> None:
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    assert M.auroc(y, p) == pytest.approx(0.75)  # 3 of 4 positive-negative pairs ranked correctly
    assert M.auprc(y, p) == pytest.approx(0.8333333, abs=1e-6)  # documented AP example
    assert M.brier(y, p) == pytest.approx(((0.1) ** 2 + 0.4**2 + 0.65**2 + 0.2**2) / 4)
    assert M.prevalence(y) == 0.5


def test_ece_and_reliability_curve() -> None:
    assert M.ece([0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9]) == pytest.approx(0.1)
    assert M.ece([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)
    curve = M.reliability_curve([0, 1, 1, 0, 1], [0.05, 0.95, 0.9, 0.5, 0.55])
    assert int(curve["n"].sum()) == 5 and curve["frac_pos"].between(0, 1).all()
    assert M.ece([0, 1], [0.0, 1.0]) == pytest.approx(0.0)  # probabilities at the bin edges


def test_fast_ece_matches_the_table_based_definition() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    p = np.clip(0.3 + 0.3 * (y - 0.3) + rng.normal(0, 0.25, 500), 0, 1)
    curve = M.reliability_curve(y, p)
    slow = float(
        np.sum(curve["n"] * (curve["frac_pos"] - curve["mean_pred"]).abs()) / curve["n"].sum()
    )
    assert M.ece(y, p) == pytest.approx(slow)


def test_undefined_cases_return_nan_not_errors() -> None:
    assert np.isnan(M.auroc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert np.isnan(M.auprc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert M.f1_at([0, 0], [0.1, 0.2], 0.5) == 0.0


def test_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        M.auroc([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        M.auroc([0, 1], [0.1, np.nan])
    with pytest.raises(ValueError):
        M.auroc([0, 1, 1], [0.1, 0.2])


def test_f1_and_threshold_selection() -> None:
    y, p = [0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.7, 0.9]
    assert M.f1_at(y, p, 0.5) == pytest.approx(1.0)
    assert M.f1_at(y, p, 0.0) == pytest.approx(2 * 2 / (2 * 2 + 3 + 0))
    threshold = M.best_f1_threshold(y, p)
    assert M.f1_at(y, p, threshold) == pytest.approx(1.0)
    assert 0.3 < threshold <= 0.7


def test_bootstrap_ci_properties() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    y = (rng.random(n) < 0.3).astype(int)
    p = np.clip(0.3 + 0.35 * (y - 0.3) + rng.normal(0, 0.25, n), 0, 1)
    fns = {"auroc": M.auroc, "auprc": M.auprc}
    a = M.bootstrap_ci(y, p, fns, n_boot=200, seed=1)
    b = M.bootstrap_ci(y, p, fns, n_boot=200, seed=1)
    c = M.bootstrap_ci(y, p, fns, n_boot=200, seed=2)
    assert a == b and a != c  # deterministic per seed
    for name in fns:
        assert a[name]["lo"] <= a[name]["point"] <= a[name]["hi"]
        assert a[name]["n_valid"] == 200
    # larger samples give narrower intervals
    small = M.bootstrap_ci(y[:200], p[:200], {"auroc": M.auroc}, n_boot=200, seed=1)["auroc"]
    assert (small["hi"] - small["lo"]) > (a["auroc"]["hi"] - a["auroc"]["lo"])


def test_bootstrap_skips_degenerate_resamples_and_reports_how_many() -> None:
    y = np.array([1] + [0] * 29)  # one positive: many resamples contain none
    p = np.linspace(0, 1, 30)
    out = M.bootstrap_ci(y, p, {"auroc": M.auroc}, n_boot=300, seed=0)["auroc"]
    assert 0 < out["n_valid"] < 300


def test_paired_delta_detects_a_better_model_and_zero_for_identical() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(800) < 0.3).astype(int)
    good = np.clip(0.3 + 0.4 * (y - 0.3) + rng.normal(0, 0.2, 800), 0, 1)
    random_ref = rng.random(800)
    better = M.paired_bootstrap_delta(y, good, random_ref, M.auroc, n_boot=300, seed=0)
    assert better["lo"] > 0.2 and better["delta"] > 0.3
    same = M.paired_bootstrap_delta(y, good, good, M.auroc, n_boot=100, seed=0)
    assert same["delta"] == 0 and same["lo"] == 0 and same["hi"] == 0


def test_summarize_reports_every_metric_with_interval() -> None:
    rng = np.random.default_rng(4)
    y = (rng.random(300) < 0.25).astype(int)
    p = np.clip(0.25 + 0.3 * (y - 0.25) + rng.normal(0, 0.2, 300), 0, 1)
    out = M.summarize(y, p, threshold=0.4, n_boot=100)
    for name in ("auroc", "auprc", "f1", "brier", "ece"):
        assert out[name + "_lo"] <= out[name] + 1e-9 <= out[name + "_hi"] + 2e-9
    assert out["prevalence"] == pytest.approx(y.mean()) and out["n"] == 300
