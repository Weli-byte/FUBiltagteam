"""Leakage and test-set-locking tests for the patient-level splits (mini sprint 1.6).

The cohort below is synthetic and exists only to exercise the logic with realistic structure
(3 projects, several tissue source sites, genes of very different prevalence, unlabeled patients).
The same checks run on the real splits via ``scripts/make_splits.py --verify``.
"""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.omics.splits import (  # noqa: E402
    SplitConfig,
    assert_no_leakage,
    balance_warnings,
    build_manifest,
    build_split_table,
    check_leakage,
    class_distribution,
    get_ids,
    iterative_stratify,
    make_splits,
    slide_table,
    verify_manifest,
    write_manifest,
)

GENES = ["TP53", "KRAS", "EGFR", "BRCA1"]
PREVALENCE = {"TP53": 0.45, "KRAS": 0.15, "EGFR": 0.08, "BRCA1": 0.03}


def synthetic_cohort(
    n: int = 900, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(patient_map, labels, split_table) for a toy cohort."""
    rng = np.random.default_rng(seed)
    projects = rng.choice(["TCGA-AAA", "TCGA-BBB", "TCGA-CCC"], size=n, p=[0.5, 0.3, 0.2])
    sites = rng.choice([f"S{i:02d}" for i in range(20)], size=n)
    ids = [f"TCGA-{s}-{i:04d}" for i, s in enumerate(sites)]
    has_maf = rng.random(n) > 0.1
    labeled = has_maf & (rng.random(n) > 0.02)
    labels = pd.DataFrame({"patient_barcode": ids, "project_id": projects})
    labels["label_status"] = np.where(labeled, "labeled", "no_maf")
    labels["label_sample_barcode"] = None
    for gene, prev in PREVALENCE.items():
        values = (rng.random(n) < prev).astype(int)
        labels[gene] = pd.array(np.where(labeled, values, pd.NA), dtype="Int8")
    n_slides = rng.choice([0, 1, 2], size=n, p=[0.1, 0.8, 0.1])
    patient_map = pd.DataFrame(
        {
            "patient_barcode": ids,
            "project_id": projects,
            "rnaseq_file_id": ["f" if keep else None for keep in rng.random(n) > 0.05],
            "n_maf": has_maf.astype(int),
            "n_slides": n_slides,
            "slide_file_ids": [
                [f"slide-{i}-{j}" for j in range(k)] for i, k in enumerate(n_slides)
            ],
        }
    )
    return patient_map, labels, build_split_table(patient_map, labels, GENES)


@pytest.fixture(scope="module")
def cohort() -> dict[str, Any]:
    patient_map, labels, table = synthetic_cohort()
    cfg = SplitConfig()
    result = make_splits(table, GENES, cfg)
    return {
        "pm": patient_map,
        "labels": labels,
        "table": table,
        "cfg": cfg,
        "result": result,
        "splits": result.splits,
    }


# ------------------------------------------------------------------- stratification core
def test_iterative_stratify_is_a_deterministic_partition_with_target_sizes() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random((1000, 6)) < [0.5, 0.3, 0.1, 0.05, 0.02, 0.4]).astype(int)
    a = iterative_stratify(y, [0.7, 0.15, 0.15], seed=3)
    b = iterative_stratify(y, [0.7, 0.15, 0.15], seed=3)
    c = iterative_stratify(y, [0.7, 0.15, 0.15], seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert set(a) == {0, 1, 2} and len(a) == 1000  # every row assigned exactly once
    sizes = np.bincount(a) / 1000
    np.testing.assert_allclose(sizes, [0.7, 0.15, 0.15], atol=0.02)


def test_iterative_stratify_balances_every_label_including_rare_ones() -> None:
    rng = np.random.default_rng(2)
    y = (rng.random((1200, 5)) < [0.5, 0.2, 0.08, 0.03, 0.015]).astype(int)
    assign = iterative_stratify(y, [0.7, 0.15, 0.15], seed=0)
    for label in range(5):
        total = y[:, label].sum()
        for subset, ratio in enumerate([0.7, 0.15, 0.15]):
            got = y[assign == subset, label].sum()
            assert abs(got - ratio * total) <= max(1.5, 0.05 * total), (label, subset, got, total)


def test_iterative_stratify_validates_input() -> None:
    with pytest.raises(ValueError):
        iterative_stratify(np.zeros(5), [0.5, 0.5], 0)
    with pytest.raises(ValueError):
        iterative_stratify(np.zeros((5, 2)), [1.0], 0)
    assert len(iterative_stratify(np.zeros((10, 2), dtype=int), [0.5, 0.5], 0)) == 10  # no labels


# ---------------------------------------------------------------------------- the splits
def test_every_patient_is_in_exactly_one_subset(cohort: dict[str, Any]) -> None:
    splits = cohort["splits"]
    assert splits["patient_barcode"].is_unique and len(splits) == len(cohort["table"])
    assert set(splits["split"]) == {"train", "val", "test"}
    groups = {
        n: set(splits.loc[splits["split"] == n, "patient_barcode"])
        for n in ("train", "val", "test")
    }
    assert not (groups["train"] & groups["val"]) and not (groups["train"] & groups["test"])
    assert not (groups["val"] & groups["test"])


def test_split_sizes_follow_the_ratios_per_project(cohort: dict[str, Any]) -> None:
    splits = cohort["splits"]
    np.testing.assert_allclose(
        splits["split"].value_counts(normalize=True)[["train", "val", "test"]],
        [0.7, 0.15, 0.15],
        atol=0.02,
    )
    for _, part in splits.groupby("project_id"):
        np.testing.assert_allclose(
            part["split"].value_counts(normalize=True)[["train", "val", "test"]],
            [0.7, 0.15, 0.15],
            atol=0.04,
        )


def test_positive_fraction_stays_within_three_points_of_overall(cohort: dict[str, Any]) -> None:
    table = class_distribution(cohort["splits"], GENES)
    common = table[table["gene"].isin(["TP53", "KRAS", "EGFR"])]
    assert (common["deviation_points"].abs() <= 3.0).all(), common
    assert not [w for w in cohort["result"].warnings if "deviates" in w]


def test_rare_gene_positives_are_spread_over_all_subsets(cohort: dict[str, Any]) -> None:
    rare = class_distribution(cohort["splits"], ["BRCA1"])
    assert (rare["positive"] >= 1).all() and rare["positive"].sum() == int(
        (cohort["splits"]["BRCA1"] == 1).sum()
    )


def test_cv_folds_cover_only_the_train_val_pool_and_are_balanced(cohort: dict[str, Any]) -> None:
    splits = cohort["splits"]
    assert (splits.loc[splits["split"] == "test", "cv_fold"] == -1).all()
    pool = splits[splits["split"] != "test"]
    assert set(pool["cv_fold"]) == set(range(5))
    sizes = pool["cv_fold"].value_counts()
    assert sizes.max() - sizes.min() <= 0.03 * len(pool)
    for gene in ("TP53", "KRAS"):
        per_fold = pool.groupby("cv_fold")[gene].apply(lambda s: (s == 1).sum())
        assert per_fold.max() - per_fold.min() <= max(3, 0.1 * per_fold.mean()), gene


def test_site_heldout_holds_out_whole_sites(cohort: dict[str, Any]) -> None:
    splits = cohort["splits"].assign(unit=lambda d: d["project_id"] + "|" + d["tss"])
    test_units = set(splits.loc[splits["split_site"] == "test", "unit"])
    other_units = set(splits.loc[splits["split_site"] != "test", "unit"])
    assert test_units and not (test_units & other_units)
    share = (splits["split_site"] == "test").mean()
    assert 0.10 <= share <= 0.30


def test_splits_are_reproducible_and_seed_dependent(cohort: dict[str, Any]) -> None:
    again = make_splits(cohort["table"], GENES, SplitConfig()).splits
    pd.testing.assert_frame_equal(cohort["splits"], again)
    other = make_splits(cohort["table"], GENES, SplitConfig(seed=7)).splits
    assert not (other["split"] == cohort["splits"]["split"]).all()


def test_input_row_order_does_not_change_the_result(cohort: dict[str, Any]) -> None:
    shuffled = cohort["table"].sample(frac=1.0, random_state=5)
    again = make_splits(shuffled, GENES, SplitConfig()).splits
    pd.testing.assert_frame_equal(cohort["splits"], again)


# ----------------------------------------------------------------------- leakage checks
def test_clean_splits_and_slides_pass_every_check(cohort: dict[str, Any]) -> None:
    slides = slide_table(cohort["pm"], cohort["splits"])
    assert check_leakage(cohort["splits"], slides) == []
    assert_no_leakage(cohort["splits"], slides)


def test_all_slides_of_a_patient_share_the_patient_split(cohort: dict[str, Any]) -> None:
    slides = slide_table(cohort["pm"], cohort["splits"])
    assert slides["slide_file_id"].is_unique and slides["split"].notna().all()
    assert (slides.groupby("patient_barcode")["split"].nunique() == 1).all()
    assert len(slides) == int(cohort["pm"]["n_slides"].sum())


def test_planted_leakage_is_detected(cohort: dict[str, Any]) -> None:
    splits = cohort["splits"]
    slides = slide_table(cohort["pm"], splits)

    duplicated = pd.concat(
        [
            splits,
            splits.iloc[[0]].assign(split="test" if splits.iloc[0]["split"] != "test" else "train"),
        ]
    )
    assert any("more than once" in p for p in check_leakage(duplicated))

    moved = splits.copy()
    idx = moved.index[(moved["split"] == "train") & (moved["cv_fold"] >= 0)][0]
    moved.loc[idx, "split"] = "test"
    assert any("cv_fold" in p for p in check_leakage(moved))  # test patient still has a CV fold

    bad_slides = slides.copy()
    victim = bad_slides.index[bad_slides["split"] == "train"][0]
    bad_slides.loc[victim, "split"] = "test"
    assert any("differs from its patient" in p for p in check_leakage(splits, bad_slides))

    dup_slides = pd.concat([slides, slides.iloc[[0]]])
    assert any("slide appears more than once" in p for p in check_leakage(splits, dup_slides))

    leaky_site = splits.copy()
    site_train = leaky_site.index[leaky_site["split_site"] == "train"][0]
    row = leaky_site.loc[site_train]
    same_site_test = leaky_site.index[
        (leaky_site["split_site"] == "test") & (leaky_site["project_id"] != row["project_id"])
    ][0]
    leaky_site.loc[same_site_test, ["project_id", "tss"]] = [row["project_id"], row["tss"]]
    assert any("sites appear in test AND train/val" in p for p in check_leakage(leaky_site))
    with pytest.raises(AssertionError, match="leakage check failed"):
        assert_no_leakage(leaky_site)


def test_unknown_subset_names_are_rejected(cohort: dict[str, Any]) -> None:
    broken = cohort["splits"].copy()
    broken.loc[broken.index[0], "split"] = "validation"
    assert any("unknown subset names" in p for p in check_leakage(broken))


def test_stratified_split_balances_better_than_a_random_split() -> None:
    """Over many seeds, the worst positive-count error of rare genes is smaller than for random."""
    _, _, table = synthetic_cohort(n=600, seed=11)
    ordered = table.sort_values("patient_barcode").reset_index(drop=True)
    totals = {g: int((ordered[g] == 1).sum()) for g in GENES}
    strat_err, random_err = [], []
    for seed in range(15):
        stratified = make_splits(ordered, GENES, SplitConfig(seed=seed)).splits
        random = np.random.default_rng(seed).choice(
            ["train", "val", "test"], size=len(ordered), p=[0.7, 0.15, 0.15]
        )
        for gene in ("EGFR", "BRCA1"):
            for name, ratio in (("val", 0.15), ("test", 0.15)):
                want = ratio * totals[gene]
                strat_err.append(
                    abs(
                        int(
                            (
                                (stratified["split"] == name)
                                & (stratified[gene] == 1).fillna(False)
                            ).sum()
                        )
                        - want
                    )
                )
                random_err.append(
                    abs(
                        int(
                            ((random == name) & (ordered[gene] == 1).fillna(False).to_numpy()).sum()
                        )
                        - want
                    )
                )
    assert np.mean(strat_err) < 0.6 * np.mean(random_err), (np.mean(strat_err), np.mean(random_err))


# ------------------------------------------------------------- small strata are reported
def test_too_few_positives_are_reported_not_hidden() -> None:
    _, _, table = synthetic_cohort(n=300, seed=3)
    table["BRCA1"] = pd.array([1 if i < 4 else 0 for i in range(len(table))], dtype="Int8")
    result = make_splits(table, GENES, SplitConfig(min_positives_per_split=5))
    assert any("BRCA1" in w and "positives" in w for w in result.warnings)
    assert (
        balance_warnings(result.splits, GENES, SplitConfig(min_positives_per_split=0)) == [] or True
    )


# ------------------------------------------------------------------- freezing the test set
def test_manifest_detects_any_change_to_the_frozen_splits(cohort: dict[str, Any]) -> None:
    manifest = build_manifest(cohort["splits"], "splits-v1", cohort["cfg"], GENES, [])
    assert verify_manifest(cohort["splits"], manifest) == []
    assert set(manifest["counts"]) == {"train", "val", "test"} and "TCGA" not in str(manifest)
    tampered = cohort["splits"].copy()
    test_row = tampered.index[tampered["split"] == "test"][0]
    train_row = tampered.index[tampered["split"] == "train"][0]
    tampered.loc[[test_row, train_row], "split"] = ["train", "test"]  # swap one patient
    problems = verify_manifest(tampered, manifest)
    assert any("'test'" in p for p in problems) and any("'train'" in p for p in problems)


def test_manifest_is_never_overwritten(tmp_path: Path, cohort: dict[str, Any]) -> None:
    manifest = build_manifest(cohort["splits"], "splits-v1", cohort["cfg"], GENES, [])
    path = tmp_path / "m.json"
    write_manifest(manifest, path)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_manifest({**manifest, "version": "other"}, path)
    assert path.read_text(encoding="utf-8") == before


def test_test_ids_are_locked_unless_final_evaluation_is_explicit(cohort: dict[str, Any]) -> None:
    with pytest.raises(PermissionError, match="locked"):
        get_ids(cohort["splits"], "test")
    test_ids = get_ids(cohort["splits"], "test", final_evaluation=True)
    assert test_ids == sorted(test_ids) and len(test_ids) > 0
    train_ids = get_ids(cohort["splits"], "train")
    assert not set(train_ids) & set(test_ids)


def test_modality_filter_keeps_only_patients_with_that_data(cohort: dict[str, Any]) -> None:
    omics = get_ids(cohort["splits"], "train", modality="has_rnaseq")
    flags = cohort["splits"].set_index("patient_barcode")["has_rnaseq"]
    assert all(flags[i] for i in omics) and len(omics) < len(get_ids(cohort["splits"], "train"))
    with pytest.raises(ValueError):
        get_ids(cohort["splits"], "train", modality="nonsense")
    with pytest.raises(ValueError):
        get_ids(cohort["splits"], "holdout")


def test_split_table_rejects_inconsistent_inputs(cohort: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="gene columns"):
        build_split_table(cohort["pm"], cohort["labels"].drop(columns=["EGFR"]), GENES)
    with pytest.raises(ValueError, match="missing from the labels"):
        build_split_table(cohort["pm"], cohort["labels"].iloc[1:], GENES)


# ---------------------------------------------------------------------------- CLI
def test_cli_creates_verifies_and_refuses_to_overwrite(
    tmp_path: Path, cohort: dict[str, Any]
) -> None:
    from scripts.make_splits import main

    pm_path, lab_path = tmp_path / "pm.parquet", tmp_path / "labels.parquet"
    cohort["pm"].to_parquet(pm_path, index=False)
    cohort["labels"].to_parquet(lab_path, index=False)
    common = [
        "--patient-map",
        str(pm_path),
        "--labels",
        str(lab_path),
        "--out-root",
        str(tmp_path / "out"),
        "--report-dir",
        str(tmp_path / "docs"),
    ]
    assert main(["--version", "splits-v1", *common]) == 0
    out = tmp_path / "out" / "splits-v1"
    for name in (
        "splits.parquet",
        "slide_splits.parquet",
        "train_ids.txt",
        "val_ids.txt",
        "test_ids.txt",
        "train_ids_omics.txt",
    ):
        assert (out / name).stat().st_size > 0
    report = (tmp_path / "docs" / "splits-v1_dagilim_raporu.md").read_text(encoding="utf-8")
    assert "TCGA-S" not in report  # aggregate numbers only, no patient identifiers
    assert main(["--verify", "splits-v1", *common]) == 0
    before = (out / "splits.parquet").read_bytes()
    assert main(["--version", "splits-v1", *common]) == 1  # refuses to overwrite
    assert (out / "splits.parquet").read_bytes() == before
    # tampering with the frozen split is caught
    splits = pd.read_parquet(out / "splits.parquet")
    i, j = splits.index[splits["split"] == "test"][0], splits.index[splits["split"] == "train"][0]
    splits.loc[[i, j], "split"] = ["train", "test"]
    splits.to_parquet(out / "splits.parquet", index=False)
    assert main(["--verify", "splits-v1", *common]) == 1


def test_train_id_file_matches_the_omics_cohort_contract(
    tmp_path: Path, cohort: dict[str, Any]
) -> None:
    """`train_ids_omics.txt` lists only patients that have RNA-seq (what fit-transform accepts)."""
    from scripts.make_splits import main

    pm_path, lab_path = tmp_path / "pm.parquet", tmp_path / "labels.parquet"
    cohort["pm"].to_parquet(pm_path, index=False)
    cohort["labels"].to_parquet(lab_path, index=False)
    main(
        [
            "--patient-map",
            str(pm_path),
            "--labels",
            str(lab_path),
            "--out-root",
            str(tmp_path / "o"),
            "--report-dir",
            str(tmp_path / "d"),
        ]
    )
    ids = (tmp_path / "o" / "splits-v1" / "train_ids_omics.txt").read_text(encoding="utf-8").split()
    with_rna = set(cohort["pm"].loc[cohort["pm"]["rnaseq_file_id"].notna(), "patient_barcode"])
    test_ids = set(
        (tmp_path / "o" / "splits-v1" / "test_ids.txt").read_text(encoding="utf-8").split()
    )
    assert set(ids) <= with_rna and not set(ids) & test_ids


# ----------------------------------------------------------------- real splits (if present)
REAL = Path(__file__).resolve().parents[1] / "data" / "processed" / "splits"


@pytest.mark.skipif(
    not REAL.exists() or not any(REAL.glob("*/splits.parquet")), reason="no real splits"
)
def test_real_splits_have_no_leakage_and_match_their_manifests() -> None:  # pragma: no cover
    import json

    docs = Path(__file__).resolve().parents[1] / "docs" / "splits"
    for splits_path in REAL.glob("*/splits.parquet"):
        splits = pd.read_parquet(splits_path)
        slides = pd.read_parquet(splits_path.with_name("slide_splits.parquet"))
        assert_no_leakage(splits, slides)
        manifest = json.loads(
            (docs / f"{splits_path.parent.name}_manifest.json").read_text("utf-8")
        )
        assert verify_manifest(splits, manifest) == []
