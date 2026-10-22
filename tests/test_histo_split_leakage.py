"""Patch/slide-level leakage checks against the patient-level split table (mini sprint 1.10)."""

from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from models.histo.patches.splits import (  # noqa: E402
    assert_no_patch_leakage,
    check_patch_split_leakage,
    slide_split_summary,
)


def _slide_splits() -> Any:
    return pd.DataFrame(
        {
            "slide_file_id": ["s1", "s2", "s3", "s4", "s5"],
            "patient_barcode": ["P1", "P1", "P2", "P3", "P4"],
            "split": ["train", "train", "val", "test", "train"],
        }
    )


def _patches(slides: list[str], splits: list[str] | None = None) -> Any:
    frame = pd.DataFrame({"slide_id": slides, "x": range(len(slides)), "y": 0})
    if splits is not None:
        frame["split"] = splits
    return frame


def test_clean_tables_pass() -> None:
    patches = _patches(["s1", "s1", "s2", "s3", "s4"], ["train", "train", "train", "val", "test"])
    assert check_patch_split_leakage(_slide_splits(), patches) == []
    assert_no_patch_leakage(_slide_splits(), patches)
    summary = slide_split_summary(_slide_splits(), patches).set_index("split")
    assert summary.loc["train", "patches"] == 3 and summary.loc["train", "patients"] == 1
    assert summary.loc["train", "slides"] == 2


def test_problems_are_named() -> None:
    splits = _slide_splits()
    splits.loc[1, "split"] = "test"  # patient P1 now has slides in train and test
    problems = check_patch_split_leakage(splits, _patches(["s1", "s2"]))
    assert any("more than one split" in p for p in problems)
    assert any("both train and test" in p for p in problems)

    problems = check_patch_split_leakage(_slide_splits(), _patches(["s1", "s9"]))
    assert any("no split" in p for p in problems)

    problems = check_patch_split_leakage(
        _slide_splits(), _patches(["s1", "s3"], ["train", "train"])
    )
    assert any("different from their slide" in p for p in problems)

    with pytest.raises(AssertionError, match="leakage"):
        assert_no_patch_leakage(_slide_splits(), _patches(["s9"]))

    missing = check_patch_split_leakage(_slide_splits().drop(columns=["split"]), _patches(["s1"]))
    assert missing == ["slide_splits lacks column 'split'"]
