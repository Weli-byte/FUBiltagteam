"""Leakage checks between the project leader's patient-level splits and the patch data (1.10).

Bu dosya ne işe yarar? Proje liderinin ürettiği ``slide_splits.parquet`` (slayt ↔ hasta ↔ küme)
ile Nisa'nın patch/slayt tablolarını karşılaştırır: aynı hasta iki kümede olamaz, bir slaytın
tüm patch'leri tek kümede olmalı, bölme dosyasında olmayan slayt eğitime giremez. Sorunları
liste olarak döndürür; boş liste = temiz.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


def check_patch_split_leakage(
    slide_splits: pd.DataFrame,
    patches: pd.DataFrame,
    slide_id_column: str = "slide_id",
    required_splits: Sequence[str] = SPLIT_NAMES,
) -> list[str]:
    """Return every problem found between ``slide_splits`` and a patch/slide table.

    ``slide_splits`` needs ``slide_file_id``, ``patient_barcode``, ``split``;
    ``patches`` needs ``slide_id`` (= ``slide_file_id``) and optionally ``split``.
    """
    problems: list[str] = []
    for column in ("slide_file_id", "patient_barcode", "split"):
        if column not in slide_splits.columns:
            problems.append(f"slide_splits lacks column {column!r}")
    if slide_id_column not in patches.columns:
        problems.append(f"patches lacks column {slide_id_column!r}")
    if problems:
        return problems

    unknown = set(slide_splits["split"]) - set(required_splits)
    if unknown:
        problems.append(f"unknown split names {sorted(unknown)}")
    if slide_splits["slide_file_id"].duplicated().any():
        problems.append("a slide appears more than once in slide_splits")
    per_patient = slide_splits.groupby("patient_barcode")["split"].nunique()
    multi = per_patient[per_patient > 1]
    if len(multi):
        problems.append(f"{len(multi)} patients have slides in more than one split")

    split_of = dict(zip(slide_splits["slide_file_id"], slide_splits["split"], strict=True))
    patient_of = dict(
        zip(slide_splits["slide_file_id"], slide_splits["patient_barcode"], strict=True)
    )
    patch_slides = set(patches[slide_id_column].astype(str))
    missing = sorted(patch_slides - set(split_of))
    if missing:
        problems.append(f"{len(missing)} patch slides have no split (e.g. {missing[:3]})")
    if "split" in patches.columns:
        mismatch = patches[
            patches[slide_id_column].astype(str).map(split_of).fillna("?") != patches["split"]
        ]
        mismatch = mismatch[mismatch[slide_id_column].astype(str).isin(split_of)]
        if len(mismatch):
            problems.append(f"{len(mismatch)} patches carry a split different from their slide's")
    patients_by_split: dict[str, set[str]] = {name: set() for name in required_splits}
    for slide in patch_slides & set(split_of):
        patients_by_split.setdefault(split_of[slide], set()).add(patient_of[slide])
    names = list(required_splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = patients_by_split[a] & patients_by_split[b]
            if both:
                problems.append(f"{len(both)} patients have patches in both {a} and {b}")
    return problems


def assert_no_patch_leakage(slide_splits: pd.DataFrame, patches: pd.DataFrame) -> None:
    problems = check_patch_split_leakage(slide_splits, patches)
    if problems:
        raise AssertionError("patch/split leakage check failed:\n- " + "\n- ".join(problems))


def slide_split_summary(slide_splits: pd.DataFrame, patches: pd.DataFrame) -> pd.DataFrame:
    """Per-split counts of slides, patients and patches (aggregates only; safe to commit)."""
    merged = patches.merge(
        slide_splits[["slide_file_id", "patient_barcode", "split"]],
        left_on="slide_id",
        right_on="slide_file_id",
        how="left",
        suffixes=("_patch", ""),
    )
    merged["split"] = merged["split"].fillna("unassigned")
    return (
        merged.groupby("split")
        .agg(
            slides=("slide_id", "nunique"),
            patients=("patient_barcode", "nunique"),
            patches=("slide_id", "size"),
        )
        .reset_index()
    )
