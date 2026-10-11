"""Tests for MAF reading and patient-level mutation labels (mini sprint 1.5).

MAF files here are tiny synthetic tables that follow the assumed GDC layout; they exist only to
exercise the labelling rules. Real-file behaviour is verified by running the script on real data.
"""

import gzip
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("matplotlib")

import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from models.omics.labels import (  # noqa: E402
    STATUS_LABELED,
    STATUS_NO_MAF,
    STATUS_NO_PRIMARY,
    LabelRules,
    build_labels,
    class_balance,
    collect_maf,
    read_maf_file,
    recommend,
    rules_from_config,
    sample_of,
    sample_type_code,
)

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "Hugo_Symbol",
    "Variant_Classification",
    "Tumor_Sample_Barcode",
    "HGVSp_Short",
    "Exon_Number",
    "Chromosome",
    "Start_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
]


def v(
    gene: str, cls: str, sample: str, hgvs: str = "p.X1Y", exon: str = "5/10", pos: int = 100
) -> list[Any]:
    return [gene, cls, sample + "-11D-A1ZV-09", hgvs, exon, "7", pos, "C", "T"]


def write_maf(path: Path, rows: list[list[Any]], gz: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#version gdc-1.0.0", "#another comment", "\t".join(COLUMNS)]
    lines += ["\t".join(str(c) for c in row) for row in rows]
    text = "\n".join(lines) + "\n"
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def rules(**kw: Any) -> LabelRules:
    cfg = OmegaConf.load(ROOT / "configs" / "omics" / "labels.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.create(kw))
    return rules_from_config(cfg)  # type: ignore[arg-type]


def test_config_rules_loaded_from_yaml_not_code() -> None:
    r = rules()
    assert r.genes == ("TP53", "KRAS", "EGFR", "BRCA1")
    assert (
        "Silent" not in r.variant_classifications
        and "Missense_Mutation" in r.variant_classifications
    )
    assert rules(genes=["TP53", "PIK3CA"]).genes == ("TP53", "PIK3CA")
    with pytest.raises(ValueError, match="duplicates"):
        rules(genes=["TP53", "TP53"])
    # annotations of genes that are not listed are skipped, not an error
    assert rules(annotations=[{"name": "x", "gene": "NOTLISTED"}]).annotations == ()
    with pytest.raises(ValueError, match="annotation"):
        rules(annotations=[{"gene": "TP53"}])


def test_barcode_helpers() -> None:
    assert sample_of("TCGA-AB-1234-01A-11D-A1ZV-09") == "TCGA-AB-1234-01A"
    assert sample_type_code("TCGA-AB-1234-06A") == "06"


@pytest.mark.parametrize("gz", [False, True])
def test_read_maf_keeps_target_genes_and_all_samples(tmp_path: Path, gz: bool) -> None:
    s1, s2 = "TCGA-AB-1234-01A", "TCGA-AB-1234-01B"
    path = write_maf(
        tmp_path / ("a.maf.gz" if gz else "a.maf"),
        [v("TP53", "Missense_Mutation", s1), v("OTHER", "Silent", s2)],
        gz=gz,
    )
    read = read_maf_file(path, ["TP53"])
    assert list(read.rows["Hugo_Symbol"]) == ["TP53"] and read.samples == {s1, s2}
    assert read.n_rows_total == 2


def test_read_maf_reports_missing_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.maf"
    bad.write_text("Hugo_Symbol\tFoo\nTP53\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.maf.*missing MAF columns"):
        read_maf_file(bad, ["TP53"])


def _scenario(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a, b = "TCGA-AA-0001-01A", "TCGA-BB-0002-01A"
    d1, d2 = "TCGA-DD-0004-01A", "TCGA-DD-0004-01B"
    e = "TCGA-EE-0005-01A"
    f = "TCGA-FF-0006-06A"
    raw = tmp_path / "raw"
    files = {
        "A": [
            v("TP53", "Missense_Mutation", a),
            v("TP53", "Nonsense_Mutation", a, pos=200),
            v("KRAS", "Silent", a),
            v("EGFR", "Intron", a),
        ],
        "B": [v("TP53", "Silent", b), v("EGFR", "3'UTR", b), v("OTHER", "Missense_Mutation", b)],
        "D": [v("EGFR", "Missense_Mutation", d2, "p.L858R"), v("TP53", "Missense_Mutation", d1)],
        "E": [
            v("KRAS", "Frame_Shift_Del", e),
            v("EGFR", "In_Frame_Del", e, "p.E746_A750del", "19/28"),
        ],
        "F": [v("TP53", "Missense_Mutation", f)],
    }
    for key, rows in files.items():
        write_maf(raw / "TCGA-T" / "maf" / f"file{key}" / "x.maf", rows)
    pm = pd.DataFrame(
        [
            {
                "patient_barcode": "TCGA-AA-0001",
                "project_id": "TCGA-T",
                "rnaseq_sample_barcode": a,
                "n_maf": 1,
                "maf_file_ids": ["fileA"],
            },
            {
                "patient_barcode": "TCGA-BB-0002",
                "project_id": "TCGA-T",
                "rnaseq_sample_barcode": b,
                "n_maf": 1,
                "maf_file_ids": ["fileB"],
            },
            {
                "patient_barcode": "TCGA-CC-0003",
                "project_id": "TCGA-T",
                "rnaseq_sample_barcode": None,
                "n_maf": 0,
                "maf_file_ids": [],
            },
            # two primary samples; RNA-seq used 01A (has no EGFR hit), 01B has the EGFR L858R hit
            {
                "patient_barcode": "TCGA-DD-0004",
                "project_id": "TCGA-T",
                "rnaseq_sample_barcode": d1,
                "n_maf": 1,
                "maf_file_ids": ["fileD"],
            },
            {
                "patient_barcode": "TCGA-EE-0005",
                "project_id": "TCGA-T",
                "rnaseq_sample_barcode": e,
                "n_maf": 1,
                "maf_file_ids": ["fileE"],
            },
            {
                "patient_barcode": "TCGA-FF-0006",
                "project_id": "TCGA-T",
                "rnaseq_sample_barcode": None,
                "n_maf": 1,
                "maf_file_ids": ["fileF"],
            },
        ]
    )
    variants, samples = collect_maf(pm, raw, rules().genes)
    return variants, samples, pm


def _label(res: Any, patient: str, gene: str) -> Any:
    return res.labels.set_index("patient_barcode").loc[patient, gene]


def test_label_rules_end_to_end(tmp_path: Path) -> None:
    variants, samples, pm = _scenario(tmp_path)
    res = build_labels(variants, samples, pm, rules())
    # A: counted missense+nonsense in TP53 -> 1 (once, not twice); Silent KRAS / Intron EGFR -> 0
    assert [_label(res, "TCGA-AA-0001", g) for g in ("TP53", "KRAS", "EGFR", "BRCA1")] == [
        1,
        0,
        0,
        0,
    ]
    # B: only Silent/UTR in target genes -> wild-type (0), and these are recorded as not counted
    assert [_label(res, "TCGA-BB-0002", g) for g in ("TP53", "EGFR")] == [0, 0]
    detail_b = res.detail[res.detail["patient_barcode"] == "TCGA-BB-0002"]
    assert len(detail_b) == 2 and not detail_b["counted"].any()
    # C: no MAF -> unknown, never 0
    assert pd.isna(_label(res, "TCGA-CC-0003", "TP53"))
    # D: label comes from the RNA-seq sample (01A): TP53 yes, EGFR (only in 01B) no
    assert _label(res, "TCGA-DD-0004", "TP53") == 1 and _label(res, "TCGA-DD-0004", "EGFR") == 0
    assert res.labels.set_index("patient_barcode").loc["TCGA-DD-0004", "label_sample_barcode"] == (
        "TCGA-DD-0004-01A"
    )
    # E: frame shift counts; exon-19 in-frame deletion counts and is annotated
    assert _label(res, "TCGA-EE-0005", "KRAS") == 1 and _label(res, "TCGA-EE-0005", "EGFR") == 1
    ann = res.detail[res.detail["patient_barcode"] == "TCGA-EE-0005"]["annotations"]
    assert "EGFR_exon19_deletion" in set(ann)
    # F: only a metastatic sample -> unknown with the right status
    status = res.labels.set_index("patient_barcode")["label_status"]
    assert status["TCGA-FF-0006"] == STATUS_NO_PRIMARY and status["TCGA-CC-0003"] == STATUS_NO_MAF
    assert status["TCGA-AA-0001"] == STATUS_LABELED
    assert pd.isna(_label(res, "TCGA-FF-0006", "TP53"))
    # nobody is dropped; label dtype is nullable integer
    assert len(res.labels) == len(pm) == 6 and str(res.labels["TP53"].dtype) == "Int8"
    assert res.n_excluded_by_class == {"3'UTR": 1, "Intron": 1, "Silent": 2}


def test_rna_seq_sample_missing_from_maf_falls_back_to_lowest_primary(tmp_path: Path) -> None:
    variants, samples, pm = _scenario(tmp_path)
    pm.loc[pm["patient_barcode"] == "TCGA-DD-0004", "rnaseq_sample_barcode"] = "TCGA-DD-0004-01C"
    res = build_labels(variants, samples, pm, rules())
    chosen = res.labels.set_index("patient_barcode").loc["TCGA-DD-0004", "label_sample_barcode"]
    assert chosen == "TCGA-DD-0004-01A"


def test_new_gene_needs_only_a_config_line(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    s = "TCGA-AA-0001-01A"
    write_maf(
        raw / "TCGA-T" / "maf" / "f1" / "x.maf", [v("PIK3CA", "Missense_Mutation", s, "p.H1047R")]
    )
    pm = pd.DataFrame(
        [
            {
                "patient_barcode": "TCGA-AA-0001",
                "project_id": "TCGA-T",
                "rnaseq_sample_barcode": s,
                "n_maf": 1,
                "maf_file_ids": ["f1"],
            }
        ]
    )
    r = rules(genes=["TP53", "PIK3CA"])
    variants, samples = collect_maf(pm, raw, r.genes)
    res = build_labels(variants, samples, pm, r)
    assert _label(res, "TCGA-AA-0001", "PIK3CA") == 1 and _label(res, "TCGA-AA-0001", "TP53") == 0


def test_duplicate_variant_across_files_counts_once(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    s = "TCGA-AA-0001-01A"
    for fid in ("f1", "f2"):
        write_maf(raw / "TCGA-T" / "maf" / fid / "x.maf", [v("TP53", "Missense_Mutation", s)])
    pm = pd.DataFrame(
        [
            {
                "patient_barcode": "TCGA-AA-0001",
                "project_id": "TCGA-T",
                "rnaseq_sample_barcode": s,
                "n_maf": 2,
                "maf_file_ids": ["f1", "f2"],
            }
        ]
    )
    variants, samples = collect_maf(pm, raw, rules().genes)
    res = build_labels(variants, samples, pm, rules())
    assert len(res.detail) == 1 and _label(res, "TCGA-AA-0001", "TP53") == 1


def test_barcode_of_another_patient_is_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_maf(
        raw / "TCGA-T" / "maf" / "f1" / "x.maf",
        [v("TP53", "Missense_Mutation", "TCGA-ZZ-9999-01A")],
    )
    pm = pd.DataFrame(
        [
            {
                "patient_barcode": "TCGA-AA-0001",
                "project_id": "TCGA-T",
                "rnaseq_sample_barcode": None,
                "n_maf": 1,
                "maf_file_ids": ["f1"],
            }
        ]
    )
    with pytest.raises(ValueError, match="do not belong"):
        collect_maf(pm, raw, ["TP53"])


def test_missing_maf_file_is_reported(tmp_path: Path) -> None:
    pm = pd.DataFrame(
        [
            {
                "patient_barcode": "TCGA-AA-0001",
                "project_id": "TCGA-T",
                "rnaseq_sample_barcode": None,
                "n_maf": 1,
                "maf_file_ids": ["nope"],
            }
        ]
    )
    with pytest.raises(FileNotFoundError, match="not found"):
        collect_maf(pm, tmp_path, ["TP53"])


def test_labels_are_deterministic(tmp_path: Path) -> None:
    variants, samples, pm = _scenario(tmp_path)
    first = build_labels(variants, samples, pm, rules())
    second = build_labels(
        variants.sample(frac=1.0, random_state=1), samples, pm.iloc[::-1], rules()
    )
    pd.testing.assert_frame_equal(first.labels, second.labels)
    pd.testing.assert_frame_equal(
        first.detail.reset_index(drop=True), second.detail.reset_index(drop=True)
    )


def test_class_balance_flags_low_positives_and_recommends(tmp_path: Path) -> None:
    variants, samples, pm = _scenario(tmp_path)
    r = rules(min_positives=2)
    res = build_labels(variants, samples, pm, r)
    balance = class_balance(res.labels, r)
    tp53 = balance[(balance["project_id"] == "ALL") & (balance["gene"] == "TP53")].iloc[0]
    assert (tp53["positive"], tp53["negative"], tp53["unknown"], tp53["patients"]) == (
        2,
        2,
        2,
        6,
    )
    assert not tp53["low_positives"]
    brca1 = balance[(balance["project_id"] == "ALL") & (balance["gene"] == "BRCA1")].iloc[0]
    assert brca1["positive"] == 0 and brca1["low_positives"]
    text = recommend(balance, r).set_index(["project_id", "gene"])["recommendation"]
    assert "yetersiz pozitif" in text[("ALL", "BRCA1")]


def test_cli_end_to_end_writes_tables_and_report(tmp_path: Path) -> None:
    from scripts.build_mutation_labels import main

    _, _, pm = _scenario(tmp_path)
    pm_path = tmp_path / "patient_map.parquet"
    pm.to_parquet(pm_path, index=False)
    out, report = tmp_path / "out", tmp_path / "report"
    code = main(
        [
            "--patient-map",
            str(pm_path),
            "--raw",
            str(tmp_path / "raw"),
            "--out",
            str(out),
            "--report-dir",
            str(report),
            "min_positives=1",
        ]
    )
    assert code == 0
    labels = pd.read_parquet(out / "labels.parquet")
    assert len(labels) == 6 and set(["TP53", "KRAS", "EGFR", "BRCA1"]) <= set(labels.columns)
    detail = pd.read_parquet(out / "labels_variant_detail.parquet")
    assert "annotations" in detail.columns and "counted" in detail.columns
    text = (report / "dengesizlik_raporu.md").read_text(encoding="utf-8")
    assert "Sayılmayan varyantlar" in text and "TCGA-T" in text
    assert (report / "sinif_dengesi.png").stat().st_size > 1000
