"""Committed generated files must match the models (run scripts/export_contracts.py)."""

from pathlib import Path

import pytest
import yaml

from onkos.contracts.openapi import build_openapi, field_reference_markdown, openapi_yaml

ROOT = Path(__file__).resolve().parents[1]
HINT = "Şema değişti: `uv run python scripts/export_contracts.py` çalıştırıp çıktıyı commit edin."


def test_openapi_yaml_is_up_to_date() -> None:
    committed = (ROOT / "onkos/contracts/openapi.yaml").read_text(encoding="utf-8")
    assert committed == openapi_yaml(), HINT


def test_field_reference_is_up_to_date() -> None:
    committed = (ROOT / "docs/veri_sozlesmesi_alanlar.md").read_text(encoding="utf-8")
    assert committed == field_reference_markdown(), HINT


def test_openapi_document_is_valid() -> None:
    validator = pytest.importorskip("openapi_spec_validator")
    validator.validate(yaml.safe_load(openapi_yaml()))


def test_all_five_endpoints_and_health_exist() -> None:
    paths = set(build_openapi()["paths"])
    assert paths == {
        "/predict/mutation",
        "/recommend/drugs",
        "/report",
        "/simulate",
        "/explain",
        "/health",
    }
