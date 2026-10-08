"""Committed generated files must match the models (run scripts/export_contracts.py)."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from onkos.contracts.openapi import build_openapi, field_reference_markdown, openapi_yaml
from onkos.contracts.schemas import (
    DrugRecommendation,
    DrugRecommendationRequest,
    ExplainRequest,
    ExplainResponse,
    MutationPrediction,
    MutationPredictionRequest,
    ReportRequest,
    ReportResponse,
    SimulationRequest,
    SimulationResponse,
)
from scripts.export_contracts import example_files

ROOT = Path(__file__).resolve().parents[1]
HINT = "Şema değişti: `uv run python scripts/export_contracts.py` çalıştırıp çıktıyı commit edin."


def test_openapi_yaml_is_up_to_date() -> None:
    committed = (ROOT / "onkos/contracts/openapi.yaml").read_text(encoding="utf-8")
    assert committed == openapi_yaml(), HINT


def test_field_reference_is_up_to_date() -> None:
    committed = (ROOT / "docs/veri_sozlesmesi_alanlar.md").read_text(encoding="utf-8")
    assert committed == field_reference_markdown(), HINT


def test_examples_are_up_to_date() -> None:
    folder = ROOT / "onkos/contracts/examples"
    expected = example_files()
    assert {p.name for p in folder.glob("*.json")} == set(expected), HINT
    for name, text in expected.items():
        assert (folder / name).read_text(encoding="utf-8") == text, f"{name}: {HINT}"


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


@pytest.mark.parametrize(
    ("file", "model"),
    [
        ("mutation_request", MutationPredictionRequest),
        ("mutation_response", MutationPrediction),
        ("drug_request", DrugRecommendationRequest),
        ("drug_response", DrugRecommendation),
        ("report_request", ReportRequest),
        ("report_response", ReportResponse),
        ("simulation_request", SimulationRequest),
        ("simulation_response", SimulationResponse),
        ("explain_request", ExplainRequest),
        ("explain_response", ExplainResponse),
    ],
)
def test_committed_examples_validate_against_models(file: str, model: type[BaseModel]) -> None:
    data = json.loads((ROOT / f"onkos/contracts/examples/{file}.json").read_text(encoding="utf-8"))
    model.model_validate(data)
