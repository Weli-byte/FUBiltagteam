from typing import Any

import pytest
from pydantic import ValidationError

from onkos.contracts import SCHEMA_VERSION
from onkos.contracts.schemas import (
    ConfidenceInterval,
    DrugRecommendation,
    ExplainResponse,
    GeneMutationScore,
    MutationPrediction,
    MutationPredictionRequest,
    ReportRequest,
    SimulationResponse,
)
from tests import contract_fixtures as fx


def test_default_message_carries_version_and_disclaimer() -> None:
    msg = MutationPrediction.model_validate(fx.mutation_prediction())
    assert msg.schema_version == SCHEMA_VERSION
    assert msg.research_use_only is True
    assert "klinik" in msg.disclaimer


def test_every_fixture_validates() -> None:
    DrugRecommendation.model_validate(fx.drug_recommendation())
    ReportRequest.model_validate(fx.report_request())
    SimulationResponse.model_validate(fx.simulation_response())
    ExplainResponse.model_validate(fx.explain_response())
    for model_name, payload in fx.request_dicts().items():
        import onkos.contracts.schemas as schemas

        getattr(schemas, model_name).model_validate(payload)


def test_patient_barcode_rejected_as_slide_id() -> None:
    with pytest.raises(ValidationError, match="hasta kimliği"):
        MutationPredictionRequest(slide_id="TCGA-AA-1234", patch_features_uri="x")


@pytest.mark.parametrize("bad", ["abc", "has space!", "x" * 65])
def test_slide_id_pattern(bad: str) -> None:
    with pytest.raises(ValidationError):
        MutationPredictionRequest(slide_id=bad, patch_features_uri="x")


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        MutationPredictionRequest.model_validate(
            {"slide_id": "slide_0001", "patch_features_uri": "x", "patient_name": "x"}
        )


def test_major_version_mismatch_rejected_minor_accepted() -> None:
    major = int(SCHEMA_VERSION.split(".")[0])
    ok = fx.request_dicts()["MutationPredictionRequest"]
    assert MutationPredictionRequest.model_validate({**ok, "schema_version": f"{major}.9.9"})
    with pytest.raises(ValidationError, match="uyumsuz"):
        MutationPredictionRequest.model_validate({**ok, "schema_version": f"{major + 1}.0.0"})


def test_probability_must_lie_in_confidence_interval() -> None:
    assert GeneMutationScore.model_validate(fx.gene_score())
    with pytest.raises(ValidationError):
        GeneMutationScore.model_validate({**fx.gene_score(), "probability": 0.95})


def test_confidence_interval_order_and_range() -> None:
    with pytest.raises(ValidationError):
        ConfidenceInterval(lower=0.7, upper=0.2)
    with pytest.raises(ValidationError):
        ConfidenceInterval(lower=-0.1, upper=0.2)


def test_duplicate_genes_rejected() -> None:
    data = fx.mutation_prediction()
    data["predictions"].append(data["predictions"][0])
    with pytest.raises(ValidationError, match="en fazla bir"):
        MutationPrediction.model_validate(data)


def test_drug_ranks_and_scores_must_be_ordered() -> None:
    data = fx.drug_recommendation()
    data["candidates"][0]["rank"] = 2
    with pytest.raises(ValidationError, match="ardışık"):
        DrugRecommendation.model_validate(data)
    data = fx.drug_recommendation()
    data["candidates"][0]["compatibility_score"] = 0.0
    with pytest.raises(ValidationError, match="azalan"):
        DrugRecommendation.model_validate(data)


def test_report_request_requires_matching_slide_ids() -> None:
    data: dict[str, Any] = fx.report_request()
    data["slide_id"] = "other_slide"
    with pytest.raises(ValidationError, match="aynı olmalı"):
        ReportRequest.model_validate(data)


def test_simulation_is_always_marked_conceptual() -> None:
    resp = SimulationResponse.model_validate(fx.simulation_response())
    assert resp.is_conceptual is True and "doğrulanmamış" in resp.limitations
    with pytest.raises(ValidationError):
        SimulationResponse.model_validate({**fx.simulation_response(), "is_conceptual": False})
    both_before = fx.simulation_response()
    both_before["images"] = [both_before["images"][0]] * 2
    with pytest.raises(ValidationError, match="before"):
        SimulationResponse.model_validate(both_before)


def test_research_use_only_cannot_be_disabled() -> None:
    with pytest.raises(ValidationError):
        MutationPrediction.model_validate({**fx.mutation_prediction(), "research_use_only": False})
