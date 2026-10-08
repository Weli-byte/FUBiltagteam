import json
from typing import Any

import pytest
from pydantic import ValidationError

from onkos.contracts import SCHEMA_VERSION
from onkos.contracts.mock import MockFactory
from onkos.contracts.schemas import (
    ConfidenceInterval,
    DrugRecommendation,
    GeneMutationScore,
    MutationPrediction,
    MutationPredictionRequest,
    ReportRequest,
    SimulationResponse,
)


def _good_score(**override: Any) -> dict[str, Any]:
    base = {
        "gene": "TP53",
        "probability": 0.5,
        "uncertainty": 0.05,
        "ci": {"lower": 0.4, "upper": 0.6},
    }
    return {**base, **override}


def test_default_message_carries_version_and_disclaimer() -> None:
    msg = MockFactory(1).mutation_prediction()
    assert msg.schema_version == SCHEMA_VERSION
    assert msg.research_use_only is True
    assert "klinik" in msg.disclaimer


def test_patient_barcode_rejected_as_slide_id() -> None:
    with pytest.raises(ValidationError, match="hasta kimliği"):
        MutationPredictionRequest(slide_id="TCGA-AA-1234", patch_features_uri="x")


@pytest.mark.parametrize("bad", ["abc", "has space!", "x" * 65])
def test_slide_id_pattern(bad: str) -> None:
    with pytest.raises(ValidationError):
        MutationPredictionRequest(slide_id=bad, patch_features_uri="x")


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        MutationPredictionRequest(slide_id="slide_0001", patch_features_uri="x", patient_name="x")  # type: ignore[call-arg]


def test_major_version_mismatch_rejected_minor_accepted() -> None:
    major = int(SCHEMA_VERSION.split(".")[0])
    ok = MockFactory(0).mutation_request().model_dump()
    assert MutationPredictionRequest.model_validate({**ok, "schema_version": f"{major}.9.9"})
    with pytest.raises(ValidationError, match="uyumsuz"):
        MutationPredictionRequest.model_validate({**ok, "schema_version": f"{major + 1}.0.0"})


def test_probability_must_lie_in_confidence_interval() -> None:
    assert GeneMutationScore.model_validate(_good_score())
    with pytest.raises(ValidationError):
        GeneMutationScore.model_validate(_good_score(probability=0.9))


def test_confidence_interval_order_and_range() -> None:
    with pytest.raises(ValidationError):
        ConfidenceInterval(lower=0.7, upper=0.2)
    with pytest.raises(ValidationError):
        ConfidenceInterval(lower=-0.1, upper=0.2)


def test_duplicate_genes_rejected() -> None:
    data = json.loads(MockFactory(2).mutation_prediction().model_dump_json())
    data["predictions"].append(data["predictions"][0])
    with pytest.raises(ValidationError, match="en fazla bir"):
        MutationPrediction.model_validate(data)


def test_drug_ranks_and_scores_must_be_ordered() -> None:
    data = json.loads(MockFactory(3).drug_recommendation().model_dump_json())
    data["candidates"][0]["rank"] = 2
    with pytest.raises(ValidationError, match="ardışık"):
        DrugRecommendation.model_validate(data)
    data = json.loads(MockFactory(3).drug_recommendation().model_dump_json())
    data["candidates"][0]["compatibility_score"] = 0.0
    with pytest.raises(ValidationError, match="azalan"):
        DrugRecommendation.model_validate(data)


def test_report_request_requires_matching_slide_ids() -> None:
    data = json.loads(MockFactory(4).report_request().model_dump_json())
    data["slide_id"] = "other_slide"
    with pytest.raises(ValidationError, match="aynı olmalı"):
        ReportRequest.model_validate(data)


def test_simulation_is_always_marked_conceptual() -> None:
    f = MockFactory(5)
    resp = f.simulation_response(f.simulation_request())
    assert resp.is_conceptual is True and "doğrulanmamış" in resp.limitations
    data = json.loads(resp.model_dump_json())
    data["is_conceptual"] = False
    with pytest.raises(ValidationError):
        SimulationResponse.model_validate(data)
    data = json.loads(resp.model_dump_json())
    data["images"] = [i for i in data["images"] if i["role"] == "before"] * 2
    with pytest.raises(ValidationError, match="before"):
        SimulationResponse.model_validate(data)


def test_research_use_only_cannot_be_disabled() -> None:
    data = json.loads(MockFactory(6).mutation_prediction().model_dump_json())
    data["research_use_only"] = False
    with pytest.raises(ValidationError):
        MutationPrediction.model_validate(data)


@pytest.mark.parametrize("seed", range(100))
def test_every_mock_validates_for_many_seeds(seed: int) -> None:
    f = MockFactory(seed)
    req = f.report_request()
    for message in (
        f.mutation_prediction(),
        f.drug_recommendation(),
        req,
        f.report_response(req),
        f.simulation_response(f.simulation_request()),
        f.explain_response(f.explain_request()),
    ):
        type(message).model_validate_json(message.model_dump_json())


def test_mock_is_deterministic() -> None:
    assert MockFactory(7).mutation_prediction() == MockFactory(7).mutation_prediction()
    assert MockFactory(7).mutation_prediction() != MockFactory(8).mutation_prediction()
