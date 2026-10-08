"""Regenerate committed contract artefacts from the Pydantic models.

Writes onkos/contracts/openapi.yaml, onkos/contracts/examples/*.json and
docs/veri_sozlesmesi_alanlar.md. Run after ANY change to onkos/contracts/schemas.py.
"""

from pathlib import Path

from onkos.contracts.mock import MockFactory
from onkos.contracts.mock_server import dump_json
from onkos.contracts.openapi import field_reference_markdown, openapi_yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "onkos" / "contracts"


def example_files() -> dict[str, str]:
    """``{file name: JSON text}`` for one request/response example per endpoint."""
    f = MockFactory(seed=42)
    report_req = f.report_request()
    mutation_req = f.mutation_request()
    drug_req = f.drug_request()
    sim_req = f.simulation_request()
    explain_req = f.explain_request()
    return {
        "mutation_request.json": dump_json(mutation_req),
        "mutation_response.json": dump_json(f.mutation_prediction(mutation_req)),
        "drug_request.json": dump_json(drug_req),
        "drug_response.json": dump_json(f.drug_recommendation(drug_req)),
        "report_request.json": dump_json(report_req),
        "report_response.json": dump_json(f.report_response(report_req)),
        "simulation_request.json": dump_json(sim_req),
        "simulation_response.json": dump_json(f.simulation_response(sim_req)),
        "explain_request.json": dump_json(explain_req),
        "explain_response.json": dump_json(f.explain_response(explain_req)),
    }


def main() -> None:
    (CONTRACTS / "openapi.yaml").write_text(openapi_yaml(), encoding="utf-8")
    examples = CONTRACTS / "examples"
    examples.mkdir(exist_ok=True)
    for name, text in example_files().items():
        (examples / name).write_text(text, encoding="utf-8")
    (ROOT / "docs" / "veri_sozlesmesi_alanlar.md").write_text(
        field_reference_markdown(), encoding="utf-8"
    )
    print("contracts exported")


if __name__ == "__main__":
    main()
