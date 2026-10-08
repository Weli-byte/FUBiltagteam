import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from onkos.contracts.mock_server import make_server
from onkos.contracts.schemas import MutationPrediction, ProblemDetails

EXAMPLES = Path(__file__).resolve().parents[1] / "onkos/contracts/examples"


@pytest.fixture
def base_url() -> Iterator[str]:
    server = make_server("127.0.0.1", 0, seed=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _post(url: str, body: bytes) -> tuple[int, dict[str, Any], str]:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.load(resp), resp.headers["Content-Type"]
    except urllib.error.HTTPError as err:
        return err.code, json.load(err), err.headers["Content-Type"]


@pytest.mark.parametrize(
    "path,example",
    [
        ("/predict/mutation", "mutation_request"),
        ("/recommend/drugs", "drug_request"),
        ("/report", "report_request"),
        ("/simulate", "simulation_request"),
        ("/explain", "explain_request"),
    ],
)
def test_each_endpoint_answers_its_example_request(base_url: str, path: str, example: str) -> None:
    status, body, _ = _post(base_url + path, (EXAMPLES / f"{example}.json").read_bytes())
    assert status == 200
    assert body["research_use_only"] is True


def test_response_echoes_request_and_validates(base_url: str) -> None:
    raw = json.loads((EXAMPLES / "mutation_request.json").read_text(encoding="utf-8"))
    raw["genes"] = ["EGFR"]
    _, body, _ = _post(base_url + "/predict/mutation", json.dumps(raw).encode())
    parsed = MutationPrediction.model_validate(body)
    assert parsed.slide_id == raw["slide_id"]
    assert [p.gene.value for p in parsed.predictions] == ["EGFR"]


def test_invalid_request_returns_problem_json(base_url: str) -> None:
    status, body, ctype = _post(base_url + "/predict/mutation", b'{"slide_id": "TCGA-AA-1234"}')
    assert status == 422 and ctype == "application/problem+json"
    problem = ProblemDetails.model_validate(body)
    assert problem.code.value == "validation_error" and problem.errors


def test_unknown_path_and_health(base_url: str) -> None:
    status, _, ctype = _post(base_url + "/nope", b"{}")
    assert status == 404 and ctype == "application/problem+json"
    with urllib.request.urlopen(base_url + "/health", timeout=5) as resp:
        assert json.load(resp)["status"] == "ok"
