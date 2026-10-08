"""Tiny stdlib HTTP server that serves contract-valid mock responses.

Lets the frontend (and any consumer) develop without the real backend. It validates incoming
requests with the real Pydantic models and returns ``application/problem+json`` errors.
Run: ``uv run python scripts/mock_server.py`` (default http://127.0.0.1:8000).
"""

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pydantic import BaseModel, ValidationError

from onkos.contracts.mock import MockFactory
from onkos.contracts.openapi import OPERATIONS, PROBLEM_JSON
from onkos.contracts.schemas import (
    ErrorCode,
    FieldError,
    HealthResponse,
    ProblemDetails,
)


def _handlers(factory: MockFactory) -> dict[str, Callable[[Any], BaseModel]]:
    return {
        "/predict/mutation": factory.mutation_prediction,
        "/recommend/drugs": factory.drug_recommendation,
        "/report": factory.report_response,
        "/simulate": factory.simulation_response,
        "/explain": factory.explain_response,
    }


def make_server(host: str = "127.0.0.1", port: int = 8000, seed: int = 0) -> ThreadingHTTPServer:
    """Create (but do not start) the mock server. Use ``port=0`` for a free port."""
    handlers = _handlers(MockFactory(seed))

    class Handler(BaseHTTPRequestHandler):
        def _send(
            self, status: int, body: BaseModel, content_type: str = "application/json"
        ) -> None:
            payload = body.model_dump_json().encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")  # dev only
            self.end_headers()
            self.wfile.write(payload)

        def _problem(self, status: int, code: ErrorCode, title: str, **extra: Any) -> None:
            self._send(
                status,
                ProblemDetails(title=title, status=status, code=code, instance=self.path, **extra),
                PROBLEM_JSON,
            )

        def do_OPTIONS(self) -> None:  # noqa: N802 (stdlib naming)
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, HealthResponse())
            else:
                self._problem(404, ErrorCode.SLIDE_NOT_FOUND, "Bulunamadı")

        def do_POST(self) -> None:  # noqa: N802
            operation = OPERATIONS.get(self.path)
            if operation is None:
                self._problem(404, ErrorCode.SLIDE_NOT_FOUND, "Bulunamadı")
                return
            request_model = operation[2]
            try:
                length = int(self.headers.get("Content-Length", 0))
                request = request_model.model_validate_json(self.rfile.read(length))
            except ValidationError as exc:
                errors = [
                    FieldError(loc=list(e["loc"]), msg=e["msg"], type=e["type"])
                    for e in exc.errors()
                ]
                self._problem(422, ErrorCode.VALIDATION_ERROR, "Doğrulama hatası", errors=errors)
                return
            self._send(200, handlers[self.path](request))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return ThreadingHTTPServer((host, port), Handler)


def dump_json(model: BaseModel) -> str:
    return json.dumps(json.loads(model.model_dump_json()), indent=2, ensure_ascii=False) + "\n"
