"""Fail loudly, naming the consumer, when a field a consumer relies on disappears."""

from typing import Any

import pytest

from onkos.contracts.consumers import CONSUMER_EXPECTATIONS
from onkos.contracts.openapi import build_components

COMPONENTS = build_components()


def _resolve(prop: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in prop:
        resolved: dict[str, Any] = COMPONENTS[str(prop["$ref"]).rsplit("/", 1)[-1]]
        return resolved
    if "anyOf" in prop:
        options = [p for p in prop["anyOf"] if p.get("type") != "null"]
        return _resolve(options[0])
    return prop


def field_exists(model: str, path: str) -> bool:
    node = COMPONENTS.get(model)
    if node is None:
        return False
    for part in path.split("."):
        is_list = part.endswith("[]")
        name = part[:-2] if is_list else part
        prop = node.get("properties", {}).get(name)
        if prop is None:
            return False
        node = _resolve(prop)
        if is_list:
            if node.get("type") != "array":
                return False
            node = _resolve(node["items"])
    return True


CASES = [
    (consumer, model, path)
    for consumer, models in CONSUMER_EXPECTATIONS.items()
    for model, paths in models.items()
    for path in paths
]


@pytest.mark.parametrize(("consumer", "model", "path"), CASES)
def test_consumer_field_still_exists(consumer: str, model: str, path: str) -> None:
    assert field_exists(model, path), (
        f"'{consumer}' {model}.{path} alanına dayanıyor ama sözleşmede yok. "
        "Alanı geri koyun veya tüketici sahibinin onayıyla consumers.py'yi güncelleyin."
    )


def test_checker_detects_removed_and_renamed_fields() -> None:
    assert field_exists("MutationPrediction", "predictions[].ci.lower")
    assert not field_exists("MutationPrediction", "predictions[].confidence_interval")
    assert not field_exists("NoSuchModel", "x")
    assert not field_exists("MutationPrediction", "slide_id[]")
