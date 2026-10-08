"""Enforce ADR-0001: feature modules must not import each other's internals."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "onkos"
FEATURE_MODULES = {"histo", "omics", "gnn", "rag_llm", "diffusion", "xai", "api"}
SHARED = {"common", "contracts"}


def _imported_onkos_modules(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[0] == "onkos" and len(parts) > 1:
                found.add(parts[1])
    return found


def test_feature_modules_do_not_import_each_other() -> None:
    violations: list[str] = []
    for module in FEATURE_MODULES:
        for py in (ROOT / module).rglob("*.py"):
            for target in _imported_onkos_modules(py) - SHARED - {module}:
                violations.append(f"{py.relative_to(ROOT.parent)} imports onkos.{target}")
    assert not violations, "\n".join(violations)


def test_all_five_architecture_layers_have_a_home() -> None:
    for module in FEATURE_MODULES | SHARED:
        assert (ROOT / module / "__init__.py").exists(), module
