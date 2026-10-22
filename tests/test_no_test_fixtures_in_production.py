"""Production code must never import the test fixtures (synthetic slides live only in tests/).

Team rule (Sprint 1, Adım 1): mock/synthetic data is not allowed in the pipeline. The synthetic
slide generator is a test fixture (``tests/histo_synthetic_fixture.py``); nothing under
``models/``, ``onkos/``, ``services/`` or ``scripts/`` may import ``tests`` or mention the
fixture modules.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRODUCTION_DIRS = ("models", "onkos", "services", "scripts")
FORBIDDEN_ROOTS = {"tests"}
FORBIDDEN_NAMES = ("histo_synthetic_fixture", "histo_fixtures", "write_synthetic_slide")


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_production_code_does_not_import_test_fixtures() -> None:
    violations: list[str] = []
    for folder in PRODUCTION_DIRS:
        for py in (REPO / folder).rglob("*.py"):
            for imported in _imports(py):
                if imported.split(".")[0] in FORBIDDEN_ROOTS:
                    violations.append(f"{py.relative_to(REPO)} imports {imported}")
            text = py.read_text(encoding="utf-8")
            for name in FORBIDDEN_NAMES:
                if name in text:
                    violations.append(f"{py.relative_to(REPO)} mentions {name}")
    assert not violations, "\n".join(violations)


def test_synthetic_generator_lives_only_under_tests() -> None:
    assert not (REPO / "models" / "histo" / "wsi" / "synthetic.py").exists()
    fixture = REPO / "tests" / "histo_synthetic_fixture.py"
    assert fixture.exists()
    assert "TEST FIXTURE" in fixture.read_text(encoding="utf-8").splitlines()[0]
