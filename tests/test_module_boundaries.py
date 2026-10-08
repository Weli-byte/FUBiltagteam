"""Enforce ADR-0001: feature modules must not import each other's internals.

Layout: models/<module>/, services/api/ are feature modules; onkos/common and
onkos/contracts are the only shared packages they may import.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL_MODULES = {"histo", "omics", "gnn", "rag_llm", "diffusion", "xai"}
SHARED = {"onkos.common", "onkos.contracts"}


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _feature_modules() -> dict[str, Path]:
    modules = {f"models.{m}": REPO / "models" / m for m in MODEL_MODULES}
    modules["services.api"] = REPO / "services" / "api"
    return modules


def _is_allowed(imported: str, own: str) -> bool:
    if imported == own or imported.startswith(own + "."):
        return True
    if any(imported == s or imported.startswith(s + ".") for s in SHARED):
        return True
    root = imported.split(".")[0]
    # Any other import from our own top-level packages is a boundary violation.
    return root not in {"models", "services", "onkos"} or imported == "onkos"


def test_feature_modules_do_not_import_each_other() -> None:
    violations: list[str] = []
    for own, folder in _feature_modules().items():
        for py in folder.rglob("*.py"):
            for imported in _imports(py):
                if not _is_allowed(imported, own):
                    violations.append(f"{py.relative_to(REPO)} imports {imported}")
    assert not violations, "\n".join(violations)


def test_all_architecture_layers_have_a_home() -> None:
    for folder in _feature_modules().values():
        assert (folder / "__init__.py").exists(), folder
    for shared in ("common", "contracts"):
        assert (REPO / "onkos" / shared / "__init__.py").exists(), shared
