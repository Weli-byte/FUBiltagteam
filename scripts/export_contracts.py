"""Regenerate committed contract artefacts from the Pydantic models.

Writes onkos/contracts/openapi.yaml and docs/veri_sozlesmesi_alanlar.md.
Run after ANY change to onkos/contracts/schemas.py.
"""

from pathlib import Path

from onkos.contracts.openapi import field_reference_markdown, openapi_yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    (ROOT / "onkos" / "contracts" / "openapi.yaml").write_text(openapi_yaml(), encoding="utf-8")
    (ROOT / "docs" / "veri_sozlesmesi_alanlar.md").write_text(
        field_reference_markdown(), encoding="utf-8"
    )
    print("contracts exported")


if __name__ == "__main__":
    main()
