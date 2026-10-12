"""TCGA slide barcode parsing (mini sprint 1.4).

Bu dosya ne işe yarar? Bir slayt dosya adından (ör. ``TCGA-A1-A0SB-01Z-00-DX1.<uuid>.svs``)
hangi hastaya ait olduğunu, örneğin tümör mü normal doku mu olduğunu ve slaytın tanı (DX, FFPE)
mı dondurulmuş kesit (TS/BS/MS) mi olduğunu çıkarır. Barkod yapısı:
``Proje-TSS-Katılımcı-Örnek(2 hane + vial)-Porsiyon-Slayt`` (GDC Encyclopedia, TCGA Barcode).
Örnek türü kodları (01 birincil tümör, 11 normal doku ...) GDC tablosundan alınmıştır; tam
liste resmi sayfadan doğrulanmalıdır.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# GDC "Sample Type Codes" (subset). Verify the complete table at
# https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/ before relying on rarer codes.
SAMPLE_TYPE_CODES: dict[str, str] = {
    "01": "Primary Solid Tumor",
    "02": "Recurrent Solid Tumor",
    "03": "Primary Blood Derived Cancer - Peripheral Blood",
    "05": "Additional - New Primary",
    "06": "Metastatic",
    "07": "Additional Metastatic",
    "10": "Blood Derived Normal",
    "11": "Solid Tissue Normal",
}
SLIDE_TYPES: dict[str, str] = {
    "DX": "diagnostic slide (FFPE, H&E)",
    "TS": "top slide (frozen section)",
    "BS": "bottom slide (frozen section)",
    "MS": "middle slide (frozen section)",
}
_SLIDE_RE = re.compile(
    r"^(?P<project>TCGA)-(?P<tss>[A-Z0-9]{2})-(?P<participant>[A-Z0-9]{4})-"
    r"(?P<sample_type>\d{2})(?P<vial>[A-Z])-(?P<portion>\d{2})-(?P<slide_type>DX|TS|BS|MS)"
    r"(?P<slide_number>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SlideBarcode:
    patient_barcode: str  # TCGA-XX-XXXX (12 characters)
    tss: str
    sample_type_code: str
    vial: str
    portion: str
    slide_type: str
    slide_number: int

    @property
    def sample_barcode(self) -> str:
        return f"{self.patient_barcode}-{self.sample_type_code}{self.vial}"

    @property
    def slide_barcode(self) -> str:
        return f"{self.sample_barcode}-{self.portion}-{self.slide_type}{self.slide_number}"

    @property
    def sample_type(self) -> str:
        return SAMPLE_TYPE_CODES.get(self.sample_type_code, f"unknown ({self.sample_type_code})")

    @property
    def is_diagnostic(self) -> bool:
        return self.slide_type == "DX"

    @property
    def is_primary_tumor(self) -> bool:
        return self.sample_type_code == "01"

    @property
    def is_normal(self) -> bool:
        return self.sample_type_code in ("10", "11")


def parse_slide_barcode(name: str) -> SlideBarcode:
    """Parse a slide barcode or a file name that starts with one; raise ``ValueError`` otherwise."""
    match = _SLIDE_RE.match(name.strip())
    if not match:
        raise ValueError(f"not a TCGA slide barcode: {name!r}")
    g = match.groupdict()
    return SlideBarcode(
        patient_barcode=f"{g['project'].upper()}-{g['tss'].upper()}-{g['participant'].upper()}",
        tss=g["tss"].upper(),
        sample_type_code=g["sample_type"],
        vial=g["vial"].upper(),
        portion=g["portion"],
        slide_type=g["slide_type"].upper(),
        slide_number=int(g["slide_number"]),
    )


def try_parse_slide_barcode(name: str) -> SlideBarcode | None:
    try:
        return parse_slide_barcode(name)
    except ValueError:
        return None
