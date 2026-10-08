"""Versioned data contracts between ONKOS modules (mini sprint 1.7).

Conventions
-----------
* Field names are English ``snake_case``; descriptions are Turkish and state meaning, unit, range.
* Every top-level message carries ``schema_version`` (``MAJOR.MINOR.PATCH``). A message whose
  MAJOR differs from :data:`onkos.contracts.SCHEMA_VERSION` is rejected. MINOR/PATCH bumps only add
  optional fields or tighten documentation, so existing producers keep working.
* Models forbid unknown fields, so typos in mocks and clients fail loudly.
* No field may carry a patient identifier. Slides are referenced by an anonymous ``slide_id``; TCGA
  barcodes are rejected.
* Every model output is research-use-only and says so (``research_use_only`` + ``disclaimer``).
"""

import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from onkos.contracts import SCHEMA_VERSION

DISCLAIMER = (
    "Araştırma prototipi çıktısıdır; klinik tanı veya tedavi kararı için kullanılamaz ve "
    "'kesin tanı' ya da 'reçete' değildir."
)
SIMULATION_LIMITATIONS = (
    "Kavramsal dijital ikiz gösterimidir; tedavi öncesi/sonrası eşleşmiş görüntü ile "
    "doğrulanmamıştır ve klinik olarak geçerli bir tedavi tahmini değildir."
)

_TCGA_BARCODE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.IGNORECASE)


def _reject_patient_barcode(value: str) -> str:
    if _TCGA_BARCODE.match(value):
        raise ValueError("TCGA barkodu hasta kimliği taşır; anonim slide_id kullanın")
    return value


SlideId = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z0-9_-]{4,64}$",
        description="Anonim slayt anahtarı (4-64 karakter: harf, rakam, _ ve -). "
        "Hasta kimliği veya TCGA barkodu OLAMAZ.",
    ),
    AfterValidator(_reject_patient_barcode),
]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class Gene(StrEnum):
    """Genes with a mutation-prediction head (extend here when a new gene is added)."""

    TP53 = "TP53"
    KRAS = "KRAS"
    EGFR = "EGFR"
    BRCA1 = "BRCA1"


class ErrorCode(StrEnum):
    """Machine-readable error codes returned in :class:`ProblemDetails`."""

    VALIDATION_ERROR = "validation_error"
    SLIDE_NOT_FOUND = "slide_not_found"
    FEATURES_INCOMPATIBLE = "features_incompatible"
    CALIBRATION_MISSING = "calibration_missing"
    MODEL_UNAVAILABLE = "model_unavailable"
    RATE_LIMITED = "rate_limited"
    INTERNAL_ERROR = "internal_error"


class ExplainMethod(StrEnum):
    GRADCAM = "gradcam"
    ATTENTION = "attention"


class ContractModel(BaseModel):
    """Base for nested value objects: strict (no unknown fields), whitespace-stripped strings."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class VersionedMessage(ContractModel):
    """Base for top-level request/response messages."""

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        pattern=r"^\d+\.\d+\.\d+$",
        description="Sözleşme sürümü (MAJOR.MINOR.PATCH). MAJOR uyuşmazsa mesaj reddedilir.",
    )

    @field_validator("schema_version")
    @classmethod
    def _same_major(cls, value: str) -> str:
        if value.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(
                f"uyumsuz sözleşme sürümü {value}; desteklenen MAJOR {SCHEMA_VERSION.split('.')[0]}"
            )
        return value


class ResearchOutput(VersionedMessage):
    """Base for every model output; carries the mandatory research-use disclaimer."""

    research_use_only: Literal[True] = Field(
        default=True, description="Her zaman true: çıktı yalnızca araştırma amaçlıdır."
    )
    disclaimer: str = Field(default=DISCLAIMER, description="Kullanıcıya gösterilecek uyarı metni.")


# --------------------------------------------------------------------------------------
# Shared value objects
# --------------------------------------------------------------------------------------
class ConfidenceInterval(ContractModel):
    lower: Probability = Field(description="Güven aralığı alt sınırı (olasılık, 0-1).")
    upper: Probability = Field(description="Güven aralığı üst sınırı (olasılık, 0-1).")
    level: float = Field(
        default=0.95, gt=0.0, lt=1.0, description="Aralığın güven düzeyi (örn. 0.95)."
    )

    @model_validator(mode="after")
    def _ordered(self) -> "ConfidenceInterval":
        if self.lower > self.upper:
            raise ValueError("lower <= upper olmalı")
        return self


class PatchRef(ContractModel):
    x: int = Field(ge=0, description="Patch sol-üst köşesi, seviye-0 piksel X koordinatı.")
    y: int = Field(ge=0, description="Patch sol-üst köşesi, seviye-0 piksel Y koordinatı.")


class PatchAttention(PatchRef):
    weight: Probability = Field(description="Patch'e verilen normalize dikkat ağırlığı (0-1).")


# --------------------------------------------------------------------------------------
# Module 2: mutation prediction (/predict/mutation)
# --------------------------------------------------------------------------------------
class GeneMutationScore(ContractModel):
    gene: Gene = Field(description="Hedef gen.")
    probability: Probability = Field(description="Kalibre edilmiş mutasyon olasılığı (0-1).")
    uncertainty: float = Field(
        ge=0.0, description="Belirsizlik: tahminin standart sapması (olasılık biriminde, >= 0)."
    )
    ci: ConfidenceInterval = Field(description="Olasılık için güven aralığı.")

    @model_validator(mode="after")
    def _ci_contains_probability(self) -> "GeneMutationScore":
        if not self.ci.lower <= self.probability <= self.ci.upper:
            raise ValueError("probability güven aralığının içinde olmalı")
        return self


class MutationPredictionRequest(VersionedMessage):
    slide_id: SlideId
    patch_features_uri: str = Field(
        min_length=1,
        description="Nisa'nın ürettiği patch özellik HDF5 dosyasının konumu (dosya yolu veya URI).",
    )
    omics_sample_id: SlideId | None = Field(
        default=None, description="Varsa omics örneğinin anonim anahtarı; yoksa yalnızca görüntü."
    )
    genes: list[Gene] = Field(
        default_factory=lambda: list(Gene),
        min_length=1,
        description="Tahmin istenen genler (varsayılan: hepsi).",
    )

    @field_validator("genes")
    @classmethod
    def _unique_genes(cls, value: list[Gene]) -> list[Gene]:
        if len(set(value)) != len(value):
            raise ValueError("genes tekrar içeremez")
        return value


class FusionAttention(ContractModel):
    image_weight: Probability = Field(description="Görüntü modalitesinin füzyon ağırlığı (0-1).")
    omics_weight: Probability = Field(description="Omics modalitesinin füzyon ağırlığı (0-1).")
    top_patches: list[PatchAttention] = Field(
        default_factory=list, max_length=50, description="En yüksek dikkatli patch'ler (<= 50)."
    )

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "FusionAttention":
        if abs(self.image_weight + self.omics_weight - 1.0) > 1e-3:
            raise ValueError("image_weight + omics_weight 1'e eşit olmalı")
        return self


class MutationPrediction(ResearchOutput):
    slide_id: SlideId
    predictions: list[GeneMutationScore] = Field(
        min_length=1, description="Gen başına mutasyon skorları (her gen en fazla bir kez)."
    )
    calibration_version: str = Field(
        min_length=1, description="Kalibrasyon modülünün sürümü (örn. 'isotonic-2026-01')."
    )
    model_version: str = Field(min_length=1, description="Mutasyon modelinin sürümü.")
    fusion_attention: FusionAttention | None = Field(
        default=None, description="Cross-attention füzyon dikkat ağırlıkları (varsa)."
    )

    @field_validator("predictions")
    @classmethod
    def _unique_genes(cls, value: list[GeneMutationScore]) -> list[GeneMutationScore]:
        genes = [score.gene for score in value]
        if len(set(genes)) != len(genes):
            raise ValueError("her gen için en fazla bir skor olmalı")
        return value


# --------------------------------------------------------------------------------------
# Module 3a: drug recommendation (/recommend/drugs)
# --------------------------------------------------------------------------------------
class DrugRecommendationRequest(VersionedMessage):
    slide_id: SlideId
    mutations: list[GeneMutationScore] = Field(
        min_length=1, description="Mutasyon profili (Modül 2 çıktısındaki skorlar)."
    )
    top_k: int = Field(default=5, ge=1, le=20, description="İstenen en fazla ilaç sayısı (1-20).")


class DrugCandidate(ContractModel):
    rank: int = Field(ge=1, description="Sıra (1 = en uyumlu).")
    drug_name: str = Field(min_length=1, description="İlaç adı (etken madde).")
    drugbank_id: str | None = Field(
        default=None, pattern=r"^DB\d{5}$", description="DrugBank kimliği (örn. DB00530)."
    )
    compatibility_score: Probability = Field(
        description="Mutasyon profili ile uyum skoru (0-1); göreli sıralama içindir, klinik etki "
        "olasılığı DEĞİLDİR."
    )
    target_proteins: list[str] = Field(default_factory=list, description="Hedef proteinler.")
    rationale: str | None = Field(default=None, description="Kısa gerekçe metni (varsa).")


class DrugRecommendation(ResearchOutput):
    slide_id: SlideId
    candidates: list[DrugCandidate] = Field(
        min_length=1, description="Uyum skoruna göre azalan sırada ilaç adayları."
    )
    model_version: str = Field(min_length=1, description="GNN modelinin sürümü.")
    knowledge_base_version: str | None = Field(
        default=None, description="Kullanılan ilaç veritabanı sürümü (örn. DrugBank sürümü)."
    )

    @model_validator(mode="after")
    def _ranked(self) -> "DrugRecommendation":
        ranks = [c.rank for c in self.candidates]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("rank değerleri 1'den başlayıp ardışık artmalı")
        scores = [c.compatibility_score for c in self.candidates]
        if any(a < b for a, b in zip(scores, scores[1:], strict=False)):
            raise ValueError("compatibility_score azalan sırada olmalı")
        return self


# --------------------------------------------------------------------------------------
# Module 3b: report (/report)
# --------------------------------------------------------------------------------------
class ReportRequest(VersionedMessage):
    slide_id: SlideId
    mutation_prediction: MutationPrediction
    drug_recommendation: DrugRecommendation | None = Field(
        default=None, description="Varsa ilaç önerisi; rapor bunu da özetler."
    )
    language: Literal["tr", "en"] = Field(default="tr", description="Rapor dili.")

    @model_validator(mode="after")
    def _same_slide(self) -> "ReportRequest":
        if self.mutation_prediction.slide_id != self.slide_id:
            raise ValueError("mutation_prediction.slide_id, slide_id ile aynı olmalı")
        if self.drug_recommendation and self.drug_recommendation.slide_id != self.slide_id:
            raise ValueError("drug_recommendation.slide_id, slide_id ile aynı olmalı")
        return self


class Citation(ContractModel):
    pmid: str | None = Field(default=None, pattern=r"^\d{1,9}$", description="PubMed kimliği.")
    title: str = Field(min_length=1, description="Kaynak başlığı.")
    year: int | None = Field(default=None, ge=1900, le=2100, description="Yayın yılı.")


class ReportResponse(ResearchOutput):
    slide_id: SlideId
    report_text: str = Field(min_length=1, description="Doktorun okuyacağı rapor metni.")
    citations: list[Citation] = Field(
        default_factory=list, description="Rapora dayanak kaynaklar (RAG)."
    )
    language: Literal["tr", "en"]
    llm_model: str = Field(min_length=1, description="Raporu üreten dil modelinin adı/sürümü.")
    retrieval_index_version: str | None = Field(
        default=None, description="Kullanılan literatür vektör indeksinin sürümü."
    )


# --------------------------------------------------------------------------------------
# Module 4: simulation (/simulate)
# --------------------------------------------------------------------------------------
class SimulationRequest(VersionedMessage):
    slide_id: SlideId
    drug_name: str = Field(min_length=1, description="Simüle edilecek ilaç.")
    region: PatchRef | None = Field(
        default=None, description="Simüle edilecek patch; yoksa sistem seçer."
    )
    seed: int = Field(default=0, ge=0, description="Tekrarlanabilirlik için rastgelelik tohumu.")
    num_inference_steps: int = Field(
        default=50, ge=1, le=1000, description="Difüzyon örnekleme adım sayısı (1-1000)."
    )


class SimulatedImage(ContractModel):
    role: Literal["before", "after"] = Field(description="Tedavi öncesi mi sonrası mı.")
    uri: str = Field(min_length=1, description="Görüntü konumu (dosya yolu veya URL).")
    width: int = Field(gt=0, description="Genişlik (piksel).")
    height: int = Field(gt=0, description="Yükseklik (piksel).")


class SimulationResponse(ResearchOutput):
    slide_id: SlideId
    drug_name: str
    images: list[SimulatedImage] = Field(
        min_length=2, description="En az bir 'before' ve bir 'after' görüntüsü."
    )
    seed: int = Field(ge=0)
    model_version: str = Field(min_length=1, description="Difüzyon modelinin sürümü.")
    is_conceptual: Literal[True] = Field(
        default=True, description="Her zaman true: simülasyon kavramsaldır, doğrulanmamıştır."
    )
    limitations: str = Field(
        default=SIMULATION_LIMITATIONS, description="Arayüzde gösterilecek sınırlılık metni."
    )

    @model_validator(mode="after")
    def _has_both_roles(self) -> "SimulationResponse":
        if {image.role for image in self.images} != {"before", "after"}:
            raise ValueError("images hem 'before' hem 'after' içermeli")
        return self


# --------------------------------------------------------------------------------------
# Module 5: explainability (/explain)
# --------------------------------------------------------------------------------------
class ExplainRequest(VersionedMessage):
    slide_id: SlideId
    method: ExplainMethod = Field(description="Açıklama yöntemi.")
    target_gene: Gene | None = Field(default=None, description="Açıklanacak gen (varsa).")


class HeatmapLayer(ContractModel):
    name: str = Field(min_length=1, description="Katman adı (arayüzde gösterilir).")
    method: ExplainMethod
    uri: str = Field(min_length=1, description="Isı haritası görüntüsü/dizisi konumu.")
    width: int = Field(gt=0, description="Katman genişliği (piksel).")
    height: int = Field(gt=0, description="Katman yüksekliği (piksel).")
    value_min: float = Field(default=0.0, description="Renk skalasının alt değeri.")
    value_max: float = Field(default=1.0, description="Renk skalasının üst değeri.")

    @model_validator(mode="after")
    def _range(self) -> "HeatmapLayer":
        if self.value_min >= self.value_max:
            raise ValueError("value_min < value_max olmalı")
        return self


class Region(ContractModel):
    x: int = Field(ge=0, description="Sol-üst X (seviye-0 piksel).")
    y: int = Field(ge=0, description="Sol-üst Y (seviye-0 piksel).")
    width: int = Field(gt=0, description="Genişlik (piksel).")
    height: int = Field(gt=0, description="Yükseklik (piksel).")
    score: Probability = Field(description="Bölgenin önem skoru (0-1).")


class ExplainResponse(ResearchOutput):
    slide_id: SlideId
    slide_width: int = Field(gt=0, description="Slayt genişliği (seviye-0 piksel).")
    slide_height: int = Field(gt=0, description="Slayt yüksekliği (seviye-0 piksel).")
    tile_size: int = Field(gt=0, description="Isı haritası döşeme boyutu (piksel).")
    layers: list[HeatmapLayer] = Field(min_length=1, description="Isı haritası katmanları.")
    regions: list[Region] = Field(default_factory=list, description="Vurgulanan bölgeler.")
    summary_text: str | None = Field(default=None, description="Kısa açıklama metni (varsa).")


# --------------------------------------------------------------------------------------
# Errors and health
# --------------------------------------------------------------------------------------
class FieldError(ContractModel):
    loc: list[str | int] = Field(description="Hatalı alanın yolu.")
    msg: str = Field(description="Hata açıklaması.")
    type: str = Field(description="Hata türü.")


class ProblemDetails(ContractModel):
    """Error body (RFC 7807 style), served as ``application/problem+json``."""

    type: str = Field(default="about:blank", description="Hata türünü tanımlayan URI.")
    title: str = Field(min_length=1, description="Kısa, insan okuyabilir başlık.")
    status: int = Field(ge=100, le=599, description="HTTP durum kodu.")
    detail: str | None = Field(default=None, description="Bu hataya özgü açıklama.")
    code: ErrorCode = Field(description="Makine tarafından okunabilir hata kodu.")
    instance: str | None = Field(default=None, description="Hatanın oluştuğu istek yolu.")
    errors: list[FieldError] | None = Field(default=None, description="Alan düzeyinde hatalar.")


class HealthResponse(VersionedMessage):
    status: Literal["ok"] = "ok"


REQUEST_MODELS: tuple[type[VersionedMessage], ...] = (
    MutationPredictionRequest,
    DrugRecommendationRequest,
    ReportRequest,
    SimulationRequest,
    ExplainRequest,
)
RESPONSE_MODELS: tuple[type[VersionedMessage], ...] = (
    MutationPrediction,
    DrugRecommendation,
    ReportResponse,
    SimulationResponse,
    ExplainResponse,
    HealthResponse,
)
NESTED_MODELS: tuple[type[ContractModel], ...] = (
    ConfidenceInterval,
    PatchRef,
    PatchAttention,
    GeneMutationScore,
    FusionAttention,
    DrugCandidate,
    Citation,
    SimulatedImage,
    HeatmapLayer,
    Region,
    FieldError,
    ProblemDetails,
)
ALL_MODELS: tuple[type[ContractModel], ...] = (*REQUEST_MODELS, *RESPONSE_MODELS, *NESTED_MODELS)
