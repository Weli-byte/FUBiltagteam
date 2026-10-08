"""Deterministic mock data that satisfies the contracts (for parallel development).

All values are random. They are NOT biological or clinical results; texts are marked ``[MOCK]``
and references are placeholders, never real citations.
"""

import random

from onkos.contracts.schemas import (
    Citation,
    ConfidenceInterval,
    DrugCandidate,
    DrugRecommendation,
    DrugRecommendationRequest,
    ExplainMethod,
    ExplainRequest,
    ExplainResponse,
    FusionAttention,
    Gene,
    GeneMutationScore,
    HeatmapLayer,
    MutationPrediction,
    MutationPredictionRequest,
    PatchAttention,
    Region,
    ReportRequest,
    ReportResponse,
    SimulatedImage,
    SimulationRequest,
    SimulationResponse,
)

# (name, DrugBank id [verify before real use], targets). Mock pool only.
_DRUG_POOL: tuple[tuple[str, str, list[str]], ...] = (
    ("Erlotinib", "DB00530", ["EGFR"]),
    ("Gefitinib", "DB00317", ["EGFR"]),
    ("Osimertinib", "DB09330", ["EGFR"]),
    ("Sotorasib", "DB15569", ["KRAS"]),
    ("Olaparib", "DB09074", ["PARP1", "PARP2"]),
    ("Carboplatin", "DB00958", ["DNA"]),
)


class MockFactory:
    """Builds contract-valid messages from a seed; same seed -> same data."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def slide_id(self) -> str:
        return f"slide_{self.rng.randrange(10**6):06d}"

    def gene_score(self, gene: Gene) -> GeneMutationScore:
        p = round(self.rng.random(), 4)
        u = round(self.rng.uniform(0.02, 0.15), 4)
        return GeneMutationScore(
            gene=gene,
            probability=p,
            uncertainty=u,
            ci=ConfidenceInterval(
                lower=round(max(0.0, p - 1.96 * u), 4), upper=round(min(1.0, p + 1.96 * u), 4)
            ),
        )

    def mutation_prediction(
        self, request: MutationPredictionRequest | None = None
    ) -> MutationPrediction:
        slide_id = request.slide_id if request else self.slide_id()
        genes = request.genes if request else list(Gene)
        image_weight = round(self.rng.uniform(0.3, 0.7), 3)
        return MutationPrediction(
            slide_id=slide_id,
            predictions=[self.gene_score(g) for g in genes],
            calibration_version="mock-isotonic-0.0",
            model_version="mock-omics-0.0",
            fusion_attention=FusionAttention(
                image_weight=image_weight,
                omics_weight=round(1 - image_weight, 3),
                top_patches=[
                    PatchAttention(x=256 * i, y=256 * (i % 3), weight=round(0.3 / (i + 1), 4))
                    for i in range(5)
                ],
            ),
        )

    def drug_recommendation(
        self, request: DrugRecommendationRequest | None = None
    ) -> DrugRecommendation:
        slide_id = request.slide_id if request else self.slide_id()
        top_k = min(request.top_k if request else 4, len(_DRUG_POOL))
        scores = sorted((round(self.rng.random(), 4) for _ in range(top_k)), reverse=True)
        pool = self.rng.sample(_DRUG_POOL, top_k)
        return DrugRecommendation(
            slide_id=slide_id,
            candidates=[
                DrugCandidate(
                    rank=i + 1,
                    drug_name=name,
                    drugbank_id=dbid,
                    compatibility_score=score,
                    target_proteins=targets,
                    rationale="[MOCK] Rastgele üretilmiş örnek gerekçe.",
                )
                for i, ((name, dbid, targets), score) in enumerate(zip(pool, scores, strict=True))
            ],
            model_version="mock-gnn-0.0",
            knowledge_base_version="mock-kb-0.0",
        )

    def report_request(self) -> ReportRequest:
        prediction = self.mutation_prediction()
        return ReportRequest(
            slide_id=prediction.slide_id,
            mutation_prediction=prediction,
            drug_recommendation=self.drug_recommendation(
                DrugRecommendationRequest(
                    slide_id=prediction.slide_id, mutations=prediction.predictions
                )
            ),
        )

    def report_response(self, request: ReportRequest) -> ReportResponse:
        return ReportResponse(
            slide_id=request.slide_id,
            report_text=(
                "[MOCK] Bu metin sözleşme testi için üretilmiştir; gerçek bir analiz değildir. "
                "Araştırma prototipi çıktısıdır, klinik karar için kullanılamaz."
            ),
            citations=[
                Citation(title="[MOCK] Örnek kaynak 1"),
                Citation(title="[MOCK] Örnek kaynak 2"),
            ],
            language=request.language,
            llm_model="mock-llm-0.0",
            retrieval_index_version="mock-index-0.0",
        )

    def simulation_response(self, request: SimulationRequest) -> SimulationResponse:
        return SimulationResponse(
            slide_id=request.slide_id,
            drug_name=request.drug_name,
            images=[
                SimulatedImage(role="before", uri="mock://sim/before.png", width=256, height=256),
                SimulatedImage(role="after", uri="mock://sim/after.png", width=256, height=256),
            ],
            seed=request.seed,
            model_version="mock-diffusion-0.0",
        )

    def explain_response(self, request: ExplainRequest) -> ExplainResponse:
        return ExplainResponse(
            slide_id=request.slide_id,
            slide_width=40960,
            slide_height=30720,
            tile_size=256,
            layers=[
                HeatmapLayer(
                    name=f"{request.method.value} (mock)",
                    method=request.method,
                    uri=f"mock://explain/{request.method.value}.png",
                    width=160,
                    height=120,
                )
            ],
            regions=[
                Region(x=2048, y=1024, width=512, height=512, score=round(self.rng.random(), 4))
            ],
            summary_text="[MOCK] Örnek açıklama metni.",
        )

    def explain_request(self) -> ExplainRequest:
        return ExplainRequest(slide_id=self.slide_id(), method=ExplainMethod.GRADCAM)

    def simulation_request(self) -> SimulationRequest:
        return SimulationRequest(slide_id=self.slide_id(), drug_name="Erlotinib")

    def mutation_request(self) -> MutationPredictionRequest:
        return MutationPredictionRequest(
            slide_id=self.slide_id(), patch_features_uri="mock://features.h5"
        )

    def drug_request(self) -> DrugRecommendationRequest:
        prediction = self.mutation_prediction()
        return DrugRecommendationRequest(
            slide_id=prediction.slide_id, mutations=prediction.predictions
        )
