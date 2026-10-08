"""Which fields each consuming module relies on (consumer-driven contract check).

If someone renames or removes a field listed here, ``tests/test_contract_consumers.py`` fails and
names the consumer, so the PR must update (and get approval from) that module's owner.
Edit this file through a PR approved by the consumer's owner.

Paths use ``.`` for nesting and ``[]`` for list items, e.g. ``predictions[].ci.lower``.
"""

CONSUMER_EXPECTATIONS: dict[str, dict[str, list[str]]] = {
    "frontend (Yusuf)": {
        "MutationPrediction": [
            "slide_id",
            "predictions[].gene",
            "predictions[].probability",
            "predictions[].uncertainty",
            "predictions[].ci.lower",
            "predictions[].ci.upper",
            "calibration_version",
            "disclaimer",
        ],
        "DrugRecommendation": [
            "candidates[].rank",
            "candidates[].drug_name",
            "candidates[].compatibility_score",
            "disclaimer",
        ],
        "ReportResponse": ["report_text", "citations[].title", "disclaimer"],
        "SimulationResponse": ["images[].role", "images[].uri", "is_conceptual", "limitations"],
        "ExplainResponse": [
            "slide_width",
            "slide_height",
            "tile_size",
            "layers[].uri",
            "layers[].value_min",
            "layers[].value_max",
            "regions[].score",
        ],
        "ProblemDetails": ["title", "status", "code", "detail"],
    },
    "backend (Idil)": {
        "MutationPredictionRequest": ["slide_id", "patch_features_uri", "genes"],
        "DrugRecommendationRequest": ["slide_id", "mutations", "top_k"],
        "ReportRequest": ["slide_id", "mutation_prediction", "drug_recommendation", "language"],
        "SimulationRequest": ["slide_id", "drug_name", "seed", "num_inference_steps"],
        "ExplainRequest": ["slide_id", "method", "target_gene"],
    },
    "gnn (Idil)": {
        "DrugRecommendationRequest": ["mutations[].gene", "mutations[].probability"],
        "DrugRecommendation": ["candidates[].drugbank_id", "candidates[].target_proteins"],
    },
    "xai (Yusuf)": {
        "MutationPrediction": [
            "fusion_attention.top_patches[].x",
            "fusion_attention.top_patches[].y",
        ],
        "ExplainRequest": ["method", "target_gene"],
    },
}
