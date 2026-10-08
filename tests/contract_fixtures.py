"""Hand-written, fixed test payloads for the contract tests.

These exist ONLY to exercise schema validation. They are not model output, not served by any
endpoint, and not meant as realistic data. Realistic payloads come from the real pipelines.
"""

import math
from pathlib import Path
from typing import Any

from onkos.contracts import SCHEMA_VERSION


def gene_score(
    gene: str = "TP53", probability: float = 0.5, uncertainty: float = 0.05
) -> dict[str, Any]:
    return {
        "gene": gene,
        "probability": probability,
        "uncertainty": uncertainty,
        "ci": {
            "lower": max(0.0, probability - 0.1),
            "upper": min(1.0, probability + 0.1),
        },
    }


def mutation_prediction(
    slide_id: str = "slide_0001", genes: tuple[str, ...] = ("TP53", "EGFR")
) -> dict[str, Any]:
    return {
        "slide_id": slide_id,
        "predictions": [gene_score(g, 0.4 + 0.1 * i) for i, g in enumerate(genes)],
        "calibration_version": "test-calibration",
        "model_version": "test-model",
    }


def drug_recommendation(slide_id: str = "slide_0001") -> dict[str, Any]:
    return {
        "slide_id": slide_id,
        "candidates": [
            {"rank": 1, "drug_name": "drug-a", "compatibility_score": 0.9},
            {"rank": 2, "drug_name": "drug-b", "compatibility_score": 0.6},
        ],
        "model_version": "test-model",
    }


def report_request(slide_id: str = "slide_0001") -> dict[str, Any]:
    return {
        "slide_id": slide_id,
        "mutation_prediction": mutation_prediction(slide_id),
        "drug_recommendation": drug_recommendation(slide_id),
    }


def simulation_response(slide_id: str = "slide_0001") -> dict[str, Any]:
    return {
        "slide_id": slide_id,
        "drug_name": "drug-a",
        "images": [
            {"role": "before", "uri": "test://before.png", "width": 256, "height": 256},
            {"role": "after", "uri": "test://after.png", "width": 256, "height": 256},
        ],
        "seed": 0,
        "model_version": "test-model",
    }


def explain_response(slide_id: str = "slide_0001") -> dict[str, Any]:
    return {
        "slide_id": slide_id,
        "slide_width": 4096,
        "slide_height": 4096,
        "tile_size": 256,
        "layers": [
            {
                "name": "layer",
                "method": "gradcam",
                "uri": "test://layer.png",
                "width": 16,
                "height": 16,
            }
        ],
    }


def request_dicts(slide_id: str = "slide_0001") -> dict[str, dict[str, Any]]:
    return {
        "MutationPredictionRequest": {"slide_id": slide_id, "patch_features_uri": "test://f.h5"},
        "DrugRecommendationRequest": {
            "slide_id": slide_id,
            "mutations": [gene_score()],
        },
        "ReportRequest": report_request(slide_id),
        "SimulationRequest": {"slide_id": slide_id, "drug_name": "drug-a"},
        "ExplainRequest": {"slide_id": slide_id, "method": "gradcam"},
    }


def write_test_patch_features(
    path: Path,
    slide_id: str = "slide_0001",
    n_patches: int = 16,
    feature_dim: int = 8,
    with_attention: bool = True,
    with_slide_embedding: bool = True,
) -> Path:
    """Write a tiny deterministic HDF5 file that follows the contract (for validator tests)."""
    import h5py
    import numpy as np

    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=np.ones((n_patches, feature_dim), dtype="float16"))
        grid = int(math.ceil(math.sqrt(n_patches)))
        idx = np.arange(n_patches)
        f.create_dataset(
            "coords",
            data=np.stack([(idx % grid) * 256, (idx // grid) * 256], axis=1).astype("int32"),
        )
        if with_attention:
            f.create_dataset("attention", data=np.full(n_patches, 1.0 / n_patches, dtype="float32"))
        if with_slide_embedding:
            f.create_dataset("slide_embedding", data=np.ones(feature_dim, dtype="float32"))
        f.attrs.update(
            schema_version=SCHEMA_VERSION,
            slide_id=slide_id,
            magnification=20.0,
            encoder_name="test-encoder",
            encoder_version="0",
            patch_size_px=256,
        )
    return path
