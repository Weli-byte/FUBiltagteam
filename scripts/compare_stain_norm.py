"""Measure the effect of stain normalization (mini sprint 1.8).

Bu dosya ne işe yarar? Örnek paketteki patch'ler üzerinde üç durumu karşılaştırır: normalizasyon
yok / Macenko / Reinhard. İki ölçüm yapar: (1) merkezler arası LAB renk istatistiği farkı
(normalizasyon sonrası düşmeli), (2) "merkez ayırt edici sınıflandırıcı": yalnızca renk
histogramından hangi merkezden geldiğini tahmin eden küçük bir lojistik regresyon (3 seed);
doğruluğu normalizasyon sonrası şansa yaklaşmalı. Ayrıca önce/sonra görsel ızgarası üretir.
Sonuçlar JSON + Markdown tabloya yazılır; yorum ``docs/histo/boya_normalizasyonu.md``'de.

Centre labels: ``patches.csv`` needs a ``centre`` column, or the pack's ``source`` column is
used as a coarse proxy (CAMELYON vs TCGA); the report says which one was used.

Usage:
    uv run --group histo python scripts/compare_stain_norm.py
    uv run --group histo python scripts/compare_stain_norm.py compare.pack_dir=outputs/demo/patch_v0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from models.histo.config import load_histo_config, section
from models.histo.patches.sample_pack import read_rows
from models.histo.stain.normalizer import (
    MacenkoNormalizer,
    ReinhardNormalizer,
    StainNormalizer,
    lab_stats,
    make_normalizer,
)

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("compare_stain_norm")


def colour_histogram(rgb: NDArray[np.uint8], bins: int = 8) -> NDArray[np.float64]:
    hist = [np.histogram(rgb[..., c], bins=bins, range=(0, 255))[0] for c in range(3)]
    vec = np.concatenate(hist).astype(np.float64)
    return np.asarray(vec / vec.sum(), dtype=np.float64)


def read_rgb(path: Path) -> NDArray[np.uint8]:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def centre_separability(
    features: NDArray[np.float64], centres: NDArray[np.int64], seed: int
) -> float:
    """Held-out accuracy of a logistic regression that predicts the centre from colour only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    x_tr, x_te, y_tr, y_te = train_test_split(
        features, centres, test_size=0.3, random_state=seed, stratify=centres
    )
    model = LogisticRegression(max_iter=2000, random_state=seed).fit(x_tr, y_tr)
    return float(model.score(x_te, y_te))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="stain")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import cv2

    cfg = load_histo_config(args.config, args.overrides)
    pack_dir = ROOT / str(cfg.compare.pack_dir)
    out_dir = ROOT / str(cfg.compare.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_rows(pack_dir / "patches.csv") if int(r["label"]) in (0, 1)]
    centre_key = (
        "centre"
        if rows and "centre" in rows[0] and any(r.get("centre") for r in rows)
        else "source"
    )
    rng = np.random.default_rng(int(cfg.compare.seeds[0]))
    by_centre: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_centre.setdefault(str(r[centre_key]), []).append(r)
    if len(by_centre) < 2:
        print(
            f"only one centre ({list(by_centre)}) in the pack: colour-shift metrics need >= 2; "
            "writing stats only."
        )
    n_per = int(cfg.compare.n_per_centre)
    selected: list[dict[str, Any]] = []
    for _, group in sorted(by_centre.items()):
        idx = rng.choice(len(group), size=min(n_per, len(group)), replace=False)
        selected += [group[i] for i in idx]
    images = [read_rgb(pack_dir / r["image"]) for r in selected]
    centres = np.array(
        [sorted(by_centre).index(str(r[centre_key])) for r in selected], dtype=np.int64
    )

    ref_path = ROOT / str(cfg.normalizer.reference_image)
    reference = read_rgb(ref_path) if ref_path.exists() else images[0]
    if not ref_path.exists():
        log.warning("reference image %s not found; using the first sampled patch", ref_path)
    methods: dict[str, StainNormalizer] = {
        "none": make_normalizer("none"),
        "macenko": MacenkoNormalizer(**section(cfg.normalizer, "macenko")).fit(reference),
        "reinhard": ReinhardNormalizer().fit(reference),
    }
    results: dict[str, Any] = {
        "centre_key": centre_key,
        "centres": sorted(by_centre),
        "n_patches": len(images),
        "methods": {},
    }
    normalised: dict[str, list[NDArray[np.uint8]]] = {}
    for name, norm in methods.items():
        outputs: list[NDArray[np.uint8]] = []
        failures = 0
        for img in images:
            try:
                outputs.append(norm.transform(img))
            except ValueError:  # e.g. too little tissue for Macenko: keep the original
                failures += 1
                outputs.append(img)
        normalised[name] = outputs
        stats = np.stack([lab_stats(o) for o in outputs])
        per_centre = np.stack([stats[centres == c].mean(axis=0) for c in range(len(by_centre))])
        spread = (
            float(np.abs(per_centre[:, :3] - per_centre[:, :3].mean(axis=0)).mean())
            if len(by_centre) > 1
            else float("nan")
        )
        features = np.stack([colour_histogram(o) for o in outputs])
        accuracies = (
            [centre_separability(features, centres, int(s)) for s in cfg.compare.seeds]
            if len(by_centre) > 1
            else []
        )
        results["methods"][name] = {
            "lab_mean_spread_between_centres": spread,
            "centre_classifier_accuracy_mean": float(np.mean(accuracies)) if accuracies else None,
            "centre_classifier_accuracy_std": float(np.std(accuracies)) if accuracies else None,
            "chance_level": (
                float(np.bincount(centres).max() / len(centres)) if len(centres) else None
            ),
            "transform_failures": failures,
        }
    (out_dir / "stain_comparison.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    k = int(cfg.compare.grid_examples)
    pick = rng.choice(len(images), size=min(k, len(images)), replace=False)
    rows_img = [np.concatenate([normalised[m][i] for m in methods], axis=1) for i in pick]
    cv2.imwrite(
        str(out_dir / "before_after_grid.png"),
        cv2.cvtColor(np.concatenate(rows_img, axis=0), cv2.COLOR_RGB2BGR),
    )
    lines = [
        "| yöntem | merkezler arası LAB farkı | merkez sınıflandırıcı doğruluğu (ort ± sd) "
        "| şans | hata |",
        "|---|---|---|---|---|",
    ]
    for name, m in results["methods"].items():
        acc = (
            f"{m['centre_classifier_accuracy_mean']:.3f} "
            f"± {m['centre_classifier_accuracy_std']:.3f}"
            if m["centre_classifier_accuracy_mean"] is not None
            else "-"
        )
        chance = f"{m['chance_level']:.3f}" if m["chance_level"] is not None else "-"
        lines.append(
            f"| {name} | {m['lab_mean_spread_between_centres']:.3f} | {acc} | {chance} "
            f"| {m['transform_failures']} |"
        )
    (out_dir / "stain_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"outputs -> {out_dir} (grid: columns = {list(methods)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
