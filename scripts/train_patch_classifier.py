"""ViT-S/16 vs ResNet-50 patch baseline over several seeds (mini sprint 1.9).

Bu dosya ne işe yarar? Aynı veri, aynı slayt bazlı bölme ve aynı tarifle her modeli her seed
için eğitir, doğrulama kümesinde AUROC/AUPRC/F1/doğruluk (bootstrap güven aralığı) hesaplar,
seed ortalamalarını tablo yapar, ROC/PR eğrileri ve öğrenme eğrilerini çizer ve bir model kartı
taslağı yazar. Resmi test bölmesi yalnızca ``eval.final_evaluation=true`` ile açılır.

Usage:
    uv run --group histo python scripts/train_patch_classifier.py   # configs/histo/patch_classifier
    uv run --group histo python scripts/train_patch_classifier.py train.epochs=1 \
        train.seeds=[0] train.models=[tiny_cnn] train.pretrained=false
    uv run --group histo python scripts/train_patch_classifier.py \
        data.pack_dir=outputs/demo/patch_v0 out_dir=outputs/demo_run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from models.histo.config import config_hash, load_histo_config, section, to_plain
from models.histo.patch_classifier.data import (
    label_counts,
    load_labelled_rows,
    make_loader,
    rows_for_split,
    split_rows_by_slide,
)
from models.histo.patch_classifier.metrics import all_metrics, summarize_seeds
from models.histo.patch_classifier.train import (
    TrainConfig,
    load_best,
    predict,
    predict_tta,
    resolve_device,
    train_classifier,
)
from models.histo.stain.augment import AugmentConfig

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("train_patch_classifier")


def plot_curves(y: Any, p: Any, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    fp = np.cumsum(1 - y[order])
    tpr, fpr = tp / max(1, y.sum()), fp / max(1, (1 - y).sum())
    precision = tp / np.arange(1, len(y) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=100)
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(xlabel="FPR", ylabel="TPR", title=f"ROC: {title}")
    axes[1].plot(tpr, precision)
    axes[1].set(xlabel="recall", ylabel="precision", title=f"PR: {title}", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_history(histories: dict[str, list[dict[str, Any]]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=100)
    for name, hist in histories.items():
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["train_loss"] for h in hist], label=f"{name} train")
        axes[0].plot(epochs, [h["val_loss"] for h in hist], "--", label=f"{name} val")
        axes[1].plot(epochs, [h["val_auroc"] for h in hist], label=name)
    axes[0].set(xlabel="epoch", ylabel="loss", title="öğrenme eğrisi")
    axes[1].set(xlabel="epoch", ylabel="val AUROC", title="doğrulama AUROC")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="patch_classifier")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_histo_config(args.config, args.overrides)
    pack_dir = ROOT / str(cfg.data.pack_dir)
    out_dir = ROOT / str(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = bool(cfg.eval.final_evaluation)

    rows = load_labelled_rows(pack_dir)
    train_all = rows_for_split(rows, "train")
    train_rows, val_rows = split_rows_by_slide(
        train_all, float(cfg.data.val_fraction), int(cfg.data.split_seed)
    )
    test_rows = rows_for_split(rows, "test", final_evaluation=True) if final else []
    print(
        f"train {label_counts(train_rows)}  val {label_counts(val_rows)}  "
        f"test {'LOCKED' if not final else label_counts(test_rows)}"
    )

    augment = AugmentConfig(**section(cfg, "augment"))
    train_section = section(cfg, "train")
    models = [str(m) for m in train_section.pop("models")]
    seeds = [int(s) for s in train_section.pop("seeds")]
    use_mlflow = bool(cfg.tracking.mlflow)
    results: dict[str, Any] = {
        "config_hash": config_hash(cfg),
        "models": {},
        "test_locked": not final,
    }
    histories: dict[str, list[dict[str, Any]]] = {}
    for model_name in models:
        per_seed: list[dict[str, Any]] = []
        for seed in seeds:
            tcfg = TrainConfig(
                model=model_name,
                seed=seed,
                image_px=int(cfg.data.image_px),
                augment=augment,
                **train_section,
            )
            run_dir = out_dir / f"{model_name}-seed{seed}"
            logger = None
            run_ctx: Any = None
            if use_mlflow:
                import mlflow

                from onkos.common.gitinfo import git_commit_hash

                mlflow.set_experiment(str(cfg.tracking.experiment))
                run_ctx = mlflow.start_run(run_name=f"histo-patch_{model_name}-{seed}")
                mlflow.log_params({k: str(v) for k, v in tcfg.to_dict().items() if k != "augment"})
                mlflow.set_tag("git_commit", git_commit_hash() or "unknown")

                def logger(key: str, value: float, step: int) -> None:
                    mlflow.log_metric(key, value, step=step)

            outcome = train_classifier(tcfg, pack_dir, train_rows, val_rows, run_dir, logger)
            metrics = all_metrics(outcome.val_y, outcome.val_p, int(cfg.eval.n_boot), seed)
            entry: dict[str, Any] = {
                "seed": seed,
                "best_epoch": outcome.best_epoch,
                "val": {
                    k: {"point": v.point, "lower": v.lower, "upper": v.upper}
                    for k, v in metrics.items()
                },
                "history": [h.__dict__ for h in outcome.history],
            }
            plot_curves(
                outcome.val_y,
                outcome.val_p,
                run_dir / "val_curves.png",
                f"{model_name} seed {seed} (val)",
            )
            if final and test_rows:
                device = resolve_device(tcfg.device)
                model = load_best(tcfg, run_dir, device)
                loader = make_loader(
                    pack_dir,
                    test_rows,
                    tcfg.batch_size,
                    False,
                    seed,
                    tcfg.num_workers,
                    None,
                    tcfg.image_px,
                )
                if tcfg.tta:
                    y, p, _ = predict_tta(model, loader, device, tcfg.amp)
                else:
                    y, p, _, _ = predict(model, loader, device, tcfg.amp)
                test_metrics = all_metrics(y, p, int(cfg.eval.n_boot), seed)
                entry["test"] = {
                    k: {"point": v.point, "lower": v.lower, "upper": v.upper}
                    for k, v in test_metrics.items()
                }
                plot_curves(y, p, run_dir / "test_curves.png", f"{model_name} seed {seed} (TEST)")
            if use_mlflow and run_ctx is not None:
                import mlflow

                mlflow.log_metrics({f"val_{k}": v.point for k, v in metrics.items()})
                mlflow.end_run()
            per_seed.append(entry)
            histories[f"{model_name}-s{seed}"] = entry["history"]
            print(
                f"{model_name} seed {seed}: "
                + ", ".join(f"{k}={v.fmt()}" for k, v in metrics.items())
            )
        summary = {
            k: summarize_seeds([e["val"][k]["point"] for e in per_seed])
            for k in ("auroc", "auprc", "f1", "accuracy")
        }
        results["models"][model_name] = {"seeds": per_seed, "val_summary": summary}
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8"
    )
    plot_history(histories, out_dir / "learning_curves.png")

    lines = [
        "| model | AUROC (ort, %95 GA) | AUPRC | F1 | doğruluk | seed sayısı |",
        "|---|---|---|---|---|---|",
    ]
    for name, res in results["models"].items():
        s = res["val_summary"]
        cells = [
            f"{s[k]['mean']:.3f} [{s[k]['ci_lower']:.3f}, {s[k]['ci_upper']:.3f}]"
            for k in ("auroc", "auprc", "f1", "accuracy")
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {int(s['auroc']['n'])} |")
    table = "\n".join(lines)
    (out_dir / "results_table.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"outputs -> {out_dir}; config: {json.dumps(to_plain(cfg)['train'])}")
    print(
        "Not: sonuçlar doğrulama kümesine aittir; "
        "test bölmesi yalnızca final değerlendirmede açılır."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
