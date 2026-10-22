"""Results tables, trivial-baseline comparison, figures and the Markdown report (mini sprint 1.8).

The rule-based verdicts below use thresholds chosen by the authors, NOT tuned on data; they are a
reading aid. The best model per gene is picked on the same out-of-fold predictions it is reported
on, so its numbers are slightly optimistic; the final unbiased estimate belongs to the locked test
set (sprint 2, final evaluation).
"""

from pathlib import Path
from typing import Any

import pandas as pd

from models.omics import metrics as M
from models.omics.baselines import POOLED, TRIVIAL, BaselineConfig

EASY_AUROC = 0.80
HARD_AUROC = 0.65
SUSPICIOUS_AUROC = 0.95


def best_pooled(results: pd.DataFrame) -> pd.DataFrame:
    """Best omics model per gene in the pooled scope, by AUPRC (point estimate)."""
    ok = results[
        (results["status"] == "ok") & (results["scope"] == POOLED) & (results["model"] != TRIVIAL)
    ]
    if ok.empty:
        return ok
    return ok.sort_values("auprc", ascending=False).groupby("gene", sort=False).head(1)


def trivial_comparison(
    results: pd.DataFrame, oof: pd.DataFrame, cfg: BaselineConfig
) -> pd.DataFrame:
    """Paired bootstrap difference between the best omics model and the cancer-type-only baseline."""
    rows: list[dict[str, Any]] = []
    for _, best in best_pooled(results).iterrows():
        gene = best["gene"]
        a = oof[(oof["gene"] == gene) & (oof["scope"] == POOLED) & (oof["model"] == best["model"])]
        b = oof[(oof["gene"] == gene) & (oof["scope"] == POOLED) & (oof["model"] == TRIVIAL)]
        if a.empty or b.empty:
            continue
        merged = a.merge(b, on="patient_barcode", suffixes=("", "_ref"))
        y, pa, pb = (
            merged["y"].to_numpy(),
            merged["proba"].to_numpy(),
            merged["proba_ref"].to_numpy(),
        )
        d_roc = M.paired_bootstrap_delta(y, pa, pb, M.auroc, cfg.n_boot, cfg.seed)
        d_prc = M.paired_bootstrap_delta(y, pa, pb, M.auprc, cfg.n_boot, cfg.seed)
        rows.append(
            {
                "gene": gene,
                "best_model": best["model"],
                "auroc_model": best["auroc"],
                "auroc_trivial": M.auroc(y, pb),
                "d_auroc": d_roc["delta"],
                "d_auroc_lo": d_roc["lo"],
                "d_auroc_hi": d_roc["hi"],
                "auprc_model": best["auprc"],
                "auprc_trivial": M.auprc(y, pb),
                "prevalence": best["prevalence"],
                "d_auprc": d_prc["delta"],
                "d_auprc_lo": d_prc["lo"],
                "d_auprc_hi": d_prc["hi"],
            }
        )
    return pd.DataFrame(rows)


def verdict(row: pd.Series) -> str:
    """Reading aid: does omics add information beyond cancer type, and how hard is the gene?"""
    if row["d_auroc_lo"] > 0 and row["d_auprc_lo"] > 0:
        added = "omics, kanser türünden fazla bilgi veriyor"
    elif row["d_auroc_hi"] < 0 or row["d_auprc_hi"] < 0:
        added = "omics, kanser türü tabanından KÖTÜ (şüpheli, kontrol et)"
    else:
        added = "omics katkısı belirsiz (güven aralığı 0'ı içeriyor)"
    auroc = row["auroc_model"]
    level = "kolay" if auroc >= EASY_AUROC else "zor" if auroc < HARD_AUROC else "orta"
    return f"{level}; {added}"


def suspicious(results: pd.DataFrame) -> list[str]:
    ok = results[results["status"] == "ok"]
    flagged = ok[ok["auroc"] > SUSPICIOUS_AUROC]
    return [
        f"{r['scope']}/{r['gene']}/{r['model']}: AUROC {r['auroc']:.3f} > {SUSPICIOUS_AUROC}; "
        "şişirilmiş metrik şüphesi: sızıntıyı (etiket kaynağı, özellik kaynağı, bölme) araştır"
        for _, r in flagged.iterrows()
    ]


def _ci(row: pd.Series, name: str) -> str:
    return f"{row[name]:.3f} [{row[name + '_lo']:.3f}, {row[name + '_hi']:.3f}]"


def results_markdown(
    results: pd.DataFrame, comparison: pd.DataFrame, cfg: BaselineConfig, run: str
) -> str:
    ok = results[results["status"] == "ok"]
    lines = [
        f"# Baseline Omics Modelleri: {run}",
        "",
        "> `scripts/train_baselines.py` tarafından üretilir. Yalnızca toplu sayılar içerir.",
        "",
        "## Protokol",
        "",
        "- Yalnızca **geliştirme havuzu** (train+val) kullanıldı; **test kümesine dokunulmadı**.",
        f"- Dış döngü: 5 katlı CV (`cv_fold`). Ön işleme (gen filtresi, {cfg.top_genes} yüksek varyanslı gen "
        f"[{cfg.feature_version}], z-skor) her dış katmanda **yalnızca eğitim satırlarından** öğrenildi.",
        f"- İç döngü: {cfg.inner_folds} katlı CV, AUPRC ile hiperparametre seçimi; F1 eşiği yalnızca eğitim "
        "tarafındaki iç CV tahminlerinden seçildi (raporlanan eşik, katman eşiklerinin medyanıdır).",
        "- Sınıf dengesizliği: sınıf ağırlıkları (her katmanın içinde); yapay örnekleme yok.",
        f"- Güven aralıkları: %95, {cfg.n_boot} bootstrap örneği (hasta düzeyinde yeniden örnekleme).",
        f"- Pozitif sayısı < {cfg.min_positives} olan kapsamlar eğitilmedi, aşağıda 'atlandı' olarak listelendi.",
        "- Tüm metrikler OOF (katman dışı) tahminlerinden hesaplanır. Seçilen en iyi model aynı OOF "
        "tahminlerinde seçildiği için sayıları hafif iyimserdir; nihai tarafsız tahmin kilitli test "
        "kümesindeki final değerlendirmeden gelir.",
        "",
        "## Birleşik (pan-cancer) model sonuçları",
        "",
        "| gen | model | hasta | pozitif | prevalans | AUROC [%95 GA] | AUPRC [%95 GA] | F1 | Brier | ECE |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    pooled = ok[ok["scope"] == POOLED].sort_values(["gene", "auprc"], ascending=[True, False])
    for _, r in pooled.iterrows():
        lines.append(
            f"| {r['gene']} | {r['model']}{' (' + r['backend'] + ')' if r['backend'] else ''} | {int(r['n'])} "
            f"| {int(r['positives'])} | {r['prevalence']:.3f} | {_ci(r, 'auroc')} | {_ci(r, 'auprc')} "
            f"| {r['f1']:.3f} | {r['brier']:.3f} | {r['ece']:.3f} |"
        )
    lines += [
        "",
        "## Kanser türüne özel eğitim ve birleşik modelin tür bazında sonucu",
        "",
        "`<tür>` = o türe özel eğitilmiş model; `pooled@<tür>` = birleşik modelin o türdeki hastalarda sonucu.",
        "",
        "| gen | kapsam | model | hasta | pozitif | AUROC [%95 GA] | AUPRC [%95 GA] |",
        "|---|---|---|---|---|---|---|",
    ]
    per = ok[ok["scope"] != POOLED].sort_values(
        ["gene", "scope", "auprc"], ascending=[True, True, False]
    )
    per = per.groupby(["gene", "scope"], sort=False).head(1)
    for _, r in per.iterrows():
        lines.append(
            f"| {r['gene']} | {r['scope']} | {r['model']} | {int(r['n'])} | {int(r['positives'])} "
            f"| {_ci(r, 'auroc')} | {_ci(r, 'auprc')} |"
        )
    lines += [
        "",
        "## Trivial taban (yalnızca kanser türü) ile karşılaştırma",
        "",
        "Fark = en iyi omics modeli − kanser-türü-tabanı, eşleştirilmiş bootstrap %95 GA.",
        "",
        "| gen | en iyi model | AUROC model / taban | ΔAUROC [GA] | AUPRC model / taban (prevalans) | ΔAUPRC [GA] | yorum (okuma yardımcısı) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in comparison.iterrows():
        lines.append(
            f"| {r['gene']} | {r['best_model']} | {r['auroc_model']:.3f} / {r['auroc_trivial']:.3f} "
            f"| {r['d_auroc']:+.3f} [{r['d_auroc_lo']:+.3f}, {r['d_auroc_hi']:+.3f}] "
            f"| {r['auprc_model']:.3f} / {r['auprc_trivial']:.3f} ({r['prevalence']:.3f}) "
            f"| {r['d_auprc']:+.3f} [{r['d_auprc_lo']:+.3f}, {r['d_auprc_hi']:+.3f}] | {verdict(r)} |"
        )
    lines += [
        "",
        f"> Yorum eşikleri (yazarların seçimi, veriye göre ayarlanmadı): AUROC ≥ {EASY_AUROC} 'kolay', "
        f"< {HARD_AUROC} 'zor'. Bunlar otomatik okuma yardımcısıdır; biyolojik yorumu ekip yapar.",
        "",
    ]
    skipped = results[results["status"] != "ok"]
    lines += ["## Atlanan kapsamlar (sessizce geçilmedi)", ""]
    lines += [
        f"- {r['scope']} / {r['gene']} / {r['model']}: {r['status']}" for _, r in skipped.iterrows()
    ] or ["- (yok)"]
    lines += ["", "## Uyarılar", ""]
    warnings = suspicious(results)
    lines += [f"- {w}" for w in warnings] or ["- Şişirilmiş metrik şüphesi (AUROC > 0.95) yok."]
    lines += [
        "",
        "## Şekiller",
        "",
        f"- `{run}_roc.png`, `{run}_pr.png`, `{run}_calibration.png`",
        "",
    ]
    return "\n".join(lines)


def save_figures(oof: pd.DataFrame, results: pd.DataFrame, out_dir: Path, run: str) -> list[Path]:
    """ROC, precision-recall and calibration plots of the best pooled model vs the trivial baseline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    out_dir.mkdir(parents=True, exist_ok=True)
    best = best_pooled(results)
    genes = list(best["gene"])
    written: list[Path] = []
    for kind in ("roc", "pr", "calibration"):
        fig, axes = plt.subplots(
            1, max(len(genes), 1), figsize=(3.4 * max(len(genes), 1), 3.4), squeeze=False
        )
        for ax, (_, row) in zip(axes[0], best.iterrows(), strict=False):
            gene = row["gene"]
            for model, style in ((row["model"], "-"), (TRIVIAL, "--")):
                sub = oof[
                    (oof["gene"] == gene) & (oof["scope"] == POOLED) & (oof["model"] == model)
                ]
                if sub.empty:
                    continue
                y, p = sub["y"].to_numpy(), sub["proba"].to_numpy()
                if kind == "roc":
                    fpr, tpr, _ = roc_curve(y, p)
                    ax.plot(fpr, tpr, style, label=model)
                    ax.plot([0, 1], [0, 1], ":", color="grey")
                elif kind == "pr":
                    prec, rec, _ = precision_recall_curve(y, p)
                    ax.plot(rec, prec, style, label=model)
                    ax.axhline(M.prevalence(y), ls=":", color="grey")
                else:
                    curve = M.reliability_curve(y, p)
                    ax.plot(curve["mean_pred"], curve["frac_pos"], "o" + style, label=model)
                    ax.plot([0, 1], [0, 1], ":", color="grey")
            ax.set_title(gene)
            ax.legend(fontsize=7)
        path = out_dir / f"{run}_{kind}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        written.append(path)
    return written


def build_report(
    results: pd.DataFrame, oof: pd.DataFrame, cfg: BaselineConfig, run: str, report_dir: Path
) -> list[Path]:
    """Write ``<run>_rapor.md`` and the figures into ``report_dir``."""
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison = trivial_comparison(results, oof, cfg)
    figures = save_figures(oof, results, report_dir, run) if not oof.empty else []
    path = report_dir / f"{run}_rapor.md"
    if path.exists():
        raise FileExistsError(f"{path} already exists; not overwriting")
    path.write_text(results_markdown(results, comparison, cfg, run), encoding="utf-8")
    return [path, *figures]
