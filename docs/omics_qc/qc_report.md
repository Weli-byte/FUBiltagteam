# Omics QC Raporu

- Örnek sayısı: 2070
- Uç değer eşiği: |robust z| > 4.0 (medyan ve MAD tabanlı)
- Bayrak (flag) taşıyan örnekler **silinmedi**; karar `--drop-flagged` ile verilir.

## Bayraklı örnek sayıları (proje bazında)

| proje | samples | flag_missing | flag_library_size | flag_detected_genes | flag_low_correlation | flag_any |
|---|---|---|---|---|---|---|
| TCGA-BRCA | 1095 | 0 | 0 | 7 | 10 | 12 |
| TCGA-COAD | 458 | 0 | 18 | 3 | 20 | 34 |
| TCGA-LUAD | 517 | 0 | 0 | 0 | 7 | 7 |

## Batch etkisi (silhouette; -1..1, yüksek = daha sıkı gruplaşma)

- Kanser türüne göre: 0.347
- Doku kaynak merkezine (TSS) göre, proje içinde ortalama: -0.169
  - TCGA-BRCA: -0.226
  - TCGA-COAD: -0.106
  - TCGA-LUAD: -0.174

> Yorum kuralı: kanser türü skoru TSS skorundan belirgin yüksekse biyolojik sinyal batch'ten baskındır. Sayılar olduğu gibi raporlanır; eşik uydurulmamıştır.

## Şekiller

- `library_size_hist.png`
- `pca_project_vs_tss.png`
- `umap_project_vs_tss.png`
- `sample_correlation.png`
