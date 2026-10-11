# Veri Envanteri (TCGA / GDC)

> `scripts/download_gdc.py` tarafından üretilir. Yalnızca toplu sayılar içerir.

- GDC veri sürümü: `Data Release 46.0 - August 10, 2026`
- Projeler: TCGA-LUAD, TCGA-BRCA, TCGA-COAD
- `has_rnaseq`: birincil tümör (Primary Tumor) RNA-seq dosyası olan hasta
- `rnaseq_and_maf`, `all_three`: kesişimler (RNA-seq + MAF; + tanı slaydı)

| Proje | patients | has_clinical | has_rnaseq | has_maf | has_slide | rnaseq_and_maf | all_three |
|---|---|---|---|---|---|---|---|
| TCGA-BRCA | 1098 | 1098 | 1095 | 969 | 1062 | 966 | 936 |
| TCGA-COAD | 461 | 461 | 458 | 428 | 451 | 427 | 418 |
| TCGA-LUAD | 585 | 585 | 517 | 559 | 478 | 507 | 465 |
| TOTAL | 2144 | 2144 | 2070 | 1956 | 1991 | 1900 | 1819 |
