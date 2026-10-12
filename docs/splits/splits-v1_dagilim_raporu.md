# Bölme Dağılım Raporu (splits-v1)

> `scripts/make_splits.py` tarafından üretilir. Yalnızca toplu sayılar içerir.

- Seed: 42; oranlar (train/val/test): (0.7, 0.15, 0.15); CV katmanı: 5
- Hasta sayısı: 2144
- Uyarı eşikleri: alt kümede < 5 pozitif; oran genelden > 3 puan sapma

## Hasta sayısı (proje x küme)

| proje | train | val | test | toplam |
|---|---|---|---|---|
| TCGA-BRCA | 771 | 165 | 162 | 1098 |
| TCGA-COAD | 323 | 69 | 69 | 461 |
| TCGA-LUAD | 409 | 88 | 88 | 585 |

## Sınıf dağılımı: Ana bölme

| küme | gen | etiketli | pozitif | pozitif oranı | genel oran | sapma (puan) |
|---|---|---|---|---|---|---|
| train | TP53 | 1367 | 588 | 0.430 | 0.430 | +0.0 |
| val | TP53 | 293 | 126 | 0.430 | 0.430 | -0.0 |
| test | TP53 | 293 | 126 | 0.430 | 0.430 | -0.0 |
| train | KRAS | 1367 | 236 | 0.173 | 0.173 | -0.0 |
| val | KRAS | 293 | 51 | 0.174 | 0.173 | +0.1 |
| test | KRAS | 293 | 51 | 0.174 | 0.173 | +0.1 |
| train | EGFR | 1367 | 66 | 0.048 | 0.048 | +0.0 |
| val | EGFR | 293 | 14 | 0.048 | 0.048 | -0.0 |
| test | EGFR | 293 | 14 | 0.048 | 0.048 | -0.0 |
| train | BRCA1 | 1367 | 36 | 0.026 | 0.027 | -0.0 |
| val | BRCA1 | 293 | 8 | 0.027 | 0.027 | +0.1 |
| test | BRCA1 | 293 | 8 | 0.027 | 0.027 | +0.1 |

## Sınıf dağılımı: Kurum tamamen test (site held-out)

| küme | gen | etiketli | pozitif | pozitif oranı | genel oran | sapma (puan) |
|---|---|---|---|---|---|---|
| train | TP53 | 1341 | 576 | 0.430 | 0.430 | -0.1 |
| val | TP53 | 287 | 123 | 0.429 | 0.430 | -0.2 |
| test | TP53 | 325 | 141 | 0.434 | 0.430 | +0.4 |
| train | KRAS | 1341 | 231 | 0.172 | 0.173 | -0.1 |
| val | KRAS | 287 | 50 | 0.174 | 0.173 | +0.1 |
| test | KRAS | 325 | 57 | 0.175 | 0.173 | +0.2 |
| train | EGFR | 1341 | 65 | 0.048 | 0.048 | +0.0 |
| val | EGFR | 287 | 14 | 0.049 | 0.048 | +0.1 |
| test | EGFR | 325 | 15 | 0.046 | 0.048 | -0.2 |
| train | BRCA1 | 1341 | 39 | 0.029 | 0.027 | +0.2 |
| val | BRCA1 | 287 | 8 | 0.028 | 0.027 | +0.1 |
| test | BRCA1 | 325 | 5 | 0.015 | 0.027 | -1.1 |

## CV katmanları (train+val havuzu)

| katman | hasta |
|---|---|
| 0 | 365 |
| 1 | 366 |
| 2 | 364 |
| 3 | 362 |
| 4 | 368 |

## Site held-out kümeleri

| küme | hasta |
|---|---|
| test | 350 |
| train | 1481 |
| val | 313 |

## Uyarılar

- (yok)
