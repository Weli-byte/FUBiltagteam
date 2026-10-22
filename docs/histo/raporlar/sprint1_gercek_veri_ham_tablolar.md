# Sprint 1 Gerçek Veri: Ham Çıktı Tabloları (18.10.2026)

> `scripts/collect_real_data_report.py` ile `data/` ve `outputs/sprint1_real/` altındaki gerçek çıktılardan otomatik üretildi; elle düzenlenmez. Yorum için `sprint1_gercek_veri_raporu.md`.

## Slaytlar (4a/4b)

| slide_id | kaynak | dosya | MB | md5 | boyut (px) | seviye | mpp | büyütme | vendor |
|---|---|---|---|---|---|---|---|---|---|
| tumor_091 | camelyon16 | tumor_091.tif | 546 | - | 61440x53760 | 8 | 0.2263 | 40.0 | philips |
| tumor_084 | camelyon16 | tumor_084.tif | 838 | - | 65536x86016 | 9 | 0.2263 | 40.0 | philips |
| normal_108 | camelyon16 | normal_108.tif | 327 | - | 45056x35840 | 8 | 0.2269 | 40.0 | philips |
| normal_150 | camelyon16 | normal_150.tif | 752 | - | 65536x60928 | 8 | 0.2269 | 40.0 | philips |
| 004d8238-0… | tcga | DX1 .svs | 479 | OK | 65963x27980 | 3 | 0.5005 | 20.0 | aperio |
| 00efe162-7… | tcga | DX1 .svs | 488 | OK | 87291x55135 | 4 | 0.2520 | 40.0 | aperio |
| 015862cf-a… | tcga | DX1 .svs | 465 | OK | 70024x82885 | 4 | 0.2520 | 40.0 | aperio |
| 00ab866f-a… | tcga | DX1 .svs | 195 | OK | 36451x57504 | 4 | 0.2520 | 40.0 | aperio |
| 01c79d06-a… | tcga | DX1 .svs | 499 | OK | 45312x53504 | 4 | 0.2325 | 20.0 | aperio |
| 032fc1ed-c… | tcga | DX1 .svs | 340 | OK | 56595x63763 | 4 | 0.2520 | 40.0 | aperio |
| 001b7d97-9… | tcga | DX1 .svs | 498 | OK | 125661x61796 | 4 | 0.2520 | 40.0 | aperio |
| 011da0ba-0… | tcga | DX1 .svs | 197 | OK | 133335x26607 | 4 | 0.2520 | 40.0 | aperio |
| 023ca8d1-d… | tcga | DX1 .svs | 245 | OK | 160194x33419 | 4 | 0.2520 | 40.0 | aperio |

## QC (4c)

| slide_id | tissue_fraction | n_tissue_tiles | blur_fraction | pen_fraction | dark_fraction | laplacian_var_median | decision |
|---|---|---|---|---|---|---|---|
| tumor_091 | 0.215 | 137 | 0.000 | 0.007 | 0.001 | 22444 | use |
| tumor_084 | 0.180 | 187 | 0.000 | 0.019 | 0.004 | 23686 | use |
| normal_108 | 0.209 | 66 | 0.000 | 0.002 | 0.001 | 32255 | use |
| normal_150 | 0.218 | 168 | 0.000 | 0.006 | 0.001 | 34455 | use |
| 004d8238-0… | 0.323 | 572 | 0.000 | 0.024 | 0.001 | 941 | use |
| 00efe162-7… | 0.603 | 705 | 0.000 | 0.000 | 0.001 | 2214 | use |
| 015862cf-a… | 0.433 | 613 | 0.000 | 0.005 | 0.004 | 2979 | use |
| 00ab866f-a… | 0.432 | 224 | 0.000 | 0.000 | 0.000 | 2864 | use |
| 01c79d06-a… | 0.156 | 78 | 0.000 | 0.064 | 0.000 | 2664 | exclude |
| 032fc1ed-c… | 0.445 | 387 | 0.000 | 0.019 | 0.001 | 2915 | use |
| 001b7d97-9… | 0.284 | 528 | 0.000 | 0.004 | 0.008 | 4017 | use |
| 011da0ba-0… | 0.293 | 249 | 0.000 | 0.020 | 0.019 | 2276 | use |
| 023ca8d1-d… | 0.211 | 277 | 0.000 | 0.029 | 0.005 | 2273 | use |

Karar dağılımı: {'use': 12, 'exclude': 1}

## Patch koordinat çıkarma (4d)

Özet: {'n_slides': 13, 'done': 13, 'skipped': 0, 'failed': 0, 'patches_total': 67116, 'seconds_per_slide': 0.8184222461539041, 'slides_per_minute': 73.31179019383231}

| slide_id | durum | patch | aday | sn |
|---|---|---|---|---|
| 001b7d97-9… | done | 8537 | 29887 | 0.80 |
| 004d8238-0… | done | 9104 | 28013 | 0.66 |
| 00ab866f-a… | done | 3510 | 8023 | 1.09 |
| 00efe162-7… | done | 11246 | 18468 | 0.87 |
| 011da0ba-0… | done | 4025 | 13624 | 0.17 |
| 015862cf-a… | done | 9715 | 22331 | 1.01 |
| 01c79d06-a… | done | 1236 | 7954 | 0.85 |
| 023ca8d1-d… | done | 4397 | 20475 | 0.20 |
| 032fc1ed-c… | done | 6203 | 13875 | 1.51 |
| normal_108 | done | 1056 | 4977 | 0.82 |
| normal_150 | done | 2728 | 12528 | 0.68 |
| tumor_084 | done | 3156 | 17365 | 0.74 |
| tumor_091 | done | 2203 | 10152 | 1.23 |

## patch_v0 gerçek örnek paket

Sayılar: {"patches": 2600, "slides": 13, "tumor": 2000, "normal": 600, "unlabelled": 0, "by_split": {"train": 800, "unassigned": 1800}, "by_source": {"camelyon16": 800, "tcga": 1800}}
Başarısız: []

## Boya normalizasyonu karşılaştırması (4e)

Merkez anahtarı: centre; merkezler: ['TCGA-BRCA-TSSA8', 'TCGA-BRCA-TSSAC', 'TCGA-COAD-TSSA6', 'TCGA-LUAD-TSS75', 'TCGA-LUAD-TSS86', 'TCGA-LUAD-TSS97', 'camelyon16-center1']; patch sayısı: 700

| yöntem | merkezler arası LAB farkı | merkez sınıflandırıcı doğruluğu (ort ± sd) | şans | hata |
|---|---|---|---|---|
| none | 6.474 | 0.676 ± 0.007 | 0.143 | 0 |
| macenko | 2.291 | 0.419 ± 0.010 | 0.143 | 0 |
| reinhard | 0.220 | 0.489 ± 0.022 | 0.143 | 0 |

## ViT-S/16 vs ResNet-50 kısa koşu (4g) — NİHAİ DEĞİL

| model | AUROC (ort, %95 GA) | AUPRC | F1 | doğruluk | seed sayısı |
|---|---|---|---|---|---|
| vit_small | 0.780 [0.729, 0.831] | 0.426 [0.373, 0.479] | 0.057 [-0.055, 0.170] | 0.745 [0.735, 0.755] | 3 |
| resnet50 | 0.769 [0.659, 0.879] | 0.453 [0.317, 0.589] | 0.169 [0.097, 0.240] | 0.738 [0.717, 0.758] | 3 |

- vit_small seed 0: en iyi epoch 0, 4 epoch koştu, AUROC 0.740 [0.691, 0.792], AUPRC 0.405, F1@0.5 0.172, doğruluk 0.735
- vit_small seed 1: en iyi epoch 7, 8 epoch koştu, AUROC 0.770 [0.727, 0.812], AUPRC 0.393, F1@0.5 0.000, doğruluk 0.750
- vit_small seed 2: en iyi epoch 3, 7 epoch koştu, AUROC 0.829 [0.790, 0.866], AUPRC 0.479, F1@0.5 0.000, doğruluk 0.750
- resnet50 seed 0: en iyi epoch 7, 8 epoch koştu, AUROC 0.803 [0.759, 0.847], AUPRC 0.490, F1@0.5 0.202, doğruluk 0.743
- resnet50 seed 1: en iyi epoch 5, 8 epoch koştu, AUROC 0.845 [0.803, 0.881], AUPRC 0.551, F1@0.5 0.208, doğruluk 0.752
- resnet50 seed 2: en iyi epoch 7, 8 epoch koştu, AUROC 0.659 [0.604, 0.714], AUPRC 0.319, F1@0.5 0.096, doğruluk 0.718
