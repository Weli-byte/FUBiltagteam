# WSI Kalite Kontrol Raporu (13 gerçek slayt, 18.10.2026)

> `scripts/run_wsi_qc.py` tarafından üretildi. Girdi listesi: `sprint1_gercek_ornek_slaytlar.csv`; 13 slayt.
> Eşikler `configs/histo/qc.yaml` içindedir; kalibrasyon notu `docs/histo/wsi_qc.md`.

## Karar dağılımı

| Karar | Slayt |
|---|---|
| exclude | 1 |
| use | 12 |

## Metrik özetleri

| Metrik | ortalama | medyan | %90 | maks |
|---|---|---|---|---|
| tissue_fraction | 0.308 | 0.284 | 0.442 | 0.603 |
| blur_fraction | 0.000 | 0.000 | 0.000 | 0.000 |
| pen_fraction | 0.015 | 0.007 | 0.028 | 0.064 |
| dark_fraction | 0.004 | 0.001 | 0.007 | 0.019 |

## Eşikler

- tissue_fraction >= 0.02, n_tissue_tiles >= 10
- blur_fraction <= 0.3 (Laplacian var < 40.0 bulanık)
- pen_fraction <= 0.05, dark_fraction <= 0.25
- 'suspect': metrik eşiğin 60%'ini aşarsa

## Dışlanan slaytlar (1)

| slide_id | gerekçe |
|---|---|
| 01c79d06-a4a7-47a7-bb8b-b98ce6652c99 | pen_fraction 0.064 > 0.05 |

## Şüpheli slaytlar (0)

| slide_id | gerekçe |
|---|---|

## Manuel doğrulama

Rastgele seçilen slaytların maske bindirme görselleri `overlay_dir` altındadır. Gözle
inceleme sonucunu (uyum yüzdesi, yanlış dışlananlar) `docs/histo/wsi_qc.md`'ye işleyin.
Dışlananların kanser türü / kurum dağılımını da orada raporlayın (QC yanlılığı kontrolü).
