# Baseline Omics Modelleri: hafif1

> `scripts/train_baselines.py` tarafından üretilir. Yalnızca toplu sayılar içerir.

## Protokol

- Yalnızca **geliştirme havuzu** (train+val) kullanıldı; **test kümesine dokunulmadı**.
- Dış döngü: 5 katlı CV (`cv_fold`). Ön işleme (gen filtresi, 500 yüksek varyanslı gen [protein_coding], z-skor) her dış katmanda **yalnızca eğitim satırlarından** öğrenildi.
- İç döngü: 3 katlı CV, AUPRC ile hiperparametre seçimi; F1 eşiği yalnızca eğitim tarafındaki iç CV tahminlerinden seçildi (raporlanan eşik, katman eşiklerinin medyanıdır).
- Sınıf dengesizliği: sınıf ağırlıkları (her katmanın içinde); yapay örnekleme yok.
- Güven aralıkları: %95, 1000 bootstrap örneği (hasta düzeyinde yeniden örnekleme).
- Pozitif sayısı < 30 olan kapsamlar eğitilmedi, aşağıda 'atlandı' olarak listelendi.
- Tüm metrikler OOF (katman dışı) tahminlerinden hesaplanır. Seçilen en iyi model aynı OOF tahminlerinde seçildiği için sayıları hafif iyimserdir; nihai tarafsız tahmin kilitli test kümesindeki final değerlendirmeden gelir.

## Birleşik (pan-cancer) model sonuçları

| gen | model | hasta | pozitif | prevalans | AUROC [%95 GA] | AUPRC [%95 GA] | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| TP53 | gbm (hist_gradient_boosting) | 1609 | 684 | 0.425 | 0.869 [0.850, 0.885] | 0.821 [0.790, 0.849] | 0.768 | 0.144 | 0.029 |
| TP53 | logreg_l2 | 1609 | 684 | 0.425 | 0.878 [0.861, 0.893] | 0.817 [0.785, 0.847] | 0.777 | 0.141 | 0.041 |
| TP53 | project_only | 1609 | 684 | 0.425 | 0.586 [0.558, 0.616] | 0.485 [0.451, 0.520] | 0.586 | 0.242 | 0.072 |

## Kanser türüne özel eğitim ve birleşik modelin tür bazında sonucu

`<tür>` = o türe özel eğitilmiş model; `pooled@<tür>` = birleşik modelin o türdeki hastalarda sonucu.

| gen | kapsam | model | hasta | pozitif | AUROC [%95 GA] | AUPRC [%95 GA] |
|---|---|---|---|---|---|---|
| TP53 | pooled@TCGA-BRCA | logreg_l2 | 820 | 276 | 0.920 [0.902, 0.938] | 0.847 [0.803, 0.884] |
| TP53 | pooled@TCGA-COAD | gbm | 364 | 197 | 0.816 [0.771, 0.859] | 0.833 [0.777, 0.884] |
| TP53 | pooled@TCGA-LUAD | gbm | 425 | 211 | 0.805 [0.760, 0.844] | 0.774 [0.705, 0.833] |

## Trivial taban (yalnızca kanser türü) ile karşılaştırma

Fark = en iyi omics modeli − kanser-türü-tabanı, eşleştirilmiş bootstrap %95 GA.

| gen | en iyi model | AUROC model / taban | ΔAUROC [GA] | AUPRC model / taban (prevalans) | ΔAUPRC [GA] | yorum (okuma yardımcısı) |
|---|---|---|---|---|---|---|
| TP53 | gbm | 0.869 / 0.586 | +0.283 [+0.250, +0.313] | 0.821 / 0.485 (0.425) | +0.336 [+0.297, +0.372] | kolay; omics, kanser türünden fazla bilgi veriyor |

> Yorum eşikleri (yazarların seçimi, veriye göre ayarlanmadı): AUROC ≥ 0.8 'kolay', < 0.65 'zor'. Bunlar otomatik okuma yardımcısıdır; biyolojik yorumu ekip yapar.

## Atlanan kapsamlar (sessizce geçilmedi)

- (yok)

## Uyarılar

- Şişirilmiş metrik şüphesi (AUROC > 0.95) yok.

## Şekiller

- `hafif1_roc.png`, `hafif1_pr.png`, `hafif1_calibration.png`
