# Mutasyon Etiketi Sınıf Dengesi

> `scripts/build_mutation_labels.py` tarafından üretilir. Yalnızca toplu sayılar içerir.

- Genler: TP53, KRAS, EGFR, BRCA1
- Mutasyon sayılan sınıflar: Frame_Shift_Del, Frame_Shift_Ins, In_Frame_Del, In_Frame_Ins, Missense_Mutation, Nonsense_Mutation, Splice_Site, Translation_Start_Site
- Pozitif sayısı < 30 olan gruplar `low_positives` ile işaretlidir.
- 'dengesiz' eşiği: pozitif oranı < 0.15 (yapılandırmadan).
- `unknown`: MAF'ı olmayan veya birincil tümör örneği bulunmayan hastalar.

| proje | gen | hasta | etiketli | pozitif | negatif | bilinmeyen | oran | düşük |
|---|---|---|---|---|---|---|---|---|
| TCGA-BRCA | TP53 | 1098 | 968 | 331 | 637 | 130 | 0.342 | hayır |
| TCGA-BRCA | KRAS | 1098 | 968 | 6 | 962 | 130 | 0.006 | evet |
| TCGA-BRCA | EGFR | 1098 | 968 | 9 | 959 | 130 | 0.009 | evet |
| TCGA-BRCA | BRCA1 | 1098 | 968 | 23 | 945 | 130 | 0.024 | evet |
| TCGA-COAD | TP53 | 461 | 428 | 234 | 194 | 33 | 0.547 | hayır |
| TCGA-COAD | KRAS | 461 | 428 | 183 | 245 | 33 | 0.428 | hayır |
| TCGA-COAD | EGFR | 461 | 428 | 11 | 417 | 33 | 0.026 | evet |
| TCGA-COAD | BRCA1 | 461 | 428 | 10 | 418 | 33 | 0.023 | evet |
| TCGA-LUAD | TP53 | 585 | 557 | 275 | 282 | 28 | 0.494 | hayır |
| TCGA-LUAD | KRAS | 585 | 557 | 149 | 408 | 28 | 0.268 | hayır |
| TCGA-LUAD | EGFR | 585 | 557 | 74 | 483 | 28 | 0.133 | hayır |
| TCGA-LUAD | BRCA1 | 585 | 557 | 19 | 538 | 28 | 0.034 | evet |
| ALL | TP53 | 2144 | 1953 | 840 | 1113 | 191 | 0.430 | hayır |
| ALL | KRAS | 2144 | 1953 | 338 | 1615 | 191 | 0.173 | hayır |
| ALL | EGFR | 2144 | 1953 | 94 | 1859 | 191 | 0.048 | hayır |
| ALL | BRCA1 | 2144 | 1953 | 52 | 1901 | 191 | 0.027 | hayır |

## Öneri (kural tabanlı; veriye bakılarak ayarlanmamış eşiklerle)

| proje | gen | öneri |
|---|---|---|
| TCGA-BRCA | TP53 | yeterince dengeli: standart BCE |
| TCGA-BRCA | KRAS | yetersiz pozitif (<30): bu grupta ayrı model eğitme; havuzlu (pan-cancer) eğitimde kullan ve sonucu AUPRC ile dürüstçe raporla |
| TCGA-BRCA | EGFR | yetersiz pozitif (<30): bu grupta ayrı model eğitme; havuzlu (pan-cancer) eğitimde kullan ve sonucu AUPRC ile dürüstçe raporla |
| TCGA-BRCA | BRCA1 | yetersiz pozitif (<30): bu grupta ayrı model eğitme; havuzlu (pan-cancer) eğitimde kullan ve sonucu AUPRC ile dürüstçe raporla |
| TCGA-COAD | TP53 | yeterince dengeli: standart BCE |
| TCGA-COAD | KRAS | yeterince dengeli: standart BCE |
| TCGA-COAD | EGFR | yetersiz pozitif (<30): bu grupta ayrı model eğitme; havuzlu (pan-cancer) eğitimde kullan ve sonucu AUPRC ile dürüstçe raporla |
| TCGA-COAD | BRCA1 | yetersiz pozitif (<30): bu grupta ayrı model eğitme; havuzlu (pan-cancer) eğitimde kullan ve sonucu AUPRC ile dürüstçe raporla |
| TCGA-LUAD | TP53 | yeterince dengeli: standart BCE |
| TCGA-LUAD | KRAS | yeterince dengeli: standart BCE |
| TCGA-LUAD | EGFR | dengesiz: ağırlıklı BCE (pos_weight) veya focal loss; AUROC yanında AUPRC raporla |
| TCGA-LUAD | BRCA1 | yetersiz pozitif (<30): bu grupta ayrı model eğitme; havuzlu (pan-cancer) eğitimde kullan ve sonucu AUPRC ile dürüstçe raporla |
| ALL | TP53 | yeterince dengeli: standart BCE |
| ALL | KRAS | yeterince dengeli: standart BCE |
| ALL | EGFR | dengesiz: ağırlıklı BCE (pos_weight) veya focal loss; AUROC yanında AUPRC raporla |
| ALL | BRCA1 | dengesiz: ağırlıklı BCE (pos_weight) veya focal loss; AUROC yanında AUPRC raporla |

## Sayılmayan varyantlar (etiket örneğinde, hedef genlerde)

| Variant_Classification | adet |
|---|---|
| 3'UTR | 3 |
| Silent | 35 |
| Splice_Region | 15 |
