# Sprint 1 — Keşifsel veri analizi (EDA) özeti

> Kaynaklar: `docs/omics_qc/qc_report.md` ve `docs/mutation_labels/dengesizlik_raporu.md`
> (dal `veri/omics-etiket-raporlari`, commit 411a7ed; gerçek TCGA verisi, GDC). Sayılar o raporlardan
> aynen alınmıştır. Bölme (1.6) ve temel model (1.8) sonuçları henüz gerçek veride çalıştırılmadığı için
> **bu raporda yoktur**; çalıştırılınca bölüm 5 doldurulur.

## 1. Kohort
| Proje | Hasta (etiket tablosu) | RNA-seq örneği (QC) |
|---|---|---|
| TCGA-BRCA | 1098 | 1095 |
| TCGA-COAD | 461 | 458 |
| TCGA-LUAD | 585 | 517 |
| Toplam | 2144 | 2070 |

LUAD'da etiketli hasta (585) ile RNA-seq örneği (517) arasındaki 68 farkın nedeni bu raporlardan
anlaşılamıyor; `patient_map` üzerinden **incelenmeli** (açık madde).

## 2. Omics kalite kontrolü
- Bayraklı örnek (herhangi bayrak): BRCA 12, COAD 34, LUAD 7. Hiçbiri silinmedi; karar `--drop-flagged` ile ekipçe verilecek.
- COAD'da 18 örnek kitaplık büyüklüğü bayrağı taşıyor (diğer projelerde 0); COAD'a özgü bir kalite farkı olabilir, **neden doğrulanmadı**.
- Batch: silhouette kanser türü 0.347; doku kaynak merkezi (TSS) proje içinde −0.226 / −0.106 / −0.174.
  Önceden belirlenen yorum kuralına göre kanser türü sinyali TSS sinyalinden belirgin yüksek: **TSS batch'i bu ölçüte göre baskın değil**.
  Bu, batch etkisinin olmadığını göstermez; yalnızca bu ölçümde belirgin görünmediğini gösterir.

## 3. Mutasyon etiketleri (hasta düzeyi, bilinmeyen ≠ 0)
| Gen | Pozitif / etiketli (ALL) | Oran |
|---|---|---|
| TP53 | 840 / 1953 | 0.430 |
| KRAS | 338 / 1953 | 0.173 |
| EGFR | 94 / 1953 | 0.048 |
| BRCA1 | 52 / 1953 | 0.027 |

- Bilinmeyen etiketli hasta: 191 (MAF yok veya birincil tümör örneği yok); eğitime/değerlendirmeye girmez.
- Proje içinde pozitif < 30 olan gruplar: BRCA-KRAS (6), BRCA-EGFR (9), BRCA-BRCA1 (23), COAD-EGFR (11), COAD-BRCA1 (10), LUAD-BRCA1 (19).
- Proje-içi dağılımlar çok farklı (ör. EGFR: LUAD 74, BRCA 9, COAD 11). Kanser türü, mutasyon etiketiyle güçlü karışabilir;
  bu yüzden 1.8'de kanser türü tabanı (`project_only`) ile eşleştirilmiş kıyas yapılır.
- Sayılmayan varyantlar (etiket örneğinde): Silent 35, Splice_Region 15, 3'UTR 3.

## 4. Çıkarımlar ve sınırlar
1. TP53 ve (COAD/LUAD içinde) KRAS yeterince dengeli; EGFR ve BRCA1 çok dengesiz.
2. EGFR/BRCA1 için test kümesinde yalnızca birkaç pozitif düşecek → güven aralıkları geniş olacak; bu sonuçlar "kesin" denemez.
3. Etiketler MAF'tan türetilmiş **somatik mutasyon** etiketleridir; fonksiyonel etki veya tedavi yanıtı etiketi değildir.
4. Tümü TCGA'dır; harici doğrulama yoktur → genelleme iddiası yok.
5. Pozitif hasta örnekleri (EGFR E746_A750del, L858R; KRAS G12D, G12V) bilinen sıcak noktalarla tutarlı; cBioPortal ile elle karşılaştırma **sizde bekliyor**.

## 5. Çalıştırma sonrası doldurulacak
- [ ] 1.6 bölme sayıları ve dağılım uyarıları (`docs/splits/splits-v1_*`)
- [ ] 1.8 temel model sonuçları (`docs/baselines/`), kanser türü tabanına karşı eşleştirilmiş fark
