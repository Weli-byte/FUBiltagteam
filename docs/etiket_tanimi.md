# Mutasyon Etiketi Tanımı (Mini sprint 1.5)

Etiketler **araştırma etiketleridir**, klinik tanı değildir. Kaynak: GDC maskeli somatik mutasyon (MAF)
dosyaları (1.3'te indirilen). Kurallar `configs/omics/labels.yaml` içindedir; kodda gömülü değildir.

## Etiket kuralı
Her hasta ve hedef gen (varsayılan: TP53, KRAS, EGFR, BRCA1) için ikili etiket:

| Etiket | Anlam |
|---|---|
| **1** | Hastanın etiket örneğinde, hedef genin en az bir varyantı **sayılan sınıflardan** birinde |
| **0** | Hastanın kullanılabilir MAF'ı var ve hedef genin sayılan türde varyantı **yok** |
| **bilinmiyor (NA)** | Etiket belirlenemedi (aşağıya bak). **0 olarak kodlanmaz.** |

**Sayılan varyant sınıfları:** Missense_Mutation, Nonsense_Mutation, Frame_Shift_Del,
Frame_Shift_Ins, Splice_Site, In_Frame_Del, In_Frame_Ins, Translation_Start_Site.
**Sayılmayanlar:** Silent, Intron, 3'/5' UTR, flank, RNA ve benzeri. Gerekçe: bu sınıflar protein
işlevini değiştirdiği varsayılamaz. Dışlanan varyantların sayısı rapora yazılır, sessizce atılmaz.
Aynı gende birden çok varyant etiketi yine 1 yapar (sayı değil, var/yok).

## Hasta düzeyinde tekilleştirme
- Etiket yalnızca **birincil tümör (TCGA örnek tipi 01)** örneğinden gelir. Normal doku ve metastaz
  örnekleri kullanılmaz.
- Hastanın birden fazla birincil tümör örneği varsa, **RNA-seq için kullanılan örnek** tercih edilir
  (etiket ve ifade aynı numuneyi anlatsın). O örnek MAF'ta yoksa en küçük örnek barkodu alınır.
- Birden fazla MAF dosyasında aynı varyant tekrar ederse bir kez sayılır.

## Etiketin bilinmediği durumlar (`label_status`)
| Durum | Açıklama |
|---|---|
| `labeled` | MAF var ve birincil tümör örneği içeriyor |
| `no_maf` | Hastanın indirilmiş MAF dosyası yok |
| `no_primary_tumor_maf` | MAF var ama birincil tümör (01) örneği yok |

Toplam hasta sayısı `patient_map` (1.3 envanteri) ile aynıdır; hiçbir hasta tabloda silinmez.

## Varyant düzeyi tablo
`labels_variant_detail.parquet`: etiket örneğindeki tüm hedef gen varyantları (sayılan/sayılmayan
bayrağıyla) ve `annotations` sütunu (EGFR L858R, EGFR exon 19 delesyonu, KRAS G12C/G12D/G12V...).
Bunlar **etiket değildir**, ek bilgidir; yeni bir anotasyon eklemek için yapılandırmaya bir satır yeter.

## Yeni gen eklemek
`configs/omics/labels.yaml` içindeki `genes` listesine gen sembolünü ekleyip betiği yeniden çalıştırın;
kod değişmez. Komut satırından: `genes=[TP53,KRAS,EGFR,BRCA1,PIK3CA]`.

## Çalıştırma
```bash
uv sync --group data --group omics
uv run python scripts/build_mutation_labels.py
```
Çıktılar: `data/processed/labels/` (git'e girmez, DVC) ve `docs/mutation_labels/` (toplu sayılar).

## Doğrulama (bitti kriteri; kısmen senin yapman gereken)
1. **5 rastgele hastayı cBioPortal ile karşılaştır** (ilgili TCGA çalışması, hasta barkodu, hedef gen
   mutasyonu var/yok). Bunu ben yapamam; ağım kapalı ve portala senin bakman gerekir.
2. Toplam hasta sayısı `docs/veri_envanteri.md` ile aynı olmalı (betik satır sayısını kontrol eder).
3. Betik aynı girdiyle aynı çıktıyı üretmeli (testle korunuyor).

## Sınırlılıklar ve dürüstlük notları
- MAF sütun adları belgelerden yazıldı; gerçek dosyada ilk çalıştırma doğrular (hata açık verir).
- Etiket **somatik kodlayan değişiklikleri** yakalar; amplifikasyon, germline BRCA1 varyantları ve
  ifade azalması **sayılmaz**. BRCA1 somatik mutasyonu nadirdir; pozitif sayısı küçük çıkabilir.
- Aşırı mutasyonlu (hypermutated) tümörler (örn. MSI/POLE, özellikle kolorektal) birçok genin
  etiketini "pozitif" yapabilir; bu bir veri özelliği olarak raporlanır, düzeltilmez.
- "MAF dosyası var" = "dizilenmiş" varsayılır; boş bir MAF'ı ayırt etmek için dosya içeriği yetmez.
- Nadir hedeflerde (örn. EGFR) pozitif sayısı azsa bu açıkça yazılır; eşik 30'dur (yapılandırmadan).
- Önerilen kayıp fonksiyonu/grup stratejisi kural tabanlıdır (eşikler ayarlanmadı) ve 1.8'deki
  baseline sonuçlarıyla doğrulama kümesinde sınanmalıdır.
