# 1.9 — Koşullu Difüzyon PoC tasarımı

> Araştırma prototipi. Çıktılar **simülasyon** (koşullu örnek üretimi) olup tahmin veya kanıt değildir.
> Kaynak numaraları `literatur_taramasi.md` içindir ve henüz doğrulanmamıştır.

## Soru
Gerçek TCGA ekspresyon profillerinde (1.4 çıktısı), kanser tipi ve mutasyon etiketine koşullu bir
difüzyon modeli, nadir sınıflar (ör. EGFR+) için **gerçek veriden ayırt edilemeyen ve alt görevde
yarar sağlayan** örnekler üretebilir mi?

## Neden omics önce, histoloji sonra
- Omics verisi hazır (1.4–1.6), küçük ve hızlı; PoC günler değil saatler sürer.
- Histoloji patch özellikleri Sprint 2'de gelir; aynı mimari 7 numaralı kaynaktaki gibi latent uzaya taşınabilir.
- Piksel düzeyinde difüzyon bu aşamada kapsam dışı.

## Veri ve bölme kuralları
- Girdi: yalnızca 1.6 `splits-v1` içindeki **train** hastaları. Val modelleme kararları için, **test hiç kullanılmaz**
  (`get_ids(final_evaluation=True)` kilidi geçerli).
- Ön işleme eğitim bölmesinde fit edilir (1.4 `ExpressionPreprocessor`), sızıntı testi (`test_no_leakage`) kapsar.
- Üretilen örnekler **hiçbir bölmeye gerçek hasta olarak eklenmez**; her zaman `synthetic=True` bayrağıyla ayrı tutulur
  ve test/val değerlendirmesine girmez.

## Model (en küçük anlamlı sürüm)
- Uzay: PCA/seçilmiş gen alt kümesi (örn. 256–2000 boyut), eğitim bölmesinde fit.
- Ağ: küçük MLP tabanlı gürültü tahmincisi, zaman gömmesi + koşul gömmesi (#1, #4).
- Koşul: kanser tipi (3 sınıf) + hedef mutasyon etiketi (EGFR/KRAS/TP53/BRCA1; **bilinmeyen = ayrı "maske" jetonu**, 0 değil).
- Kılavuzluk: sınıflandırıcısız, koşul bırakma olasılığı 0.1–0.2 (#6).
- Örnekleme: DDIM ile az adım (#2).

## Değerlendirme (hepsi önceden tanımlı)
1. **Alt görev faydası (ana ölçüt):** 1.8 temel modelini (a) yalnızca gerçek train ile, (b) gerçek + üretilen ile
   eğit; **val** üzerinde AUPRC farkı, eşleştirilmiş bootstrap GA ile (≥1000). GA 0'ı kesiyorsa "fayda kanıtlanmadı" denir.
2. **Dağılım benzerliği:** gerçek-vs-üretilen için ayırt edici sınıflandırıcı AUROC'u (0.5'e yakın iyi), ana bileşen
   uzayında kapsama (density/coverage, #17). FID doğrudan kullanılmaz (özellik çıkarıcı gerektirir, #16).
3. **Ezberleme:** her üretilen örnek için en yakın eğitim hastasına uzaklık; gerçek-gerçek (train-val) uzaklık
   dağılımıyla kıyas (#14, #15). Belirgin biçimde daha yakınsa sonuç **raporlanır ve paylaşım yapılmaz**.
4. **Biyolojik makullük:** bilinen belirteç genlerinde (örn. EGFR+ LUAD) yön tutarlılığı; yalnızca betimleyici,
   kanıt iddiası yok.

## Dürüstlük kuralları
- Negatif sonuç geçerli sonuçtur ve aynı ayrıntıyla raporlanır.
- Sınıf başına pozitif < 30 ise (1.5 raporu) o hedef için üretim *denenir ama başarı iddia edilmez*.
- Arayüzde/raporda üretilen çıktı "simülasyon" etiketi ve araştırma prototipi uyarısı taşır.
- Gerçek hasta tedavi yanıtı veya klinik öneri üretilmez.

## Riskler
| Risk | Etki | Önlem |
|---|---|---|
| Ezberleme (az pozitif) | Mahremiyet, sahte başarı | Ölçüm 3, paylaşım yok |
| Alt görev faydası yok | PoC başarısız | Dürüst rapor; SMOTE/ağırlık ile kıyas |
| Sızıntı (test'ten öğrenme) | Geçersiz sonuç | Kilitli test, mevcut sızıntı testleri |
| Kaynak yanlışlığı | İtibar | Doğrulama sütunu zorunlu |

## Kabul ölçütleri (Sprint 3'te PoC için)
- Tüm değerlendirme adımları betikli ve tohumlu; MLflow'da kayıtlı.
- En az 1 hedefte alt görev sonucu (olumlu ya da olumsuz) GA ile raporlanmış.
- Ezberleme raporu mevcut.

## Açık sorular (ekip kararı)
1. Sprint 3'te yalnızca omics mi, histoloji özellik uzayı da mı?
2. Pozitif sayısı düşük hedefler (BRCA1) için PoC denenmeli mi?
3. Ezberleme eşiği kim tarafından, hangi değerde onaylanacak?
