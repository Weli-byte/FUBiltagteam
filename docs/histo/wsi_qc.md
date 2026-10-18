# WSI Kalite Kontrolü ve Doku Tespiti v1 (Mini sprint 1.6)

> **Durum:** Hat ve testler hazır (`models/histo/wsi/{tissue,qc}.py`, `scripts/run_wsi_qc.py`);
> birim testleri (sentetik test düzeneği, yalnızca `tests/` altında) temiz / kalem izli / bulanık /
> boş cam ayrımını sınar. **Eşikler gerçek slaytlarda kalibre edilmedi.** Kalibrasyon protokolü
> §4'tedir ve gerçek veriyle ilk 50 slaytta uygulanacak.

**Sade anlatım:** Bazı doku görüntüleri kötüdür: bulanık, üzerinde kalem izi olan, katlanmış ya da
neredeyse boş camlar. Bu adım "kalite kontrol bandı" gibi çalışır: her slaytı ölçer, kötüleri
gerekçesiyle işaretler, doku olmayan beyaz alanları eler.

## 1. Doku tespiti v1 (`tissue_mask`)

HSV doygunluk kanalı (boyalı doku doygun, cam değil) → medyan bulanıklaştırma (7 px) → Otsu
eşiği (asla `saturation_floor=15`'in altına inmez; boş camda her şeyi doku sanmayı önler) →
morfolojik açma (5 px, lekeleri siler) ve kapama (9 px, küçük boşlukları kapatır) → küçük nesne
temizliği (< 400 px) → delik doldurma (≤ 2.000 px). Ölçek: thumbnail 2048 px; eşikler thumbnail
pikselidir. v0 (1.3) = yalnızca Otsu. Referanslar: HistoQC (Clear BSD) ve CLAM (GPLv3) yaklaşımları
incelendi; kod GPL kodu **içermez**, yeniden yazıldı.

## 2. Slayt düzeyi QC metrikleri (`compute_qc`)

Analiz ~5x'te (`analysis_mpp=2.0`), 256 px karelerle, yalnızca dokusu ≥ %50 olan karelerde:

| Metrik | Nasıl | Yakaladığı sorun |
|---|---|---|
| `tissue_fraction`, `n_components`, `largest_component_fraction` | maske istatistikleri | boş/çok küçük doku, parçalanmış kesit |
| `blur_fraction`, `laplacian_var_median` | Laplacian varyansı < 40 olan kare oranı | odak hatası |
| `pen_fraction` | HSV aralık filtreleri (mavi, yeşil, siyah mürekkep) | kalem/işaretçi |
| `dark_fraction` | V < 60 piksel oranı | katlanma, aşırı boyama, siyah ink |

Karar (`decide`): `exclude` = herhangi bir metrik eşiği aşıyor ya da doku yok; `suspect` = eşiğin
%60'ını aşıyor; `use` = hiçbiri. Her karar gerekçesiyle (`reasons`: metrik, değer, eşik) parquet'e
yazılır. Eşikler `configs/histo/qc.yaml`'da; kod içinde sihirli sayı yok.

## 3. Komut

```bash
uv run --group histo python scripts/run_wsi_qc.py                       # data/manifests/slides_to_extract.csv
uv run --group histo python scripts/run_wsi_qc.py run.manual_sample=50 qc.blur_laplacian_var_min=30
```
Çıktı: `data/interim/qc_results.parquet`, `docs/histo/qc_raporu.md` (dağılımlar, dışlama listesi),
`data/interim/qc_overlays/*.png` (manuel inceleme için 50 rastgele slayt; thumbnail + yeşil maske).
`select_slides(inventory, qc)` (1.4) `exclude`'ları elemek için bu parquet'i kullanır.

## 4. Kalibrasyon protokolü (gerçek veride yapılacak)

1. 50 rastgele slayt (kanser türü ve merkeze göre tabakalı) **gözle** etiketlenir: doku maskesi
   doğru mu (evet/kısmen/hayır), kalem izi var mı, bulanık mı, katlanma var mı, genel karar.
2. Otomatik karar ile karşılaştırılır; hedef uyum ≥ %90 (`use/exclude` ikili). Uyumsuz vakalar
   hangi metrikten kaynaklandıysa o eşik değiştirilir, **test performansına bakılmaz**.
3. Dışlananların kanser türü / kurum (TSS) dağılımı raporlanır; bir kurum orantısız eleniyorsa
   eşik kurum bazında incelenir (QC yanlılığı).
4. Laplacian eşiği tarayıcıya bağlıdır (JPEG sıkıştırma, mpp); CAMELYON (3DHistech/Hamamatsu) ve
   TCGA (Aperio) için ayrı medyan dağılımları kaydedilir.
5. Sonuç tablosu (uyum %, eşikler, tarih) bu belgeye eklenir.

## 5. Sınırlılıklar

- Kırmızı kalem H&E pembesine benzediği için renk aralığıyla ayırt edilmez (literatürde de zor).
- Laplacian varyansı doku yoğunluğuna da duyarlıdır: seyrek doku bulanık sanılabilir; bu yüzden
  karar kare oranı üzerinden verilir ve `suspect` ara sınıfı vardır.
- Katlanma yalnızca "koyu" olarak yakalanır; dokuya bakan daha iyi yöntemler (HistoQC
  modülleri) Sprint 2'de değerlendirilebilir.

## 6. Doğrulama (bitti kriteri)

- [x] Doku maskesi mantığı birim testlerinde doğrulandı (IoU > 0,9 test düzeneğinde; speck/hole temizliği).
- [x] Kalem izli, bulanık ve boş cam ayrımı birim testlerinde doğru (`tests/test_histo_tissue_qc.py`); gerçek slaytta henüz değil.
- [ ] 50 gerçek slaytta manuel-otomatik uyum ≥ %90 raporlandı (tablo buraya).
- [ ] Dışlananların kanser türü ve kurum dağılımı raporlandı.
- [ ] 20 rastgele gerçek slaytta doku maskesi gözle doğrulandı.
