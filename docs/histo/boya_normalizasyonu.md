# Boya Normalizasyonu ve Histolojik Augmentation (Mini sprint 1.8)

> **Durum:** Macenko ve Reinhard normalizasyonu ile histolojiye uygun augmentation modülleri
> yazıldı ve birim testleriyle sınandı (`models/histo/stain/`, `tests/test_histo_stain.py`).
> **Gerçek patch'lerde ölçüldü** (18.10.2026, 13 slayt, §4); merkez/doku karışıklığı nedeniyle
> varsayılan karar askıda (§5). 3 seed'li kısa ViT deneyi (normalizasyon yok/var) yapılmadı.

**Sade anlatım:** Her hastane dokuyu biraz farklı tonda boyar; biri pembe, diğeri mor. Model
renk farkını "hastalık" sanabilir. Normalizasyon tüm görüntüleri benzer tona getirir;
augmentation ise eğitimde döndürme/renk oynama gibi küçük değişikliklerle modeli renge ve yöne
karşı dayanıklı yapar.

## 1. Yöntemler

| Yöntem | Ne yapar | Artı | Eksi |
|---|---|---|---|
| `none` | hiçbir şey | hızlı, veri bozulmaz | merkez farkı kalır |
| `macenko` (Macenko ve ark., ISBI 2009; doğrulanmalı) | görüntüyü optik yoğunluğa çevirir, H ve E boya vektörlerini SVD ile bulur, yoğunlukları hedefe ölçekler | fiziksel model (Beer-Lambert), boya ayrımı korunur | az dokulu patch'te başarısız (ValueError → orijinal tutulur); yavaş (ms/patch; GPU için torchstain MIT alternatif) |
| `reinhard` (Reinhard ve ark., 2001; doğrulanmalı) | LAB uzayında ortalama/std eşleme | çok hızlı, kararlı | boya anlamını bilmez; nadiren renk kayması |

Hepsi `fit(hedef_görüntü)` → `transform(görüntü)`; durum (`state()`) JSON olarak kaydedilir
(provenance). Hedef görüntü: temsili bir patch (`configs/histo/stain.yaml: reference_image`);
seçim kriteri: medyan LAB istatistiğine en yakın eğitim patch'i (betik Sprint 2'de). **Test
kümesine ayrı normalizasyon parametresi uydurulmaz**; hedef yalnızca eğitim verisinden seçilir.

## 2. Augmentation (`HistoAugment`)

| Dönüşüm | Varsayılan | Neden histolojiye uygun |
|---|---|---|
| 90° katları döndürme, yatay/dikey çevirme | her zaman | doku yönsüzdür |
| HED titreşimi (α ∈ 1±0,05, β ∈ ±0,05; p=0,8) | açık | boya yoğunluğu farkını taklit eder (Tellez 2018 tarzı; doğrulanmalı) |
| parlaklık/kontrast ±%8 (p=0,8) | açık | tarayıcı farkı |
| Gauss bulanıklık σ≤1 (p=0,1), gürültü σ=4 (p=0,1) | açık | odak/sensör farkı |
| **yok**: elastik bükme, büyük renk kayması, kesme | kapalı | doku morfolojisini bozar |

Deterministik: aynı `seed` aynı dizi (test: `test_hed_roundtrip_and_augment_determinism`).
TTA: 8 dihedral görünüm (`dihedral_views`) + ortalama (`predict_tta`), `train.tta=true`.

## 3. Ölçüm protokolü (`scripts/compare_stain_norm.py`)

1. Her merkezden N patch (`compare.n_per_centre`), üç durum: none / macenko / reinhard.
2. **Renk istatistiği**: merkez başına LAB ortalaması; merkezler arası ortalama mutlak fark
   (normalizasyon sonrası düşmeli).
3. **Merkez ayırt edici sınıflandırıcı**: yalnızca renk histogramından (3x8 bin) merkezi tahmin
   eden lojistik regresyon, 3 seed, %30 held-out; doğruluk şansa yaklaşmalı.
4. Önce/sonra görsel ızgarası (`before_after_grid.png`), gözle 50 örnek kontrolü.
5. Kısa ViT deneyi (3 seed, none vs macenko vs reinhard, aynı bölme) → `train_patch_classifier.py`
   `train.stain=...` bayrağı Sprint 2'de eklenecek; sonuç AUROC + GA olarak buraya.

## 4. Gerçek veri sonucu (18.10.2026, küçük örnek)

`scripts/compare_stain_norm.py`, gerçek `patch_v0` örneğinde (13 slayt; 7 "merkez" = 6 TCGA TSS +
CAMELYON16 center1; 100 patch/merkez; 3 seed):

| yöntem | merkezler arası LAB farkı | merkez sınıflandırıcı doğruluğu (ort ± sd) | şans |
|---|---|---|---|
| none | 6,47 | 0,676 ± 0,007 | 0,143 |
| macenko | 2,29 | 0,419 ± 0,010 | 0,143 |
| reinhard | 0,22 | 0,489 ± 0,022 | 0,143 |

Her iki yöntem renk farkını azaltıp merkez tahminini zorlaştırıyor; şansa inmiyor. **Sınırlılık:**
merkezler kanser türü/dokuyla karışık (akciğer, meme, kolon, lenf düğümü), farkın bir kısmı
biyolojik. Macenko 700 gerçek patch'te 0 hata. Görsel: `raporlar/img/boya_once_sonra.jpg`.
Kısa ViT deneyi (yok/var) henüz yapılmadı.

## 5. Karar ve öneri

- Varsayılan boyut: **256 px @ 0,5 µm/px (20x)** (CLAM/UNI yaygın pratiği; 224'e yeniden
  boyutlanır).
- Varsayılan yöntem: **henüz karar yok.** Gerçek CAMELYON16 (RUMC vs UMCU) ve TCGA (farklı TSS)
  ölçümü + 3 seed'li kısa ViT deneyi sonrası seçilecek. Ön beklenti (literatür, doğrulanmalı):
  HED augmentation ile normalizasyonsuz eğitim çoğu zaman normalizasyon kadar iyidir; bu yüzden
  **augmentation her zaman açık**, normalizasyon ölçüme göre.
- Hız: normalizasyon eğitimde darboğaz olursa `precompute` seçeneği (patch'leri bir kez
  normalize edip PNG/HDF5'e yazmak) eklenir.

## 6. Doğrulama (bitti kriteri)

- [x] Birim testi (sentetik test düzeneği, gerçek patch değil): Macenko iki boyalı karışımı neredeyse birebir geri üretiyor; tarayıcı farkı (döndürülmüş boya vektörleri) %50+ azalıyor (`tests/test_histo_stain.py`).
- [x] Birim testi (sentetik test düzeneği): Reinhard LAB istatistiklerini < 3 birim hatayla eşliyor; gerçek patch'te ölçülmedi.
- [ ] Gerçek merkezlerde renk istatistiği farkı azaldı (sayısal, bu belgeye).
- [ ] 50 rastgele augmentation örneği gözle doku anlamını bozmuyor.
- [ ] Etki (yok/var) en az 3 seed ile raporlandı.
