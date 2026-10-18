# Sprint 1 Gerçek Veri Raporu (Nisa, 18.10.2026)

> Bu rapordaki her sayı bu makinede gerçek slaytlarla üretilen çıktılardan (parquet/CSV/JSON)
> okunmuştur; sentetik veri yoktur. Ham tablolar: `sprint1_gercek_veri_ham_tablolar.md`
> (`collect_report.py` çıktısı). Küçük örnek (13 slayt) hattın gerçek veriyle çalıştığını
> gösterir; **bilimsel sonuç değildir** ve "NİHAİ DEĞİL" işaretli bölümler Sprint 2'de tam veriyle
> yinelenecektir. Büyük çıktılar `data/` ve `outputs/sprint1_real/` altında kalır (git'e girmez).

## 0. Ortam

- Dizüstü: Windows 11, Python 3.11.16, RTX 4060 Laptop (8 GB), sürücü 581.57.
- Ana ortam (`uv.lock`): torch 2.14.1+cpu. GPU koşusu için ayrı ortam: torch 2.14.1+cu126
  (`uv pip install --index-url https://download.pytorch.org/whl/cu126`); `verify_env.py --require-cuda
  --slide normal_108.tif` geçti (CUDA 12.6, fp16 matmul 245 ms, OpenSlide 4.0.1 gerçek slaytı açtı).
- Ağ: GDC ≈ 0,6 MB/s, CAMELYON S3 ≈ 1,1 MB/s (20 MB aralık isteğiyle ölçüldü); bu yüzden küçük örnek.
- MLflow: 6 eğitim koşusu `mlflow.db`'ye yazıldı (`histo-patch_<model>-<seed>`), config ve git hash ile.

## 1. Veri seçimi ve indirme (4a, 4f)

**Kural (deterministik):** her TCGA kohortundan, commit'li `TCGA-<proje>_dx_slides_selected.tsv`
manifestindeki 100–500 MB bandındaki ilk 3 dosya (dosya kimliği sırasıyla); CAMELYON16'dan tam
anotasyonlu en küçük 2 tümör slaydı (`tumor_091`, `tumor_084`) ve en küçük 2 normal slayt
(`normal_108`, `normal_150`) + 160 anotasyon XML + `evaluation/reference.csv`.
Listeler: `data/manifests/sprint1_gercek_ornek_slaytlar.csv` (QC/çıkarma) ve
`sprint1_gercek_ornek_pack.csv` (paket); göreli yollar CSV klasörüne göre çözülür.

| Kaynak | Slayt | Boyut | Doğrulama | Süre |
|---|---|---|---|---|
| TCGA (LUAD 3, BRCA 3, COAD 3) | 9 | 3,2 GB | md5: 9/9 OK (`data/interim/tcga_slides_download_log.csv`, `failed=0`) | ~2,5 sa (2 paralel) |
| CAMELYON16 | 4 görüntü + 160 XML (83 MB) + reference.csv | 2,4 GB | boyut doğrulaması (çok parçalı ETag md5 değildir), `failed=0` | ~45 dk |

Komutlar: `scripts/download_tcga_slides.py --from-manifests --download selection.per_project_limit=3
selection.min_size_mb=100 selection.max_size_mb=500` ve `scripts/download_camelyon.py
download.include=[annotations,evaluation,images] download.only=[tumor_091,normal_108,tumor_084,normal_150]`.

## 2. Slayt açma (4b)

13/13 slayt OpenSlide ile açıldı (tam tablo ham tablolarda). Örnekler:

| Slayt | Boyut (px) | Seviye | mpp | Büyütme | Vendor |
|---|---|---|---|---|---|
| TCGA-LUAD `004d8238…` (TCGA-75-7030 DX1) | 65963 x 27980 | 3 (1, 4, 16) | 0,5005 | 20 (aperio.AppMag) | aperio |
| TCGA-LUAD `00efe162…` (TCGA-86-6562 DX1) | 87291 x 55135 | 4 | 0,2520 | 40 | aperio |
| TCGA-COAD `023ca8d1…` (TCGA-A6-6653 DX1) | 160194 x 33419 | 4 | 0,2520 | 40 | aperio |
| CAMELYON16 `normal_108` | 45056 x 35840 | 8 (1 … 128) | 0,2269 | 40 (mpp'den) | philips |
| CAMELYON16 `tumor_084` | 65536 x 86016 | 9 | 0,2263 | 40 | philips |

Bulgular: TCGA tanı slaytları 20x (mpp 0,50) veya 40x (0,25; bir BRCA slaydı 0,2325 ile 20 olarak
etiketli) ve yalnızca 3–4 piramit seviyesi; CAMELYON16 TIFF'leri Philips biçiminde, mpp etiketi **var**
(0,226) → `mpp_override` gerekmedi. `reference.csv` `center` sütunu merkez adı değil 0/1 kodudur.

## 3. QC ve doku tespiti (4c)

13 slayt: **12 use, 1 exclude**. Dışlanan: TCGA-BRCA `01c79d06…` (pen_fraction 0,064 > 0,05).
Thumbnail'de (`img/tcga_brca_qc_exclude_kalem.jpg`) belirgin bir kalem izi görünmüyor; slayt
yoğun yağ dokusu içinde koyu hücre kümeleri taşıyor, bu yüzden "siyah mürekkep" HSV aralığının
(koyu + doygunluğu düşük piksel) **olası bir yanlış pozitifi**. Eşik değiştirilmedi; kalibrasyon
protokolünde (`wsi_qc.md` §4) siyah aralığı için tam çözünürlükte manuel kontrol kalemi açıldı.
Doku oranı 0,16–0,60; bulanık kare oranı tüm slaytlarda 0; koyu alan ≤ %1,9.

Görsel kontrol (3 slayt: `img/normal_108_doku_maskesi.jpg`, `img/tcga_luad_20x_doku_maskesi.jpg`,
`img/tcga_luad_40x_doku_maskesi.jpg`): maske doku kesitini yakalıyor; soluk yağ/bağ dokusu
doygunluk tabanlı Otsu'nun dışında kalıyor (beklenen davranış).

**Kalibrasyon bulgusu:** Laplacian varyans medyanı tarayıcıya göre çok farklı: Philips/CAMELYON
22.000–34.000, Aperio/TCGA 940–4.000. Varsayılan bulanıklık eşiği (40) bu ölçekte anlamsız
derecede düşük; eşik tarayıcı bazında ve 50 slaytlık manuel incelemeyle kalibre edilmeli
(`wsi_qc.md` §4). Eşik **değiştirilmedi** (test performansına göre ayar yok).

## 4. Patch koordinat çıkarma (4d)

13/13 slayt, **67.116 koordinat**, slayt başına 0,82 sn (yalnızca koordinat aşaması: thumbnail +
maske + ızgara; patch okuma dahil değil), 73 slayt/dk, 4 işçi. Slayt başına 1.056–11.246 patch
(256 px @ 0,5 µm/px; ≥ %50 doku). Her dosya `validate_coords_h5` ile doğrulandı.

Lider sözleşmesiyle uyum: `coords` (N,2) int32 seviye-0, `schema_version`, `slide_id`,
`magnification` (= çıkarım büyütmesi 20), `patch_size_px` mevcut; `features` Sprint 2'de eklenir
(köprü testi `tests/test_histo_patches.py`). Dosyada slayt yolu/barkod yoktur; 40x slaytlarda
`patch_level0_px` = 508–564 px (mpp'ye göre), 20x'te 256.

## 5. Tümör maskeleri ve resmi bölme (4f)

- `tumor_084`: 54 çokgen, tümör/slayt alanı %0,61; `tumor_091`: 7 çokgen (6 Tumor + 1 Exclusion),
  %1,49. Bindirme görselleri (`img/tumor_084_tumor_maskesi.jpg`, `img/tumor_091_tumor_maskesi.jpg`)
  gözle kontrol edildi: kırmızı bölgeler lenf düğümü dokusunun içinde, cam/yağ üstünde değil.
- Gerçek XML grupları `Tumor` ve `Exclusion` (README'deki `_0/_1/_2` değil); parser ikisini de kabul eder.
- `data/manifests/camelyon16_splits.json` tam kova listesinden: eğitim 270 (111 tumor, 159 normal),
  test 129 (49 tumor, 80 normal; hepsi etiketli), merkez kodları 0/1. Eğitim/test kesişimi 0.

## 6. Örnek paket (patch_v0), boya karşılaştırması (4e)

Paket: 2.600 patch / 13 slayt; CAMELYON16 800 (200 tümör + 600 normal; tümör slaytlarında 100/100
dengeli), TCGA 1.800 (slayt düzeyi zayıf etiket "tümör", `split=unassigned` → eğitime girmez).
Görsel: `img/camelyon_patch_ornekleri.jpg` (üst 2 satır tümör: epitelyal/glandüler hücre grupları;
alt 2 satır normal: yoğun lenfosit, yağ) — etiketler tutarlı.

Boya: 7 "merkez" (6 TCGA TSS + CAMELYON center1), 100 patch/merkez, 3 seed:

| yöntem | merkezler arası LAB farkı | merkez sınıflandırıcı doğruluğu (ort ± sd) | şans |
|---|---|---|---|
| none | 6,47 | 0,676 ± 0,007 | 0,143 |
| macenko | 2,29 | 0,419 ± 0,010 | 0,143 |
| reinhard | 0,22 | 0,489 ± 0,022 | 0,143 |

Yorum: her iki yöntem merkezler arası renk farkını azaltıyor ve merkezi renkten tahmin etmeyi
zorlaştırıyor (0,676 → 0,42/0,49); hiçbiri şansa (0,14) indirmiyor. **Sınırlılık:** "merkez" burada
kanser türü/doku ile karışık (akciğer, meme, kolon, lenf düğümü); farkın bir kısmı biyolojiktir,
tarayıcı değil. Macenko 700 patch'te 0 hata (yeterli doku). Önce/sonra görseli
`img/boya_once_sonra.jpg` (sütunlar: none, macenko, reinhard). Varsayılan yöntem kararı için gerçek
çok merkezli (CAMELYON17 / TSS'ye göre eşlenmiş aynı doku) ölçüm gerekir.

## 7. Kısa ViT-S/16 vs ResNet-50 koşusu (4g) — NİHAİ DEĞİL

Veri: yalnızca CAMELYON16 eğitim slaytları (resmi test kilitli): eğitim 2 slayt (1 tümörlü + 1 normal,
400 patch: 100 tümör / 300 normal), doğrulama 2 slayt (400 patch, aynı oran). Tarif:
`configs/histo/patch_classifier.yaml` ile 8 epoch, batch 32, AMP, 3 seed, GPU (RTX 4060).

| model | AUROC (3 seed ort, %95 GA) | AUPRC | F1@0,5 | doğruluk |
|---|---|---|---|---|
| vit_small | 0,780 [0,729, 0,831] | 0,426 | 0,057 | 0,745 |
| resnet50 | 0,769 [0,659, 0,879] | 0,453 | 0,169 | 0,738 |

Seed bazında AUROC: ViT 0,740 / 0,770 / 0,829; ResNet 0,803 / 0,845 / 0,659 (ham tablolarda GA ile).
Okuma: iki modelin güven aralıkları örtüşüyor, **fark anlamlı değil**; "ViT %8-15 daha iyi" iddiası
bu örnekle desteklenmiyor (ve çürütülmüyor: 2+2 slayt). F1@0,5'in düşüklüğü modelin tümör
olasılıklarını 0,5'in altında tutması (sınıf oranı 1:3, 8 epoch); AUROC sıralamayı ölçer. ViT
eğitim kaybı 0'a inerken doğrulama kaybı artıyor (`img/ogrenme_egrileri.jpg`): 400 patch'te
ezberleme. Bu koşu yalnızca hattın (veri yükleme, AMP, checkpoint, MLflow, bootstrap GA) gerçek
veride çalıştığını gösterir; model kartı sonuç satırı "ön bulgu" olarak işaretlendi.

## 8. Yapılamayanlar ve sınırlılıklar

- 13 slayt küçük bir örnektir; QC eşik kalibrasyonu (50 slayt manuel inceleme) ve tam ölçekli
  performans ölçümü (patch okuma + özellik çıkarımı) yapılmadı.
- Liderin `patient_map.parquet` ve `slide_splits.parquet` dosyaları gelmediği için TCGA kesişimi
  ve `check_split_compat.py` çalıştırılmadı (bekleyen iş: `docs/sprint1/nisa_sprint2_hazirlik.md`).
- Boya ölçümünde merkez ve doku türü karışık; ViT/ResNet sonucu 2+2 slaytla ön bulgu.
- Büyük çıktılar (slaytlar, HDF5, PNG paketi, checkpoint'ler) yalnızca bu makinede; DVC remote
  henüz ekip kararı bekliyor.
