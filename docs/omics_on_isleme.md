# Omics Ön İşleme: Tasarım Notu ve Çalıştırma (Mini sprint 1.4)

Girdi: GDC "STAR - Counts" RNA-seq dosyaları (`tpm_unstranded` + `unstranded` sayımlar) ve
`data/interim/patient_map.parquet` (1.3). Çıktı: temiz ifade matrisleri + QC raporu.

## İki aşama (neden?)
Plan "normalizasyon parametreleri yalnızca eğitim kümesinden öğrenilmeli" diyor, ama hasta bazlı bölme
(eğitim/doğrulama/test) 1.6'da yapılıyor. Bu yüzden ön işleme ikiye ayrıldı:

| Aşama | Komut | Veriden **öğrenir mi?** | Ne yapar |
|---|---|---|---|
| 1. `build` | `preprocess_omics.py build` | **Hayır** | Matrisi kurar (TPM), gen sözlüğü, örnek QC bayrakları, PCA/UMAP, batch etkisi |
| 2. `fit-transform` | `preprocess_omics.py fit-transform --train-ids ...` | **Evet, yalnızca eğitim örneklerinden** | Gen filtresi, log2(TPM+1), yüksek varyanslı gen seçimi, z-skor |

`fit-transform` bir eğitim listesi **veya** açıkça `--eda-all-samples` ister; ikisi birden ya da hiçbiri
hata verir. `--eda-all-samples` çıktıları `fit_on = ALL_SAMPLES_EDA_ONLY_NOT_FOR_MODEL_TRAINING` ile
damgalanır ve model eğitiminde kullanılmaz. Sızıntı testi (`tests/test_omics_*.py`): öğrenilen
ortalama/std/gen seçimi yalnızca eğitim satırlarından yeniden hesaplanıp karşılaştırılıyor; test
verisi değişince parametreler değişmiyor.

## Parametreler (`configs/omics/preprocess.yaml`) ve gerekçeleri
| Parametre | Değer | Kaynağı |
|---|---|---|
| gen ifade eşiği | TPM > 1, örneklerin ≥ %20'sinde | **Plan** (sprint planı 1.4); eğitim kümesinde uygulanır |
| dönüşüm | log2(TPM + 1), sonra gen bazlı z-skor | **Plan** |
| yüksek varyanslı alt kümeler | 2.000 / 5.000 / 10.000 gen | **Plan** |
| gen evreni | `all` ve `protein_coding` (iki sürüm) | **Plan** |
| uç değer eşiği | \|robust z\| > 4.0 (medyan ve MAD) | **Benim seçimim**, veriye bakılarak ayarlanmadı; 3-5 aralığı yaygındır. Bayraklı örnekler **silinmez**, karar `--drop-flagged` ile verilir |
| z-skor kırpma (`clip`) | kapalı | Varsayılan; ihtiyaç görülürse (ör. 5) açılır |
| PCA | en yüksek varyanslı 2.000 gen, 20 bileşen | Yalnızca görselleştirme/QC; model girdisi değildir |

Karar gerektiren konular **ekibe bırakılmıştır**: `all` mı `protein_coding` mi, kaç gen, bayraklı
örnekleri atma. Bunlar 1.8'deki baseline sonuçlarına ve **doğrulama kümesine** bakılarak seçilmeli,
test kümesine bakılarak değil.

## QC ve batch etkisi
- Örnek bayrakları: eksik/sonsuz değer, kütüphane büyüklüğü, tespit edilen gen sayısı, medyan profile
  düşük korelasyon.
- Batch etkisi: PCA uzayında silhouette skoru; kanser türüne göre (istenen sinyal) ve **proje içinde**
  doku kaynak merkezine (TSS, batch vekili) göre. Sayılar raporda **ölçüldüğü gibi** yazılır; "batch
  sınırlı" demek için önceden uydurulmuş eşik yoktur. Plan kriteri ("UMAP'te kanser türleri ayrışmalı,
  batch kümelenmesi sınırlı kalmalı") gerçek veriyle sonuçlara bakılarak değerlendirilir.
- Gen sembolü çakışmaları sessizce geçilmez: `symbol_is_duplicate` / `ensembl_is_duplicate` sütunlarıyla
  işaretlenir ve loglanır (pseudo-otozomal `_PAR_Y` genleri X kopyalarıyla aynı Ensembl kimliğini paylaşır).

## Çalıştırma (kendi bilgisayarında, veri `data/raw/gdc` altında)
```bash
uv sync --group data --group omics
# 0) Dosya biçimini doğrula (ilk gerçek dosyada; sütunlar beklenenden farklıysa hata verir)
uv run python scripts/preprocess_omics.py inspect <data/raw/gdc/TCGA-LUAD/rnaseq/<id>/<dosya>.tsv>
# 1) Matris + QC (öğrenme yok), tahmini 5-15 dk, tepe bellek ~1,5 GB
uv run python scripts/preprocess_omics.py build
# 2a) Keşif amaçlı (model eğitiminde KULLANMA):
uv run python scripts/preprocess_omics.py fit-transform --eda-all-samples
# 2b) Gerçek kullanım (1.6'dan sonra): yalnızca eğitim hastalarının listesiyle
uv run python scripts/preprocess_omics.py fit-transform --train-ids data/interim/train_ids.txt
```
Çıktılar `data/processed/omics/` altında (git'e girmez, DVC): `expression_tpm.h5ad`,
`expression_<all|protein_coding>_top<k>.h5ad`, `preprocessor_*.joblib`, `sample_qc.parquet`.
Rapor ve şekiller `docs/omics_qc/` altında (toplu sayılar ve grafikler; hasta kimliği içermez).
Opsiyonel yol/gen kümesi skorları: `--gene-sets <dosya.gmt>` (ör. MSigDB Hallmark; indirmek ve
lisansını doğrulamak sana ait).

## Bellek az olan bilgisayarlarda
Tepe bellek kullanımı yaklaşık **1,5 GB**'tır. Önce doğru şeye bak:
1. **"Boş" (free) değil "kullanılabilir" (available) belleğe bak.** Görev Yöneticisi → Performans →
   Bellek → "Kullanılabilir"; ya da PowerShell: `(Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue`.
   Windows önbelleği (standby) "boş" saymaz ama gerektiğinde hemen serbest bırakır. Kullanılabilir
   bellek ≥ 2 GB ise çalıştırmak güvenlidir.
2. **Hiçbir uygulamayı kapatamıyorsan sanal belleği (pagefile) büyüt:** Ayarlar → Sistem → Hakkında →
   Gelişmiş sistem ayarları → Performans → Ayarlar → Gelişmiş → Sanal bellek → Değiştir → özel boyut
   (başlangıç 8192 MB, en büyük 24576 MB). Yavaşlar ama bellek yetmezliğiyle düşmez.
3. **Projeyi tek tek çalıştır** (her proje < 0,3 GB):
   `build --projects TCGA-LUAD --out data/processed/omics_LUAD --report-dir docs/omics_qc/LUAD`
   (aynısını BRCA ve COAD için). Havuzlu (pan-cancer) matris gerekiyorsa sonra tek seferde kurulur.

## Sınırlılıklar
- STAR-Counts sütun adları belgelerden yazıldı; `inspect` ilk gerçek dosyada doğrular.
- Ölçek (2.100 örnek x 60.660 gen) için bellek/süre sentetik matrisle ölçüldü: tepe ~1,2 GB
  (matris 0,5 GB + ~0,55 GB geçici), ~12 sn işlem; gerçek dosya okuma süresi (disk) ayrıca eklenir.
  Tüm işlemler gen blokları halinde yapılır, tam boyutlu kopya üretilmez.
- Yol (KEGG/Hallmark) gruplaması kod olarak vardır ama gen kümesi dosyası repoda yoktur.
- `joblib` ile kaydedilen model dosyaları yalnızca güvendiğiniz kaynaklardan yüklenmelidir.
