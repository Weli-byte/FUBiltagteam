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
**En kolay yol (Claude Code gerekmez):** PowerShell'de depo klasöründe tek komut:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local_pipeline.ps1
```
Betik 1.4 ve 1.5'i sırayla çalıştırır, tüm çıktıyı `logs\pipeline_<tarih>.txt` dosyasına kaydeder
(paylaşabilirsin). **Hiçbir dosyayı silmez/taşımaz/üzerine yazmaz, hiçbir işlemi sonlandırmaz, sistem
ayarı değiştirmez, git'e bir şey eklemez.** Çıktı klasörleri doluysa üzerine yazmadan durur.

Adım adım (elle):
```bash
uv sync --group data --group omics
# 0) Dosya biçimini doğrula (ilk gerçek dosyada; sütunlar beklenenden farklıysa hata verir)
uv run python scripts/preprocess_omics.py inspect <data/raw/gdc/TCGA-LUAD/rnaseq/<id>/<dosya>.tsv>
# 1) Matris + QC (öğrenme yok); tepe özel bellek ~0,5 GB, süre disk hızına bağlı
uv run python scripts/preprocess_omics.py build
# 2a) Keşif amaçlı (model eğitiminde KULLANMA):
uv run python scripts/preprocess_omics.py fit-transform --eda-all-samples
# 2b) Gerçek kullanım (1.6'dan sonra): yalnızca eğitim hastalarının listesiyle
uv run python scripts/preprocess_omics.py fit-transform --train-ids data/interim/train_ids.txt
```
Çıktılar `data/processed/omics/` altında (git'e girmez, DVC):
- `expression_tpm.npy` (örnek x gen, float32, **diskte**; bellek-eşlemeli okunur),
  `expression_tpm_obs.parquet` (örnek tablosu + QC bayrakları), `expression_tpm_var.parquet`
  (gen sözlüğü), `expression_tpm_meta.json`, `sample_qc.parquet`
- `expression_<all|protein_coding>_top<k>.h5ad` (z-skorlu, seçilmiş genler), `preprocessor_*.joblib`

Rapor ve şekiller `docs/omics_qc/` altında (toplu sayılar ve grafikler; hasta kimliği içermez).
Opsiyonel yol/gen kümesi skorları: `--gene-sets <dosya.gmt>` (ör. MSigDB Hallmark; indirmek ve
lisansını doğrulamak sana ait).

**Üzerine yazma yoktur:** `build` ve `fit-transform`, çıktı dosyası zaten varsa `FileExistsError` ile
durur. Yeniden çalıştırmak için eski çıktıyı kendin başka yere taşı/yeniden adlandır.

## Bellek az olan bilgisayarlarda
Matris RAM'e yüklenmez; diskte tutulup gen blokları ve satır parçaları halinde okunur. Sentetik
2.100 x 60.660'lık ölçümde **tepe özel (private) bellek ~0,5 GB**, ek olarak ~1 GB dosya-eşlemeli
sayfa vardır (bunlar bellek baskısında sistem tarafından geri alınabilir). Sonuç olarak az boş RAM'li
bir bilgisayarda da çalışır, yalnızca disk okuması nedeniyle daha yavaş olur.
1. **"Kullanılabilir" RAM'e değil taahhüt payına (commit headroom) bak.** Çalıştırmanın başarılı olup
   olmayacağını RAM + sanal bellek toplamı belirler. `run_local_pipeline.ps1` bunu dilden bağımsız
   WMI ile okur (Türkçe Windows'ta `Get-Counter '\Memory\...'` çalışmaz) ve pay 3000 MB altındaysa durur.
2. **Pay yetersizse sanal belleği büyüt:** Ayarlar → Sistem → Hakkında → Gelişmiş sistem ayarları →
   Performans → Ayarlar → Gelişmiş → Sanal bellek → Değiştir → özel boyut (başlangıç 8192 MB,
   en büyük 24576 MB). Bu "kullanılabilir RAM" sayısını yükseltmez, taahhüt payını yükseltir.
3. **Claude Code içinde ağır işleri çalıştırma:** Claude Code bellek azaldığında alt işlemleri durdurabilir
   (indirmede yaşandı). Yukarıdaki PowerShell betiğini doğrudan terminalde çalıştır.
4. **Yedek:** projeleri tek tek çalıştır (`build --projects TCGA-LUAD --out data/processed/omics_LUAD
   --report-dir docs/omics_qc/LUAD`); havuzlu matris gerekiyorsa sonra tek seferde kurulur.

## Sınırlılıklar
- STAR-Counts sütun adları belgelerden yazıldı; `inspect` ilk gerçek dosyada doğrular.
- Ölçek (2.100 örnek x 60.660 gen) sentetik matrisle ölçüldü (Linux, `/proc` ile): tepe özel bellek
  ~0,5 GB; build adımlarının toplamı soğuk diskte ~45 sn, iki fit-transform varyantı ~13 sn. Windows'ta
  ve gerçek dosya okuma süresiyle (4.000+ dosya) değerler farklı olabilir; yerel olarak doğrulanmalı.
- Yol (KEGG/Hallmark) gruplaması kod olarak vardır ama gen kümesi dosyası repoda yoktur.
- `joblib` ile kaydedilen model dosyaları yalnızca güvendiğiniz kaynaklardan yüklenmelidir.
