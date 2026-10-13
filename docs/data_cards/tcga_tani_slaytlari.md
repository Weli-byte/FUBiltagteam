# Veri Kartı: TCGA Tanı Slaytları (LUAD, BRCA, COAD)

| Alan | İçerik |
|---|---|
| Amaç | Modül 1 (ViT + ABMIL) slayt düzeyi özellikler; proje liderinin omics/mutasyon verisiyle hasta bazında eşleştirme (mutasyon tahmini) |
| Kaynak | NCI Genomic Data Commons (GDC), TCGA programı; https://portal.gdc.cancer.gov ; API https://api.gdc.cancer.gov |
| Sürüm | GDC **Data Release 46.0 (10 Ağustos 2026)** (canlı sorgu, 13.10.2026) |
| Lisans / kullanım | Açık erişim (kimlik doğrulama yok); NIH GDS politikası; yeniden kimliklendirme yasak; yayınlarda TCGA teşekkür cümlesi zorunlu (https://gdc.cancer.gov/access-data/data-access-policies) |
| İçerik | Tanı (DX, FFPE, H&E) tam slayt görüntüleri, SVS (Aperio); LUAD 541 / BRCA 1.133 / COAD 459 = 2.133 slayt, 1.991 hasta, 1,86 TB; seçilen (hasta başına 1 birincil tümör DX) 1.990 slayt, 1,71 TB |
| Kimlik | Anonim TCGA barkodu (hasta: ilk 12 karakter); `slide_id` = GDC dosya UUID'si (proje liderinin `slide_file_id`) |
| Toplama | GDC API `/files` filtreleri (`access=open`, `data_type=Slide Image`, `experimental_strategy=Diagnostic Slide`), `scripts/download_tcga_slides.py`; md5 doğrulamalı indirme |
| İşleme | Barkod ayrıştırma, seçim kuralı (`docs/histo/tcga_slayt_indirme.md` §3), QC (1.6), doku maskesi, patch koordinatları (1.7), özellikler (Sprint 2) |
| Kalite | Tarayıcı/mpp slayta göre değişir (20x/40x); kalem izi, katlanma, bulanıklık QC ile işaretlenir; QC kalibrasyonu gerçek slaytlarda yapılacak |
| Bölme | Proje liderinin hasta düzeyi `splits-v1` (`slide_splits.parquet`); patch'ler slaytın kümesini alır; `scripts/check_split_compat.py` |
| Sınırlılıklar | Slayt düzeyi etiket yok (tümör slaydı = zayıf etiket); merkez (TSS) etkisi; FFPE artefaktları; **indirme henüz yapılmadı** |
| Depolama | `data/raw/tcga_slides/<proje>/slides/<uuid>/` (git'te değil; DVC ile yalnızca küçük türevler) |
| Atıf | TCGA Research Network: https://www.cancer.gov/tcga ; GDC: Grossman RL ve ark., NEJM 2016 (doğrulanmalı) |
| Sorumlu | Nisa |
