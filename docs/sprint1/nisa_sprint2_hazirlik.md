# Sprint 2 Hazırlık Listesi — Nisa (özellik çıkarımı ve ABMIL)

> Proje liderinin `sprint2_baslangic_listesi.md` "Nisa" satırını açar. Sıra önem sırasıdır.

## Hafta 1 (engel kaldırma)

- [ ] **GPU makinesi** (RunPod/Lambda A100 veya L4; 2 TB NVMe) kirala, `infra/gpu/setup_gpu_machine.sh`
      çalıştır, `verify_env.py --require-cuda` çıktısını `docs/gpu_ortam_rehberi.md` §7'ye yaz.
- [ ] **CAMELYON16 indirme**: önce `download.limit=10`, sonra tamamı (0,75 TB); `checksums.md5`
      ile kontrol; 20 slaytta maske overlay'i gözle kontrol → `camelyon_veri_notu.md` §3.
- [ ] **TCGA**: proje liderinden `patient_map.parquet` al → `download_tcga_slides.py` yeniden
      (kesişim sayıları) → ilk 300 slayt (`selection.limit=300`, 3 proje dengeli).
- [ ] **Temel model erişimi** (Hugging Face, hepsi "gated"; 13.10.2026): UNI / UNI2-h / CONCH
      (MahmoodLab, CC-BY-NC-ND 4.0, otomatik onay), Virchow (paige-ai, Apache-2.0),
      Virchow2 (CC-BY-NC-ND), Prov-GigaPath (Apache-2.0), H-optimus-0 (bioptimus, Apache-2.0),
      H-optimus-1 (manuel onay), Phikon/Phikon-v2 (Owkin, ticari olmayan lisans). **Başvur:**
      UNI, CONCH, Virchow, H-optimus-0. Lisans notu: NC-ND lisanslar TEKNOFEST/TÜBİTAK için
      sorun değil (araştırma) ama ürünleştirmede kısıt; raporda belirt.
- [ ] Kota/kredi: Azure for Students, GitHub Student Pack (herkes); Google Research Credits
      (danışman); AWS/GCP kota talebi şablonu (`gpu_strateji.md` §6).
- [x] CODEOWNERS'a `/models/histo/`, `/configs/histo/`, `/docs/histo/`, `/infra/gpu/` için `@Nananisnana` eklendi (bu dalda).

## Hafta 2-3 (gerçek veriyle Sprint 1'in açık kalan doğrulamaları)

- [ ] `patch_v0` gerçek paketi üret, 30 patch gözle kontrol, DVC ile paylaş; ekibe 5 dk özet.
- [ ] QC eşik kalibrasyonu: 50 slayt manuel inceleme, uyum % → `wsi_qc.md` §4.
- [ ] Extraction performansı: 10 gerçek slaytta slayt/dakika → `koordinat_sistemi.md` §5.
- [ ] Boya normalizasyonu gerçek ölçümü (RUMC vs UMCU) + 3 seed kısa ViT → `boya_normalizasyonu.md`.
- [ ] ViT-S vs ResNet-50 gerçek sonuç tablosu + model kartı (3 seed).
- [ ] `check_split_compat.py` liderin `slide_splits.parquet`'iyle çalıştır (kesişim 0).

## Sprint 2 ana işler (Nisa planı 2.1-2.10)

- Görüntü omurgası seçimi (ViT-S ImageNet vs UNI/Virchow; aynı patch seti, aynı bölme).
- Özellik çıkarımı hattı: koordinat HDF5 → `features (N, D) float16` + `encoder_name/version`
  (sözleşme); slayt başına dakikalar; resume; DVC.
- ABMIL (Ilse 2018) slayt düzeyi karar + dikkat ağırlıkları (`attention (N,)`, toplam 1) →
  Yusuf'un XAI'si; `slide_embedding (D,)` → proje liderinin cross-attention'ı.
- Patch veri paketi → Diffusion (proje lideri) için tümör patch'leri listesi.
- DVC remote (ekip kararı: Google Drive) ve `dvc push` düzeni; `.dvc/config` PR'ı.

## Açık sorular (ekibe)

1. TCGA'da hasta başına tek DX slayt yeterli mi, yoksa tüm DX slaytlar (ABMIL torbasına) mı? (öneri: önce tek)
2. Hedef çözünürlük 20x (0,5 µm/px) kalsın mı? UNI 20x ile eğitildi; CONCH de. (öneri: 20x)
3. GDC transport yardımcısını `onkos/common`'a taşıyalım mı? (sözleşme PR'ı, lider onayı)
4. DVC remote: Google Drive mı, S3 uyumlu ücretsiz depo mu?
