# Sprint 2 Hazırlık Listesi — Nisa (özellik çıkarımı ve ABMIL)

> Proje liderinin `sprint2_baslangic_listesi.md` "Nisa" satırını açar. Sıra önem sırasıdır.

## Hafta 1 (engel kaldırma)

- [ ] **GPU makinesi** (RunPod/Lambda A100 veya L4; 2 TB NVMe) kirala, `infra/gpu/setup_gpu_machine.sh`
      çalıştır, `verify_env.py --require-cuda` çıktısını `docs/gpu_ortam_rehberi.md` §7'ye yaz.
- [ ] **CAMELYON16 indirme**: önce `download.limit=10`, sonra tamamı (0,75 TB); `checksums.md5`
      kova kökünde var mı önce bakılacak (öneklerle listelemede görünmedi), varsa md5 kontrolü;
      20 slaytta maske overlay'i gözle kontrol → `docs/histo/camelyon_veri_notu.md` §3.
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

- [x] (küçük örnek, 18.10.2026) `patch_v0` gerçek paketi: 13 slayt, 2.600 patch; 32 patch gözle
      kontrol edildi (`docs/histo/raporlar/img/camelyon_patch_ornekleri.jpg`). **Kalan:** ~50 slaytlık
      paket ve DVC paylaşımı (remote kararı), ekibe 5 dk özet.
- [ ] QC eşik kalibrasyonu: 50 slayt manuel inceleme, uyum % → `docs/histo/wsi_qc.md` §4. (18.10
      bulgusu: Laplacian ölçeği tarayıcıya bağlı; "siyah mürekkep" aralığı BRCA slaydında olası
      yanlış pozitif, tam çözünürlükte kontrol açık madde.)
- [x] Extraction performansı (koordinat aşaması, ÖN BULGU): 13 gerçek slayt, işçi başına 0,82
      sn/slayt (seri eşdeğeri 73 slayt/dk; 4 işçiyle 13 slayt ≈ 3 sn) → `docs/histo/koordinat_sistemi.md`
      §5. **Kalan:** patch okuma/özellik çıkarımı hızı (Sprint 2).
- [x] Boya normalizasyonu gerçek ölçümü (ÖN BULGU: 7 merkez, merkez-doku karışık, hedef patch rastgele)
      → `docs/histo/boya_normalizasyonu.md` §4. **Kalan:** temsili hedef patch seçimi, aynı doku/çok
      merkez (CAMELYON17) ölçümü ve 3 seed kısa ViT yok/var deneyi.
- [x] ViT-S vs ResNet-50 (3 seed, 2+2 slayt) ön bulgu → `docs/histo/vit_baseline_v1.md` §5, model kartı.
      **Kalan:** ≥ 20 slaytla nihai karşılaştırma (NİHAİ DEĞİL).
- [ ] `check_split_compat.py` liderin `slide_splits.parquet`'iyle çalıştır (kesişim 0) — dosya gelmedi.

## Sprint 2 ana işler (Nisa planı 2.1-2.10)

- Görüntü omurgası seçimi (ViT-S ImageNet vs UNI/Virchow; aynı patch seti, aynı bölme).
- Özellik çıkarımı hattı: koordinat HDF5 → `features (N, D) float16` + `encoder_name/version`
  (sözleşme); slayt başına dakikalar; resume; DVC.
- ABMIL (Ilse 2018) slayt düzeyi karar + dikkat ağırlıkları (`attention (N,)`, toplam 1) →
  Yusuf'un XAI'si; `slide_embedding (D,)` → proje liderinin cross-attention'ı.
- Patch veri paketi → Diffusion (proje lideri) için tümör patch'leri listesi.
- DVC remote (ekip kararı: Google Drive) ve `dvc push` düzeni; `.dvc/config` PR'ı.

## Bekleyen iş: proje liderinden gelecek dosyalar (Adım 5, 2026-10-18)

| Dosya | Nereden | Ne için | Durum |
|---|---|---|---|
| `data/interim/patient_map.parquet` | Liderin 1.3 betiği (`scripts/download_gdc.py`) | TCGA slayt seçimini omics+MAF'ı olan hastalarla kesiştirmek (`download_tcga_slides.py`, `patients_in_leader_list` sütunları) | **gelmedi** (parquet git'e girmez; DVC remote henüz yok) |
| `data/processed/splits/<sürüm>/slide_splits.parquet` | Liderin 1.6 betiği (`scripts/make_splits.py`) | Patch/slayt tablolarının hasta düzeyi bölmeyle uyumu: `scripts/check_split_compat.py --slide-splits ... --coords-dir ...` | **gelmedi** |

Kural: bu dosyalar bu dalda **üretilmez** (liderin kimlik sözleşmesi); gelene kadar
`check_split_compat.py` yalnızca birim testleriyle (`tests/test_histo_split_leakage.py`) doğrulanmış
sayılır. Dosyalar gelince komut çalıştırılır, çıktı `docs/histo/raporlar/` altına eklenir ve
`docs/data_cards/tcga_tani_slaytlari.md` "Bölme" satırı güncellenir.

## Açık sorular (ekibe)

1. TCGA'da hasta başına tek DX slayt yeterli mi, yoksa tüm DX slaytlar (ABMIL torbasına) mı? (öneri: önce tek)
2. Hedef çözünürlük 20x (0,5 µm/px) kalsın mı? UNI 20x ile eğitildi; CONCH de. (öneri: 20x)
3. GDC transport yardımcısını `onkos/common`'a taşıyalım mı? (sözleşme PR'ı, lider onayı)
4. DVC remote: Google Drive mı, S3 uyumlu ücretsiz depo mu?
