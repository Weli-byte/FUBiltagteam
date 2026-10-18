# Sprint 1 Retrospektifi — Nisa (veri hattı, Modül 1, altyapı)

> Proje liderinin `retrospektif.md` tablosundaki "Nisa" satırı için girdi. Güncelleme 18.10.2026:
> gerçek veriyle yapılan doğrulamalar işlendi (`docs/histo/raporlar/sprint1_gercek_veri_raporu.md`).
> Diğer üyelerin görüşleri burada yoktur.

## Ne yapıldı (1.1–1.10) ve gerçek veriyle doğrulama durumu

| Mini sprint | Çıktı | Gerçek veriyle doğrulandı mı |
|---|---|---|
| 1.1 GPU/maliyet | `docs/gpu_strateji.md`: kaynaklı fiyat tablosu, 645 GPU-saat tahmini, 200 USD/ay sınır, kredi takvimi, kota şablonu | Belge; ekip onayı bekliyor |
| 1.2 Ortam | `infra/gpu/` (Dockerfile, compose, kurulum, auto-shutdown), `scripts/verify_env.py`, `docs/gpu_ortam_rehberi.md` | **Kısmen**: dizüstü RTX 4060 + cu126 ortamında `verify_env --require-cuda --slide` gerçek slaytla geçti; bulut kurulumu denenmedi |
| 1.3 patch_v0 | `models/histo/patches/sample_pack.py`, `scripts/make_sample_pack.py`, teslim notu | **Evet (küçük)**: 13 gerçek slayt, 2.600 patch; 50 slaytlık paket tam indirme sonrası |
| 1.4 TCGA slayt | barkod/envanter/seçim, `download_tcga_slides.py` (+`--from-manifests`) | **Evet**: canlı GDC listeleme (2.133 DX), 9 slayt indirildi, md5 9/9; liderin hasta listesi kesişimi yok |
| 1.5 CAMELYON | S3 listeleme/indirme, XML→maske, resmi bölme, PAIP notu | **Evet**: 4 slayt + 160 XML; 2 tümör maskesi gözle doğrulandı; bölme belgesi 270/129 tam listeden |
| 1.6 QC | doku maskesi v1, QC metrikleri/karar, rapor | **Evet (13 slayt)**: 12 use / 1 exclude; maskeler 3 slaytta gözle; **eşik kalibrasyonu yapılmadı** (Laplacian ölçeği tarayıcıya bağlı) |
| 1.7 Extraction | HDF5 koordinat hattı (paralel, resume, config-hash) | **Evet**: 13/13 slayt, 67.116 koordinat, 0,82 sn/slayt; sözleşme köprüsü testle |
| 1.8 Boya | Macenko/Reinhard, HED augmentation, TTA, karşılaştırma | **Evet (ölçüm)**: 7 merkez, LAB farkı 6,47 → 2,29/0,22; merkez-doku karışıklığı; ViT yok/var deneyi yapılmadı |
| 1.9 ViT baseline | eğitim hattı (AMP, layer-decay, MLflow), metrik+bootstrap, model kartı | **Evet (ön bulgu, NİHAİ DEĞİL)**: ViT-S 0,780 vs ResNet-50 0,769 AUROC, 3 seed, 2+2 slayt; fark anlamsız |
| 1.10 Kapanış | veri kartları, dvc.yaml, sızıntı denetimi, MkDocs, references.bib, bu belge | `check_split_compat` liderin `slide_splits.parquet`'i gelmediği için **bekliyor**; DVC remote ekip kararı |

Kalite: 210 test (ekip testleri dahil) geçiyor; `ruff`, `black`, `mypy --strict` temiz; CI yeşil.
Sentetik slayt üreticisi yalnızca `tests/histo_synthetic_fixture.py` (test düzeneği); üretim kodu
ona bağımlı değil (koruma testi var).

## Devam (iyi çalışanlar)

- "Dry-run → limit → tam" sırası ve kesintiden devam eden indirme/çıkarma: 9 TCGA slaydı 0,6 MB/s'lik
  bağlantıda sorunsuz indi, md5 hepsinde tuttu.
- Sayıları canlı kaynaktan okuyup kaynakla yazmak (GDC, S3, fiyat sayfaları); raporda her sayı dosyadan.
- Sözleşme-önce: koordinat dosyası liderin doğrulayıcısıyla köprü testinden geçiyor; dosyada barkod yok.
- Bağımsız kod denetimi (iki tur): 416/resume, `download.only`, etiket doğrulama, config-hash
  gibi gerçek hatalar yakalandı ve testle kapatıldı.

## Başla

- Gerçek veriyi **ilk hafta** almak: ilk gerçek slayt Sprint 1'in son gününde açıldı; QC eşikleri ve
  boya kararı bu yüzden hâlâ askıda.
- QC eşiklerini tarayıcı bazında tutmak (Aperio vs Philips Laplacian ölçeği 10 kat farklı).
- Daha büyük örnekle (≥ 20 CAMELYON slaydı) ViT/ResNet karşılaştırmasını yinelemek; 400 patch'te ViT ezberliyor.
- Sentetik olan her şeyi baştan `tests/` altında tutmak (bu sprintte sonradan taşındı).

## Bırak

- Çok sayıda modülü tek PR'da teslim etmek (PR #6 büyük; Sprint 2'de mini sprint başına PR).
- Gerçek veri olmadan "uçtan uca çalıştı" demek: artık yalnızca gerçek veriyle doğrulananlar "evet".

## Riskler (Sprint 2'ye taşınan)

| Risk | Olasılık | Etki | Önlem |
|---|---|---|---|
| WSI indirme süresi (0,6 MB/s ile 1,7 TB ≈ 33 gün) | Y | Y | Bulut/veri makinesi ya da kurum ağı; önce 300 slayt alt kümesi |
| QC eşikleri kalibre değil (tarayıcıya bağlı) | Y | O | 50 slaytlık manuel inceleme, eşikler tarayıcı/vendor bazında |
| GPU kotası/kredi gecikmesi | O | Y | RunPod/Lambda ön ödemeli; dizüstü RTX 4060 küçük işler için yeterli |
| Özellik çıkarımı I/O darboğazı (patch okuma) | O | O | 8+ CPU, DataLoader işçileri; ölçüm Sprint 2 ilk hafta |
| Boya kararı merkez-doku karışıklığı | O | O | Aynı doku/çok merkez (CAMELYON17) ile ölçüm |
| Temel model erişimi (UNI/CONCH gated) | O | O | Hemen başvur; Apache lisanslı Virchow/H-optimus-0 yedek |
