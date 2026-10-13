# Sprint 1 Retrospektifi — Nisa (veri hattı, Modül 1, altyapı)

> Proje liderinin `retrospektif.md` tablosundaki "Nisa" satırı için girdi. Ekip toplantısında
> birlikte okunur; diğer üyelerin görüşleri burada yoktur.

## Ne yapıldı (1.1-1.10 özeti)

| Mini sprint | Çıktı | Durum |
|---|---|---|
| 1.1 GPU/maliyet | `docs/gpu_strateji.md`: kaynaklı fiyat tablosu, 645 GPU-saat tahmini, 200 USD/ay sınır, kredi takvimi, kota şablonu | ekip onayı bekliyor |
| 1.2 Ortam | `infra/gpu/` (Dockerfile, compose, kurulum betiği, auto-shutdown), `scripts/verify_env.py`, `docs/gpu_ortam_rehberi.md` | kod hazır; bulutta denenmedi |
| 1.3 patch_v0 | `models/histo/patches/sample_pack.py`, `scripts/make_sample_pack.py`, teslim notu | hat hazır; gerçek paket WSI bekliyor |
| 1.4 TCGA slayt | barkod/envanter/seçim modülü, `scripts/download_tcga_slides.py`; **canlı GDC**: 2.133 DX slayt | listeleme yapıldı; indirme yapılmadı |
| 1.5 CAMELYON | XML→maske, resmi bölme, S3 indirme (resume), PAIP notu; **canlı S3** manifest | indirme yapılmadı |
| 1.6 QC | doku maskesi v1, QC metrikleri/karar, rapor betiği | eşikler kalibre edilmedi |
| 1.7 Extraction | HDF5 koordinat hattı (paralel, resume, hata toleranslı), koordinat sistemi belgesi, sözleşme köprü testi | gerçek slaytta performans ölçülmedi |
| 1.8 Boya | Macenko/Reinhard, HED augmentation, TTA, karşılaştırma betiği | gerçek merkez ölçümü yok |
| 1.9 ViT baseline | eğitim hattı (AMP, layer-decay, erken durdurma, MLflow), metrikler+bootstrap, model kartı taslağı | gerçek eğitim yok |
| 1.10 Kapanış | veri kartları (3), `dvc.yaml`, sızıntı testi/betiği, MkDocs iskeleti, `references.bib`, bu belge, Sprint 2 hazırlık | DVC remote ekip kararı bekliyor |

Testler: 159 test (ekip testleri dahil) geçiyor; `ruff`, `black`, `mypy --strict` temiz.

## Devam (iyi çalışanlar)

- Sentetik piramit slaytlar: gerçek veri olmadan tüm hattı test edebilmek, CI'da da çalışır.
- Sözleşme-önce: HDF5 koordinat dosyası proje liderinin doğrulayıcısıyla köprü testinden geçiyor.
- Betiklerde "dry-run → limit → tam" sırası ve kesintiden devam (indirme, çıkarım).
- Sayıları canlı kaynaktan okuyup belgeye kaynakla yazmak (GDC, S3, fiyat sayfaları).

## Başla

- **Gerçek veriyi erken almak:** Sprint 1'de tek bir gerçek slayt bile açılmadı; 10 CAMELYON
  slaydı (~15 GB) ilk haftada indirilmeliydi. Sprint 2'nin ilk işi.
- QC eşiklerini ve boya kararını gerçek veride ölçmeden "varsayılan" dememek (belgelerde askıda).
- GPU makinesini Sprint 1 ortasında kiralayıp kurulum betiğini bir kez gerçekten çalıştırmak.
- CODEOWNERS'a GitHub kullanıcı adımı eklemek; `models/histo/` için sahiplik.

## Bırak

- Çok sayıda modülü tek PR'da teslim etmek (bu PR büyük; Sprint 2'de mini sprint başına PR).
- `models.omics.gdc`'yi betik düzeyinde yeniden kullanmak yerine GDC transportunu
  `onkos/common`'a taşımayı ertelemek.

## Riskler (Sprint 2'ye taşınan)

| Risk | Olasılık | Etki | Önlem |
|---|---|---|---|
| WSI indirme süresi/diski (1,7 TB) | Y | Y | Önce 300 slayt alt kümesi; NVMe 2 TB; `limit` ile kademeli |
| GPU kotası/kredi gecikmesi | O | Y | RunPod/Lambda ön ödemeli; Colab Pro+ |
| Özellik çıkarımı beklenenden yavaş (OpenSlide I/O) | O | O | 8+ CPU, DataLoader işçileri, önbellek; UNI yerine ViT-S ile başla |
| CAMELYON mpp etiketi eksik → yanlış ölçek | O | Y | `mpp_override` + bir slaytta cetvel kontrolü (bilinen tümör çapı) |
| Temel model erişim onayı (UNI/CONCH gated) | O | O | Hemen başvur; Apache lisanslı Virchow/H-optimus-0/Prov-GigaPath yedek |
