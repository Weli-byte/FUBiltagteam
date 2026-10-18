# Sprint 2 başlangıç kontrol listesi

> Kapsam: yalnızca Sprint 2. Bu liste Sprint 1'den **bilerek Sprint 2'ye bırakılan** işleri ve Sprint 2 hazırlığını içerir.

## Sprint 1 kapanış durumu
- Kod ve belgeler 1.1–1.10 tamamlandı, `claude/kind-euler-8kj48k` dalında.
- Gerçek veride doğrulananlar: indirme (1.3), ön işleme ve QC (1.4), etiketler (1.5), bölmeler ve sızıntı testi (1.6, 25/25), ön temel model (1.8, `hafif1`: yalnızca TP53, nihai değil).
- Raporlar: `veri/omics-etiket-raporlari`, `veri/omics-bolme-baseline-raporlari` dalları.

## Sprint 2'ye bırakılanlar (sorumlu: proje lideri, aksi belirtilmedikçe)
- [ ] **Tam `train_baselines` koşusu** (tüm genler/projeler/modeller). Önerilen yol: Colab (`docs/colab_calistirma.md`, denenmedi).
      Sonuçlarla `docs/sprint1/eda_raporu.md` bölüm 5 güncellenecek.
- [ ] `gbm` adımının yavaşlığının nedeni (500 gen/1609 örnekte ~28 dk): profil çıkar. Bellek mi, ayar mı: kanıtlanmadı.
- [ ] DVC remote (Google Drive) kurulumu ve `data/raw/gdc` için sürümleme. Ham veri GDC'den yeniden indirilebilir, bu yüzden acil değil.
- [ ] `data/interim` ve `data/processed` için ayrı yedek (küçük klasörler).
- [ ] cBioPortal ile 5 pozitif hasta kontrolü (EGFR p.E746_A750del TCGA-49-4501, EGFR p.L858R TCGA-86-8668, KRAS p.G12D TCGA-AA-3861, KRAS p.G12V TCGA-AY-A71X, TP53 p.C141Y TCGA-A8-A081).
- [ ] 1.9 kaynak listesinin doğrulanması (18 kaynak, hiçbiri doğrulanmadı).
- [ ] COAD'daki 18 kütüphane-boyutu bayrağının nedeni (15'i AA merkezinden, doğrulanmadı).
- [ ] PR #5 (manifestler) ve PR #3 (veri sözleşmesi) için ekip onayları; kod dalı için `main` PR'ı.
- [ ] Retrospektif tablosunun ekip sütunları (`retrospektif.md`).

## Kişi bazlı hazırlık
- **Nisa:** GPU/CUDA; WSI indirme için `slide_splits.parquet` kimlik sözleşmesi (`docs/bolme_kullanim_rehberi.md`); HDF5 patch sözleşmesine uyum.
- **İdil:** API ve GNN için `onkos.contracts` şemalarını gözden geçir; CODEOWNERS'a GitHub kullanıcı adını ekle.
- **Yusuf:** Arayüz metinlerinde araştırma uyarısı ve terim ayrımı; sözleşmeden tüketen taslak.
- **Proje lideri:** Omics encoder'ı temel modellerle kıyaslanabilir kur; sızıntı kurallarını koru.

## Sprint 2 kapısı
Test kümesine yalnızca `final_evaluation=True` ile ve ekip kararıyla dokunulur. Sprint 2 kapsamı dışına çıkılmaz.
