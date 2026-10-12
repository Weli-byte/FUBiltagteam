# Sprint 2 başlangıç kontrol listesi

> Kapsam: yalnızca Sprint 2. Sonraki sprintler için bu dosya plan içermez.

## Girişte bitmiş olmalı (Sprint 1 kapanışı)
- [ ] `splits-v1` oluşturuldu, manifest SHA-256 doğrulandı (`make_splits.py --version splits-v1 --verify`)
- [ ] `train_baselines.py --quick`, ardından tam çalıştırma; sonuçlar `docs/baselines/` altında
- [ ] EDA raporu bölüm 5 dolduruldu
- [ ] PR #5 birleştirildi, PR #3 için onaylar tamam
- [ ] DVC Google Drive remote kuruldu, `data\raw\gdc` harici diske yedeklendi
- [ ] cBioPortal ile 5 pozitif hasta kontrolü; LUAD 585→517 fark nedeni
- [ ] Kaynak doğrulama turu başladı (1.9)
- [ ] Risk kaydı ve retrospektif toplantıda onaylandı

## Kişi bazlı hazırlık
- **Nisa:** GPU/CUDA; WSI indirme için `slide_splits.parquet` kimlik sözleşmesi (`docs/bolme_kullanim_rehberi.md`); HDF5 patch sözleşmesine uyum.
- **İdil:** API ve GNN için `onkos.contracts` şemalarını gözden geçir; CODEOWNERS'a GitHub kullanıcı adını ekle.
- **Yusuf:** Arayüz metinlerinde araştırma uyarısı ve terim ayrımı; sözleşmeden tüketen taslak.
- **Proje lideri:** Omics encoder'ı temel modellerle kıyaslanabilir kur; sızıntı kurallarını koru.

## Sprint 2 kapısı
Test kümesine yalnızca `final_evaluation=True` ile ve ekip kararıyla dokunulur. Sprint 2 kapsamı dışına çıkılmaz.
