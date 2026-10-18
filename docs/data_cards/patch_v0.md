# Veri Kartı: patch_v0 Örnek Patch Seti

| Alan | İçerik |
|---|---|
| Amaç | Ekibin ViT baseline, veri yükleme ve arayüz denemelerine hızlı başlaması; sözleşme uyumlu küçük ama gerçek bir başlangıç paketi |
| Kaynak | Hedef: CAMELYON16 eğitim+test slaytları (~40) ve TCGA tanı slaytları (~10); **mevcut (18.10.2026): 4 CAMELYON16 + 9 TCGA slaydı**; türev veri |
| Sürüm | v0 (plan: Sprint 1 sonu); `pack_meta.json` içinde config, sayılar, oluşturma zamanı, tissue-mask sürümü |
| Lisans | Kaynak lisanslarını devralır: CAMELYON16 CC0; TCGA açık erişim (TCGA teşekkürü zorunlu) |
| İçerik | Hedef ~10.000, **mevcut 2.600** adet 256x256 RGB PNG patch @ 0,5 µm/px; `patches.csv` (slayt, kaynak, merkez, bölme, etiket, etiket yöntemi, seviye-0 koordinat, doku/tümör oranı) |
| Etiket | CAMELYON: XML maskesiyle patch ≥ %50 tümör → 1, hiç tümör pikseli yok → 0, arası atlanır; TCGA: slayt düzeyi zayıf etiket (`slide_level`) |
| Bölme | CAMELYON16 resmi train/test (`split`); test kilitli; TCGA `unassigned` → proje liderinin hasta bölmesi |
| İşleme | thumbnail → doku maskesi v0 (Otsu) → 256 px ızgara, ≥ %50 doku → dengeli rastgele seçim (seed 42) → PNG + overlay |
| Kalite | Küçük örnekte 32 patch gözle kontrol edildi (18.10.2026, etiketler tutarlı); patch kutusu overlay'leri üretildi (13 PNG) ve 2 slaytta gözle bakıldı (kutular doku üzerinde; 100 koordinatlık tekil kontrol henüz yapılmadı, `docs/histo/koordinat_sistemi.md` §6); QC (1.6) 13 slaytta çalıştı (1 slayt `exclude`, pakete yine de girdi: v0'da QC filtresi uygulanmadı) |
| Sınırlılıklar | Küçük (hedef ~50 slayt, mevcut 13); doku tespiti v0 basit; TCGA etiketi zayıf; yalnızca gerçek slaytlardan üretilir (sentetik veri yok) |
| Depolama | `data/processed/patch_v0/` (DVC; git'te değil); mevcut paket 0,44 GB (2.600 PNG + overlay'ler), hedef ~1-2 GB |
| Durum | Küçük gerçek örnek üretildi (18.10.2026): 13 slayt, 2.600 patch (`data/processed/patch_v0`, git'te değil); ~50 slaytlık tam paket WSI indirmesi sonrası (`docs/histo/patch_v0_teslim_notu.md`) |
| Sorumlu | Nisa |
