# Veri Kartı: patch_v0 Örnek Patch Seti

| Alan | İçerik |
|---|---|
| Amaç | Ekibin ViT baseline, veri yükleme ve arayüz denemelerine hızlı başlaması; sözleşme uyumlu küçük ama gerçek bir başlangıç paketi |
| Kaynak | CAMELYON16 eğitim+test slaytları (~40) ve TCGA tanı slaytları (~10); türev veri |
| Sürüm | v0 (plan: Sprint 1 sonu); `pack_meta.json` içinde config, sayılar, oluşturma zamanı, tissue-mask sürümü |
| Lisans | Kaynak lisanslarını devralır: CAMELYON16 CC0; TCGA açık erişim (TCGA teşekkürü zorunlu) |
| İçerik | ~10.000 adet 256x256 RGB PNG patch @ 0,5 µm/px; `patches.csv` (slayt, kaynak, bölme, etiket, etiket yöntemi, seviye-0 koordinat, doku/tümör oranı) |
| Etiket | CAMELYON: XML maskesiyle patch ≥ %50 tümör → 1, hiç tümör pikseli yok → 0, arası atlanır; TCGA: slayt düzeyi zayıf etiket (`slide_level`) |
| Bölme | CAMELYON16 resmi train/test (`split`); test kilitli; TCGA `unassigned` → proje liderinin hasta bölmesi |
| İşleme | thumbnail → doku maskesi v0 (Otsu) → 256 px ızgara, ≥ %50 doku → dengeli rastgele seçim (seed 42) → PNG + overlay |
| Kalite | Patch'ler gözle 30 örnekte kontrol edilecek; overlay ile koordinat doğrulaması; QC (1.6) uygulanmamış olabilir (v0) |
| Sınırlılıklar | Küçük (~50 slayt); doku tespiti v0 basit; TCGA etiketi zayıf; yalnızca gerçek slaytlardan üretilir (sentetik veri yok) |
| Depolama | `data/processed/patch_v0/` (DVC; git'te değil); boyut ~1-2 GB |
| Durum | Hat hazır; gerçek paket WSI indirmesi sonrası üretilecek (`docs/histo/patch_v0_teslim_notu.md`) |
| Sorumlu | Nisa |
