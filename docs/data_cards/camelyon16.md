# Veri Kartı: CAMELYON16

| Alan | İçerik |
|---|---|
| Amaç | Patch düzeyi tümör/normal baseline (1.9), doku/maske hattı doğrulaması, slayt düzeyi (ABMIL) karşılaştırma ölçütü; yayınlanmış sonuçlarla kıyas |
| Kaynak | CAMELYON16 challenge (Radboud UMC & UMC Utrecht); AWS Open Data `s3://camelyon-dataset/CAMELYON16/` (us-west-2); GigaDB doi:10.5524/100439; https://camelyon16.grand-challenge.org/ |
| Sürüm | AWS Open Data kopyası (listeleme 13.10.2026): 399 görüntü, 160 XML, 399 maske, `evaluation/reference.csv`, `README.md`, `license.txt`, `checksums.md5` |
| Lisans | **CC0 1.0** (`license.txt`; registry.opendata.aws/camelyon) |
| İçerik | Sentinel lenf nodu H&E WSI (meme kanseri metastazı); eğitim 270 (159 normal + 111 tumor), test 129; iki merkez (`reference.csv` `center`); lezyon sınıfı negative/micro/macro |
| Anotasyon | ASAP XML çokgenler (seviye-0 piksel), gruplar `Tumor` / `Exclusion` (eski: `_0`/`_1`/`_2`); 20 tumor + test_114 tam anotasyonlu değil |
| Format / boyut | BigTIFF (tile 512, JPEG), 0,33-3,97 GB/dosya, görüntüler 752,5 GB; mpp etiketi eksik olabilir (RUMC ~0,243, UMCU ~0,226 µm/px; doğrulanmalı) |
| Toplama | `scripts/download_camelyon.py` (anonim HTTPS, resume, boyut/md5 kontrolü) |
| İşleme | `scripts/build_tumor_masks.py`: XML → ikili maske (1/32 ölçek) + overlay; `camelyon16_splits.json` (resmi bölme, merkez, etiket); patch etiketi ≥ %50 tümör |
| Kalite | Resmi test/train ayrımı korunur; kısmi anotasyonlu slaytlar "normal" kaynağı olarak kullanılmaz; 20 slaytlık görsel maske kontrolü indirme sonrası |
| Sınırlılıklar | Tek organ/tek hastalık (lenf nodu metastazı); iki merkez; test etiketleri slayt düzeyi; ONKOS hedef dokularına (akciğer/kolon/meme primer) doğrudan genellenmez |
| Depolama | `data/raw/camelyon/CAMELYON16/` (git'te değil); maskeler ve koordinatlar DVC |
| Atıf | Ehteshami Bejnordi B ve ark. JAMA 2017;318(22):2199-2210, doi:10.1001/jama.2017.14585; Litjens G ve ark. GigaScience 2018;7(6):giy065, doi:10.1093/gigascience/giy065 |
| Durum | Tam set indirilmedi (0,75 TB); yalnızca küçük gerçek örnek (bkz. `docs/histo/raporlar/`). XML→maske hattı birim testleriyle sınandı; gerçek slaytta doğrulama örnek üzerinde yapıldı/yapılacak (rapora bakın) |
| Sorumlu | Nisa |
