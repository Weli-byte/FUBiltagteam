# ONKOS Dokümantasyonu

ONKOS, H&E histopatoloji görüntüleri, omics verisi ve tıbbi literatürü birlikte kullanan
**araştırma prototipidir** (TEKNOFEST + TÜBİTAK 2209-A). Hiçbir çıktısı klinik tanı veya tedavi
kararı değildir.

| Bölüm | İçerik |
|---|---|
| Mimari ve kararlar | modül sınırları, veri sözleşmesi (HDF5 + JSON), ADR'ler |
| Ortam ve altyapı | yerel/Colab/GPU kurulumu, bulut maliyet planı |
| Veri hattı (omics) | TCGA RNA-seq/MAF indirme ve ön işleme (proje lideri) |
| Veri hattı (histoloji) | WSI indirme, QC, patch koordinatları, boya normalizasyonu, ViT baseline (Nisa) |
| Veri kartları | TCGA tanı slaytları, CAMELYON16, patch_v0 |
| Sprint 1 kapanışı | retrospektif, risk ve Sprint 2 hazırlık listeleri |

Kod ve testler: https://github.com/Weli-byte/FUBiltagteam — kalite kontrolleri
`uv run ruff check . && uv run black --check . && uv run mypy && uv run pytest`.

Kaynakça: `docs/references.bib` (Zotero'ya içe aktarılabilir; `dogrulanmali` notlu kayıtlar
alıntılanmadan önce doğrulanır).
