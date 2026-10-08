# ONKOS

Yapay zeka destekli multimodal dijital onkoloji **araştırma prototipi** (TEKNOFEST + TÜBİTAK 2209-A).

> **Önemli:** ONKOS bir araştırma prototipidir. Klinik tanı veya tedavi kararı için kullanılmaz;
> hiçbir çıktısı "kesin tanı" veya "reçete" olarak sunulmaz. Diffusion simülasyonu klinik olarak
> doğrulanmış bir tahmin değil, kavramsal bir dijital ikiz gösterimidir (tedavi öncesi/sonrası
> eşleşmiş TCGA görüntüsü yoktur).

## Modüller ve sahipleri

| Modül | Paket | Sahip |
|---|---|---|
| 1. ViT + ABMIL histopatoloji | `onkos/histo` | Nisa |
| 2. Omics Encoder + Cross-Attention | `onkos/omics` | Proje lideri |
| 3a. GNN ilaç önerisi | `onkos/gnn` | İdil |
| 3b. RAG + LLM rapor | `onkos/rag_llm` | Proje lideri |
| 4. Conditional Diffusion (kavramsal) | `onkos/diffusion` | Proje lideri |
| 5. XAI (GradCAM / Attention) | `onkos/xai` | Yusuf |
| Backend API | `onkos/api` | İdil |
| Frontend | `apps/web` | Yusuf |

Mimari ve veri akışı: [docs/architecture.md](docs/architecture.md). Kararlar: [docs/adr/](docs/adr/).

## Klasör yapısı

```
.
├── onkos/            Python paketi (modüller burada; ADR-0001)
│   ├── common/       Ortak yardımcılar (seed, config, log)
│   ├── contracts/    Modüller arası sürümlü Pydantic şemaları
│   ├── histo/ omics/ gnn/ rag_llm/ diffusion/ xai/ api/
├── apps/web/         React/Next.js arayüzü
├── data/             raw/ interim/ processed/ (git'te değil, DVC ile)
├── infra/            Dockerfile, docker-compose, bulut betikleri
├── docs/             Mimari, ADR'ler, rehberler
├── notebooks/        Keşif defterleri
└── tests/            Testler (modül sınırı testi dahil)
```

## Kurulum

Gereksinimler: [uv](https://docs.astral.sh/uv/) (Python 3.11'i kendisi indirir), git.

```bash
git clone https://github.com/Weli-byte/FUBiltagteam.git
cd FUBiltagteam
uv sync                       # Python 3.11 + bağımlılıklar
uv run pre-commit install     # commit öncesi kontroller
uv run pytest                 # kurulum doğrulaması
```

Kalite kontrolleri: `uv run ruff check . && uv run black --check . && uv run mypy && uv run pytest`

Katkı kuralları için [CONTRIBUTING.md](CONTRIBUTING.md).
