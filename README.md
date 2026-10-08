# ONKOS

Yapay zeka destekli multimodal dijital onkoloji **araştırma prototipi** (TEKNOFEST + TÜBİTAK 2209-A).

> **Önemli:** ONKOS bir araştırma prototipidir. Klinik tanı veya tedavi kararı için kullanılmaz;
> hiçbir çıktısı "kesin tanı" veya "reçete" olarak sunulmaz. Diffusion simülasyonu klinik olarak
> doğrulanmış bir tahmin değil, kavramsal bir dijital ikiz gösterimidir (tedavi öncesi/sonrası
> eşleşmiş TCGA görüntüsü yoktur).

## Modüller ve sahipleri

| Modül | Paket | Sahip |
|---|---|---|
| 1. ViT + ABMIL histopatoloji | `models/histo` | Nisa |
| 2. Omics Encoder + Cross-Attention | `models/omics` | Proje lideri |
| 3a. GNN ilaç önerisi | `models/gnn` | İdil |
| 3b. RAG + LLM rapor | `models/rag_llm` | Proje lideri |
| 4. Conditional Diffusion (kavramsal) | `models/diffusion` | Proje lideri |
| 5. XAI (GradCAM / Attention) | `models/xai` | Yusuf |
| Backend API | `services/api` | İdil |
| Frontend | `apps/web` | Yusuf |

Mimari ve veri akışı: [docs/architecture.md](docs/architecture.md). Kararlar: [docs/adr/](docs/adr/).

## Klasör yapısı

```
.
├── models/           AI modülleri: histo/ omics/ gnn/ rag_llm/ diffusion/ xai/
├── services/api/     FastAPI backend
├── onkos/            Ortak paket: common/ (seed, config, log), contracts/ (sürümlü Pydantic şemaları)
├── apps/web/         React/Next.js arayüzü
├── data/             raw/ interim/ processed/ (git'te değil, DVC ile)
├── configs/          YAML konfigürasyonları (model/data/train grupları)
├── scripts/          Çalıştırılabilir betikler (hello_world.py)
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

TCGA veri indirme: [docs/veri_indirme_rehberi.md](docs/veri_indirme_rehberi.md).

Modüller arası veri sözleşmesi ve değişiklik süreci: [docs/veri_sozlesmesi.md](docs/veri_sozlesmesi.md).

Eğitim araçları (torch, MLflow, DVC) ve Docker/Colab/AWS kurulumu: [docs/ortam_kurulumu.md](docs/ortam_kurulumu.md).

Kalite kontrolleri: `uv run ruff check . && uv run black --check . && uv run mypy && uv run pytest`

Katkı kuralları için [CONTRIBUTING.md](CONTRIBUTING.md).
