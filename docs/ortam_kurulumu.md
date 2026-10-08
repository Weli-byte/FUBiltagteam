# Ortam Kurulum Rehberi (Mini sprint 1.2)

Amaç: aynı komut, aynı seed → aynı metrik; yerelde, Colab'da ve bulut GPU'da.

## Bağımlılık grupları (`uv`)

| Grup | İçerik | Komut |
|---|---|---|
| (çekirdek) | pydantic, pyyaml, omegaconf | `uv sync` |
| `train` | torch, numpy, mlflow | `uv sync --group train` |
| `data` | dvc | `uv sync --group data` |

Sürümler `uv.lock` ile sabittir (requirements lock). CI yalnızca çekirdek + dev kurar; torch/mlflow
gerektiren testler orada **atlanır** (`importorskip`), `--group train` ile yerelde çalışır.

## 1) Yerel (CPU veya küçük GPU)
```bash
uv sync --group train --group data
cp .env.example .env
uv run pytest
uv run python scripts/hello_world.py              # 100 adımlık seed'li deney, MLflow'a loglanır
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db    # http://localhost:5000
```

## 2) Docker (GPU için NVIDIA Container Toolkit gerekir)
```bash
docker compose -f infra/docker-compose.yml up -d mlflow
docker compose -f infra/docker-compose.yml run --rm dev uv run pytest
docker compose -f infra/docker-compose.yml run --rm train        # GPU'lu eğitim
```
GPU kontrolü (Nisa'nın bulut ortamında yapılmalı):
`docker compose -f infra/docker-compose.yml run --rm train python -c "import torch; print(torch.cuda.is_available())"`

> Dockerfile ve compose bu ortamda **derlenip çalıştırılmadı**; CUDA taban imajı etiketi ve
> `ghcr.io/mlflow/mlflow` etiketi ilk kullanımda doğrulanmalıdır.

## 3) Google Colab Pro+
```python
!git clone https://github.com/Weli-byte/FUBiltagteam.git && cd FUBiltagteam
!pip install uv && uv sync --group train --group data
!uv run python scripts/hello_world.py
```
Colab'da `MLFLOW_TRACKING_URI`'yi paylaşılan MLflow sunucusuna yönlendir veya yerel SQLite kullan.

## 4) AWS SageMaker / EC2 GPU
GPU'lu bir instance'ta (A100/V100) sürücü + Docker + NVIDIA Container Toolkit kur, sonra Docker
bölümündeki komutlar. Maliyeti düşük tutmak için kullanılmayan instance'ı durdur.

## Tekrarlanabilirlik kuralları
- Her deney `seed` alır (`configs/config.yaml`); `onkos.common.reproducibility.set_seed` python, numpy
  ve torch'u (CPU+CUDA) sabitler. DataLoader için `seed_worker` + `make_generator` kullan.
- Run adı: `<modül>-<deney>-<seed>`; her run config'i ve git commit hash'ini MLflow'a yazar.
- Checkpoint: `onkos.common.checkpoint.CheckpointManager` (`last.pt`, `best.pt`; optimizer durumu dahil).
- AMP / gradient checkpointing: `onkos.common.precision`.
- **Not:** GPU'da bit-bit aynı sonuç garanti edilmez (bazı CUDA işlemleri deterministik değildir);
  bu testler CPU'da doğrulandı.

## DVC ile veri sürümleme
```bash
uv sync --group data
dvc remote add -d storage <uzak-depo>           # ücretsiz seçenekler aşağıda
dvc add data/raw/<dosya>                        # .dvc dosyasını git'e ekle, dosyanın kendisini ekleme
git add data/raw/<dosya>.dvc data/raw/.gitignore && git commit
dvc push                                        # veriyi uzak depoya gönder
dvc pull                                        # yeni makinede veriyi geri al
```
Uzak depo seçenekleri (ücretsiz): Google Drive (`gdrive://<klasör-id>`), ücretsiz katmanlı
S3 uyumlu depo (örn. Cloudflare R2, `s3://...` + `endpointurl`), veya paylaşımlı SSH sunucusu.
**Henüz ortak bir remote seçilmedi**; ekip karar verince `.dvc/config`'e eklenir. Hasta düzeyinde
kişisel veri yüklenmez, yalnızca anonim TCGA kimlikleri.
