# GPU Eğitim Ortamı: Kurulum, Depolama, Erişim ve Kullanım (Mini sprint 1.2, Nisa)

> Proje liderinin `docs/ortam_kurulumu.md` rehberi yerel/CPU ortamını ve `uv` gruplarını anlatır;
> bu belge onun üstüne **GPU makinesi** katmanını ekler. Docker imajı ve kurulum betiği bu
> ortamda (Windows dizüstü, CPU torch) **derlenip çalıştırılmadı**; sürüm etiketleri 13.10.2026'da
> resmi kaynaklardan doğrulandı ve ilk bulut kurulumunda `scripts/verify_env.py` ile sınanacak.

**Sade anlatım:** Yapay zekanın eğitileceği "laboratuvarı" kurarız: ekran kartı sürücüsü, kütüphaneler,
veri için hızlı disk, ekip arkadaşlarının güvenli girişi. Hepsini tek bir paket (Docker imajı)
ve tek bir kurulum betiği haline getiririz ki herkes aynı sonucu alsın.

## 1. Dosyalar

| Dosya | Ne işe yarar |
|---|---|
| `infra/gpu/Dockerfile.train` | CUDA 12.6 + Python 3.11 + torch (cu126) + OpenSlide (openslide-bin) + proje bağımlılıkları; `train` ve `dev` hedefleri |
| `infra/gpu/docker-compose.gpu.yml` | `train`, `verify`, `jupyter`, `mlflow` servisleri; GPU rezervasyonu; disk bağlama |
| `infra/gpu/setup_gpu_machine.sh` | Temiz Ubuntu 22.04/24.04 makineyi 15-30 dk'da hazırlar (sürücü kontrolü, Docker + NVIDIA Container Toolkit, uv, repo, diskler, auto-shutdown, doğrulama) |
| `infra/gpu/auto_shutdown.sh` | 60 dk GPU boşta → makineyi kapat (cron, her 10 dk) |
| `scripts/verify_env.py` | torch.cuda, AMP matmul, OpenSlide sürümü, `--slide <gerçek WSI>` verilirse slayt açma + patch okuma (verilmezse bu kontrol "SKIP"), h5py/cv2/timm sürümleri |
| `configs/histo/*.yaml` | Tüm hat parametreleri (patch boyutu, eşikler, eğitim tarifi) |

## 2. Sürümler (sabitlenmiş; kaynak: PyPI/Docker Hub/NVIDIA, 13.10.2026)

| Bileşen | Sürüm | Not |
|---|---|---|
| Taban imaj | `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` | Host sürücüsü **≥ 560.35** (CUDA 12.6); 12.4 için ≥ 550.54, 12.8 için ≥ 570.124 |
| Python | 3.11 (`uv python install`) | `.python-version` |
| uv | 0.12.6 | `uv.lock` ile tüm Python paketleri sabit |
| torch / torchvision | 2.14.1 / 0.29.1 | `uv.lock` CPU derlemesini pinler (CI için); GPU'da `cu126` wheel'i üstüne kurulur. **cu124 için 2.14 yok**; cu126/cu130/cu132 var |
| openslide-python / openslide-bin | 1.4.6 / 4.0.1.2 | `openslide-bin` libopenslide'ı paketler; apt `libopenslide0` **gerekmez** (apt'teki 3.4.1 eski) |
| OpenCV (headless), tifffile, timm, h5py | 5.0.0.93, 2026.3.3, 1.0.30, 3.16.0 | tifffile ≥ 2026.9 Python 3.12 ister; 3.11 için 2026.3.3 son sürüm |
| MLflow sunucu imajı | `ghcr.io/mlflow/mlflow:v3.17.0` | `latest` yerine sabit etiket |
| NVIDIA Container Toolkit | 1.20.1-1 | Resmi kurulum komutları betikte |

Neden torch CPU pinli? `uv.lock` tek bir platform çözümü tutar; CI (GitHub Actions, GPU'suz) ve
dizüstü bilgisayarlar CPU wheel'i kullanır. GPU makinesinde `uv pip install --index-url
https://download.pytorch.org/whl/cu126 torch==2.14.1 torchvision==0.29.1` ile CUDA derlemesi
üstüne yazılır (hem Dockerfile hem kurulum betiği bunu yapar). Alternatif (Sprint 2 kararı):
`[tool.uv.index]` + `[tool.uv.sources]` ile Linux'ta cu126, diğerlerinde CPU seçen marker tabanlı
pin (uv dokümanı: https://docs.astral.sh/uv/guides/integration/pytorch/).

## 3. Depolama düzeni (ne nerede)

```
/mnt/nvme/onkos-data/          hızlı yerel NVMe (geçici ama büyük; makine silinirse yeniden indirilir)
├── raw/
│   ├── tcga_slides/<proje>/slides/<file_id>/<barkod>.svs     GDC'den (md5 doğrulamalı)
│   └── camelyon/CAMELYON16/{images,annotations,masks,evaluation}/
├── interim/     qc_results.parquet, slide_inventory.parquet, patient_slide_map.parquet
├── processed/
│   ├── patch_v0/            örnek paket (PNG + patches.csv)
│   ├── patches/coords_v1/   slayt başına koordinat HDF5 (küçük, DVC'ye girer)
│   ├── camelyon16/masks/    tümör maskeleri (PNG) + overlay'ler
│   └── features/<encoder>/  Sprint 2: özellik HDF5 (sözleşme) — DVC'ye girer
└── manifests/   (git'te; data/manifests ile aynı)

/mnt/persist/                  kalıcı disk (küçük ama kaybolmamalı)
├── onkos-outputs/   checkpoint'ler (last.pt, best.pt), sonuç tabloları, grafikler
├── onkos-mlflow/    mlflow.db + artifacts
└── hf-cache/        timm / Hugging Face ağırlıkları (tekrar indirmemek için)
```

Repo içindeki `data/` ve `outputs/` bu klasörlere **sembolik bağ**dır (kurulum betiği yapar).
Boyut planı: TCGA seçilen slaytlar ~1,7 TB (`docs/histo/tcga_slayt_envanteri.md`), CAMELYON16
~0,76 TB; başlangıçta **2 TB NVMe** yeter (CAMELYON16 + TCGA alt kümesi), tam TCGA için 4 TB.

## 4. Sıfırdan kurulum (bulut makinesi, Ubuntu)

```bash
# 1) makineyi aç (tek GPU, ≥ 8 vCPU, ≥ 32 GB RAM, 2 TB NVMe, SSH anahtarıyla; parola kapalı)
# 2) betik
curl -fsSL https://raw.githubusercontent.com/Weli-byte/FUBiltagteam/main/infra/gpu/setup_gpu_machine.sh -o setup.sh
DATA_DISK=/dev/nvme1n1 bash setup.sh        # boş ek diski biçimlendirip /mnt/nvme'ye bağlar
# 3) doğrulama (betik sonunda otomatik): uv run python scripts/verify_env.py --require-cuda
#    gerçek slayt okuma kontrolü için: VERIFY_SLIDE=/mnt/nvme/onkos-data/raw/.../slayt.svs bash setup.sh
#    (ya da elle: uv run python scripts/verify_env.py --require-cuda --slide <slayt>)
```

Docker ile (isteğe bağlı, aynı imajı herkes kullanır):

```bash
docker compose -f infra/gpu/docker-compose.gpu.yml build
docker compose -f infra/gpu/docker-compose.gpu.yml run --rm verify
docker compose -f infra/gpu/docker-compose.gpu.yml run --rm train \
    python scripts/train_patch_classifier.py train.epochs=1
```

Colab'da: `docs/ortam_kurulumu.md` §3 + `!uv sync --group train --group data --group histo` ve
`!uv run python scripts/verify_env.py`.

## 5. Erişim (güvenli)

- SSH yalnızca anahtarla: `ssh-keygen -t ed25519`, açık anahtar makineye (bulut konsolundan);
  `PasswordAuthentication no` (bulut imajlarında varsayılan). Her üyenin kendi anahtarı
  `~/.ssh/authorized_keys`'e eklenir; ayrılan üyenin anahtarı silinir.
- Jupyter ve MLflow **yalnızca localhost**'a bağlanır. Tarayıcıdan erişim SSH tüneliyle:
  `ssh -L 8888:localhost:8888 -L 5000:localhost:5000 ubuntu@<ip>`; Jupyter token'ı `.env`'de
  (`JUPYTER_TOKEN`), herkese açık port yok.
- Uzun koşular `tmux new -s train` içinde; bağlantı kopsa da sürer. `tmux attach -t train`.

## 6. Günlük kullanım (maliyet için kapatmayı unutma!)

```bash
ssh gpu-box && cd ~/FUBiltagteam && git pull
tmux new -s train
uv run python scripts/verify_env.py --require-cuda                # 30 sn
uv run python scripts/download_camelyon.py download.limit=5       # veri çek (resume edebilir)
uv run python scripts/extract_patches.py run.workers=8            # koordinatlar (resume edebilir)
uv run python scripts/train_patch_classifier.py                   # eğitim; MLflow'a loglar
# bitince:
dvc add data/processed/patches/coords_v1 && git add *.dvc && git commit -m "coords v1" && dvc push
sudo shutdown -h now        # veya bulut konsolundan Stop. auto_shutdown 60 dk sonra zaten kapatır.
```

Checkpoint/resume: `onkos.common.checkpoint.CheckpointManager` her epoch `last.pt`/`best.pt`
yazar; kesilen eğitim aynı komutla `train.resume=true` (Sprint 2'de eklenecek bayrak) ile devam eder.
Patch çıkarımı ve indirmeler zaten dosya düzeyinde devam eder.

## 7. Doğrulama ve kabul

- [ ] `setup_gpu_machine.sh` temiz makinede 30 dk içinde bitti (süreyi buraya yazın: __).
- [ ] `verify_env.py --require-cuda`: torch.cuda OK, fp16 matmul OK, OpenSlide 4.0.1 OK, örnek SVS açıldı.
- [ ] En az iki ekip üyesi kendi SSH anahtarıyla bağlanıp `scripts/hello_world.py` çalıştırdı.
- [ ] `auto_shutdown.sh` cron'da; `/var/log/onkos-auto-shutdown.log` doluyor.
- [ ] Docker imajı bir kez derlendi ve `run --rm verify` geçti (imaj boyutunu not edin).

## Kaynaklar (erişim: 13.10.2026)

- NVIDIA Container Toolkit kurulum: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- CUDA sürücü uyumluluk tablosu: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- nvidia/cuda etiketleri: https://hub.docker.com/r/nvidia/cuda/tags — mlflow imajı: https://github.com/mlflow/mlflow/pkgs/container/mlflow
- torch wheel index'leri: https://download.pytorch.org/whl/cu126 — uv + PyTorch: https://docs.astral.sh/uv/guides/integration/pytorch/
- openslide-bin: https://pypi.org/project/openslide-bin/ — openslide-python: https://openslide.org/api/python/
