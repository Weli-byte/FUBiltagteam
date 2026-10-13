#!/usr/bin/env bash
# ONKOS GPU machine bootstrap (mini sprint 1.2, Nisa). Ubuntu 22.04/24.04, NVIDIA GPU.
#
# Bu betik ne işe yarar? Temiz bir bulut makinesini ~15-30 dakikada eğitim ortamına çevirir:
# sürücü kontrolü, Docker + NVIDIA Container Toolkit, uv + Python 3.11, repo, bağımlılıklar,
# depolama klasörleri ve ortam doğrulaması. İki kez çalıştırmak güvenlidir (idempotent).
#
# Kullanım:
#   curl -fsSL https://raw.githubusercontent.com/Weli-byte/FUBiltagteam/main/infra/gpu/setup_gpu_machine.sh -o setup.sh
#   bash setup.sh                      # tam kurulum (Docker dahil)
#   SKIP_DOCKER=1 bash setup.sh        # yalnızca uv ortamı
#   DATA_DISK=/dev/nvme1n1 bash setup.sh   # ek NVMe diski /mnt/nvme'ye bağla (biçimlendirir! boş disk olmalı)
#
# Güvenlik: parola ile SSH kapalı tutulur (bulut imajlarında varsayılan); Jupyter/MLflow yalnızca
# localhost'a bağlanır, SSH tüneliyle erişilir. Boşta kalan makine için auto_shutdown.sh kurulur.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Weli-byte/FUBiltagteam.git}"
REPO_DIR="${REPO_DIR:-$HOME/FUBiltagteam}"
BRANCH="${BRANCH:-main}"
DATA_ROOT="${DATA_ROOT:-/mnt/nvme/onkos-data}"
PERSIST_ROOT="${PERSIST_ROOT:-/mnt/persist}"
UV_VERSION="${UV_VERSION:-0.12.6}"
NCT_VERSION="${NCT_VERSION:-1.20.1-1}"   # nvidia-container-toolkit (doc 2026-10-13)

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

log "1/7 NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi bulunamadı. Bulut imajı sürücü içermiyorsa: sudo ubuntu-drivers install (sonra reboot)."
  echo "CUDA 12.6 için sürücü >= 560.35 gerekir (NVIDIA CUDA release notes, tablo 'Corresponding Driver Versions')."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

log "2/7 System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends git curl ca-certificates gnupg2 tmux htop unzip jq

if [[ "${SKIP_DOCKER:-0}" != "1" ]]; then
  log "3/7 Docker + NVIDIA Container Toolkit"
  if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER" || true
  fi
  if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
      nvidia-container-toolkit="${NCT_VERSION}" nvidia-container-toolkit-base="${NCT_VERSION}" \
      libnvidia-container-tools="${NCT_VERSION}" libnvidia-container1="${NCT_VERSION}"
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
  fi
  sudo docker run --rm --gpus all ubuntu:22.04 nvidia-smi >/dev/null && echo "Docker GPU OK"
else
  log "3/7 Docker skipped (SKIP_DOCKER=1)"
fi

log "4/7 Storage layout"
if [[ -n "${DATA_DISK:-}" ]]; then
  if ! blkid "$DATA_DISK" >/dev/null 2>&1; then
    echo "formatting $DATA_DISK as ext4 (DATA_DISK must be an EMPTY disk)"; sudo mkfs.ext4 -F "$DATA_DISK"
  fi
  sudo mkdir -p /mnt/nvme
  grep -q "/mnt/nvme" /etc/fstab || echo "$DATA_DISK /mnt/nvme ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
  sudo mount -a
fi
sudo mkdir -p "$DATA_ROOT"/{raw,interim,processed,manifests} \
  "$PERSIST_ROOT"/{onkos-outputs,onkos-mlflow,hf-cache}
sudo chown -R "$USER":"$USER" "$DATA_ROOT" "$PERSIST_ROOT"
df -h "$DATA_ROOT" "$PERSIST_ROOT" | sed 's/^/   /'

log "5/7 uv + repository"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull --ff-only || true
uv python install 3.11
uv sync --group train --group data --group histo
# CUDA build of torch on top of the CPU lock (see infra/gpu/Dockerfile.train for the rationale)
uv pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.14.1" "torchvision==0.29.1"
# data/ and outputs/ live on the fast/persistent disks, linked into the repo
[[ -L data ]] || { rm -rf data; ln -s "$DATA_ROOT" data; }
[[ -L outputs ]] || { rm -rf outputs; ln -s "$PERSIST_ROOT/onkos-outputs" outputs; }
cp -n .env.example .env || true

log "6/7 Idle auto-shutdown (cost control)"
sudo cp infra/gpu/auto_shutdown.sh /usr/local/bin/onkos-auto-shutdown
sudo chmod +x /usr/local/bin/onkos-auto-shutdown
( sudo crontab -l 2>/dev/null | grep -v onkos-auto-shutdown; echo "*/10 * * * * /usr/local/bin/onkos-auto-shutdown >> /var/log/onkos-auto-shutdown.log 2>&1" ) | sudo crontab -
echo "cron: every 10 min; shuts the machine down after 60 min of idle GPU (edit IDLE_MINUTES in the script)"

log "7/7 Verify"
uv run python scripts/verify_env.py --require-cuda
echo
echo "Done. Next: tmux new -s train; uv run python scripts/train_patch_classifier.py"
echo "Stop the machine when finished (cloud console or: sudo shutdown -h now)."
