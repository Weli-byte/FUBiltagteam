#!/usr/bin/env bash
# Shut the GPU machine down after a period of idle GPU (mini sprint 1.1/1.2 cost control).
#
# Bu betik ne işe yarar? Cron her 10 dakikada çalıştırır; GPU kullanımı IDLE_MINUTES boyunca
# %5'in altında kaldıysa ve etkin bir eğitim süreci yoksa makineyi kapatır. Bulutta "açık
# unutulan GPU" en pahalı hatadır; bu betik onu önler. Kapatmak istemiyorsanız
# `touch /tmp/onkos-keep-alive` (dosya 24 saat sonra yok sayılır).
set -euo pipefail
IDLE_MINUTES="${IDLE_MINUTES:-60}"
THRESHOLD="${THRESHOLD:-5}"           # GPU utilisation percent
STATE=/var/tmp/onkos-gpu-idle-since
KEEP=/tmp/onkos-keep-alive

if [[ -f "$KEEP" && $(( $(date +%s) - $(stat -c %Y "$KEEP") )) -lt 86400 ]]; then
  echo "$(date -Is) keep-alive present, skipping"; exit 0
fi
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1)
busy_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
now=$(date +%s)
if [[ "$util" -ge "$THRESHOLD" || "$busy_procs" -gt 0 ]]; then
  rm -f "$STATE"; echo "$(date -Is) busy (util=${util}% procs=${busy_procs})"; exit 0
fi
[[ -f "$STATE" ]] || echo "$now" > "$STATE"
idle_since=$(cat "$STATE")
idle_min=$(( (now - idle_since) / 60 ))
echo "$(date -Is) idle for ${idle_min} min (util=${util}%)"
if [[ "$idle_min" -ge "$IDLE_MINUTES" ]]; then
  echo "$(date -Is) idle >= ${IDLE_MINUTES} min: shutting down"
  rm -f "$STATE"
  sudo shutdown -h now
fi
