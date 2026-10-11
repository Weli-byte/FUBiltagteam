<#
ONKOS - mini sprint 1.4 (omics on isleme) ve 1.5 (mutasyon etiketleri) gercek veriyle calistirma.
Dogrudan PowerShell'de calistirin (Claude Code gerekmez):

    powershell -ExecutionPolicy Bypass -File scripts\run_local_pipeline.ps1

GUVENLIK: Bu betik HICBIR dosyayi silmez, tasimaz veya uzerine yazmaz; hicbir islemi sonlandirmaz;
sistem ayari degistirmez; git'e bir sey eklemez veya gondermez. data\raw ve data\interim yalnizca
OKUNUR. Cikti klasorleri doluysa durur. Tum cikti logs\ klasorune kaydedilir.
#>
$ErrorActionPreference = "Continue"  # native komut uyarilari (stderr) betigi dusurmesin; hatalar $LASTEXITCODE ile denetlenir
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Stop-Pipeline($message) {
    Write-Host ""
    Write-Host "DUR: $message" -ForegroundColor Red
    try { Stop-Transcript | Out-Null } catch {}
    exit 1
}

function Invoke-Step($title, [scriptblock]$action) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
    & $action
    if ($LASTEXITCODE -ne 0) { Stop-Pipeline "'$title' hata verdi (cikis kodu $LASTEXITCODE). Yukaridaki hata metnini paylasin." }
}

$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$log = Join-Path $logDir ("pipeline_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")
Start-Transcript -Path $log | Out-Null
Write-Host "Kayit dosyasi: $log"

# --- A) Durum ve guvenlik ---------------------------------------------------------------
Write-Host "=== A) Durum kontrolu ===" -ForegroundColor Cyan
git status --short | Where-Object { $_ -notmatch '^\?\?' } | ForEach-Object { Write-Host $_ }
$tracked = git status --porcelain --untracked-files=no
if ($tracked) { Stop-Pipeline "Takip edilen dosyalarda kaydedilmemis degisiklik var (yukarida)." }
Write-Host ("Dal: " + (git branch --show-current))
Write-Host ("Son commit: " + (git log --oneline -1))

$rawFiles = (Get-ChildItem (Join-Path $root "data\raw\gdc") -Recurse -File -ErrorAction SilentlyContinue | Measure-Object)
Write-Host ("data\raw\gdc: {0} dosya" -f $rawFiles.Count)
if ($rawFiles.Count -lt 4000) { Stop-Pipeline "data\raw\gdc beklenenden az dosya iceriyor (4199 olmali)." }
if (-not (Test-Path (Join-Path $root "data\interim\patient_map.parquet"))) { Stop-Pipeline "data\interim\patient_map.parquet yok." }

foreach ($dir in @("data\processed\omics", "data\processed\labels", "docs\omics_qc", "docs\mutation_labels")) {
    $full = Join-Path $root $dir
    if ((Test-Path $full) -and (Get-ChildItem $full -Force -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0) {
        Stop-Pipeline "$dir klasorunde zaten dosya var; uzerine yazmamak icin durdum. Icerigi kontrol edip karar verin."
    }
}

# --- Bellek (yalnizca okuma; dilden bagimsiz WMI) ------------------------------------------
$perf = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
$availMB = [math]::Round($perf.AvailableMBytes)
$headroomMB = [math]::Round(($perf.CommitLimit - $perf.CommittedBytes) / 1MB)
$freeDiskGB = [math]::Round((Get-PSDrive -Name ($root.Substring(0,1))).Free / 1GB, 1)
Write-Host ("Kullanilabilir RAM: {0} MB | Taahhut payi (RAM + sanal bellek): {1} MB | Bos disk: {2} GB" -f $availMB, $headroomMB, $freeDiskGB)
Write-Host "Not: Betik tepe ozel bellek olarak yaklasik 0,5-1 GB kullanir; matris diskte tutulur."
if ($headroomMB -lt 3000) {
    Stop-Pipeline "Taahhut payi $headroomMB MB (< 3000 MB). Sanal bellegi buyutun (docs\omics_on_isleme.md) ve yeniden calistirin."
}
if ($freeDiskGB -lt 10) { Stop-Pipeline "Bos disk $freeDiskGB GB (< 10 GB)." }
if ($availMB -lt 500) {
    Write-Host "UYARI: Kullanilabilir RAM cok dusuk ($availMB MB); islem yavas (sayfalama) olabilir ama calisir." -ForegroundColor Yellow
}

# --- B) Mini sprint 1.4 ---------------------------------------------------------------------
Invoke-Step "uv sync (yalnizca proje .venv)" { uv sync --group data --group omics }

$sample = Get-ChildItem (Join-Path $root "data\raw\gdc\TCGA-LUAD\rnaseq") -Recurse -Filter *.tsv | Select-Object -First 1
if (-not $sample) { Stop-Pipeline "TCGA-LUAD RNA-seq .tsv dosyasi bulunamadi." }
Invoke-Step "1.4 inspect (dosya bicimi dogrulamasi)" { uv run python scripts/preprocess_omics.py inspect $sample.FullName }

$t0 = Get-Date
Invoke-Step "1.4 build (matris + QC)" { uv run python scripts/preprocess_omics.py build }
Write-Host ("build suresi: {0:N1} dk" -f ((Get-Date) - $t0).TotalMinutes)
Write-Host "--- docs\omics_qc\qc_report.md ---"
Get-Content (Join-Path $root "docs\omics_qc\qc_report.md") -Encoding UTF8
Get-ChildItem (Join-Path $root "data\processed\omics") | Select-Object Name, @{n="MB";e={[math]::Round($_.Length/1MB,1)}} | Format-Table -AutoSize | Out-String | Write-Host

Invoke-Step "1.4 fit-transform (kesif amacli, model egitiminde KULLANILMAZ)" { uv run python scripts/preprocess_omics.py fit-transform --eda-all-samples }
Get-ChildItem (Join-Path $root "docs\omics_qc") -Filter *.png | Select-Object Name, @{n="KB";e={[math]::Round($_.Length/1KB)}} | Format-Table -AutoSize | Out-String | Write-Host

# --- C) Mini sprint 1.5 ---------------------------------------------------------------------
Invoke-Step "1.5 mutasyon etiketleri" { uv run python scripts/build_mutation_labels.py }
Write-Host "--- docs\mutation_labels\dengesizlik_raporu.md ---"
Get-Content (Join-Path $root "docs\mutation_labels\dengesizlik_raporu.md") -Encoding UTF8
Get-ChildItem (Join-Path $root "data\processed\labels") | Select-Object Name, @{n="MB";e={[math]::Round($_.Length/1MB,2)}} | Format-Table -AutoSize | Out-String | Write-Host
Invoke-Step "1.5 rastgele 5 hasta (cBioPortal kontrolu icin)" { uv run python scripts/show_label_sample.py }

Write-Host ""
Write-Host "BITTI. Hicbir dosya silinmedi, git'e bir sey eklenmedi, hicbir sey push edilmedi." -ForegroundColor Green
Write-Host "Tum cikti: $log  (bu dosyanin icerigini paylasabilirsiniz)"
Stop-Transcript | Out-Null
