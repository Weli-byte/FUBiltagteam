# Colab'da sıfırdan yeniden üretim ve tam temel model koşusu

> **DENENMEDİ.** Bu hücreler yazıldı ama Colab'da çalıştırılmadı; hata çıkarsa hücre çıktısını olduğu gibi paylaşın.
> Amaç: veriyi **kendi internetinizle indirmeden/yüklemeden**, Colab'ın bağlantısıyla GDC'den yeniden indirip
> bütün hattı (1.3 → 1.6 → 1.8) orada çalıştırmak. Bu, aynı zamanda hattın yeniden üretilebilirlik sınavıdır.

## Neden bu yol?
- Ham veri GDC'den yeniden indirilebilir; ayrı bir yedeğe ihtiyaç yoktur. Gerçek yedek: kod + manifest + SHA-256 bölme özeti (GitHub).
- Colab'ın RAM'i (yaklaşık 12 GB) yerel makinedeki 200–900 MB boş RAM'den çok fazladır; `gbm` yavaşlığı ve tam koşu süresi düşebilir.
- Yeniden üretilen bölme, GitHub'daki dondurulmuş özetle (`docs/splits/splits-v1_manifest.json`) karşılaştırılır.

## Sınırlar (dürüstçe)
- Ücretsiz Colab oturumu boşta/12 saat civarında kopabilir; kopunca Colab diski silinir. Sonuçları **Drive'a** kopyalayın.
- GDC yeni bir veri sürümü yayımlamışsa (şu an kullanılan: Data Release 46.0) yeniden üretilen küme farklı olabilir.
  Bu durumda `--verify` başarısız olur; bu bir hata değil, sürüm farkıdır (bölme için `query_meta.json`'a bakın).
- Colab diskinde ~9 GB ham veri ve işlenmiş çıktı için yer yeterlidir; Drive'a yalnızca küçük sonuçlar yazılır.

## Hücreler

**Hücre 1 — Drive'ı bağla (yalnızca sonuçları saklamak için)**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Hücre 2 — Kodu al** (depo gizliyse parola yerine GitHub token kullanın, token'ı not defterine yazıp paylaşmayın)
```bash
%cd /content
!git clone -b claude/kind-euler-8kj48k https://github.com/Weli-byte/FUBiltagteam.git
%cd /content/FUBiltagteam
!pip -q install uv
!uv sync --group dev --group train --group data --group omics
```

**Hücre 3 — Ham veriyi GDC'den indir (Colab'ın interneti kullanılır)**
```bash
!uv run python scripts/download_gdc.py --dry-run
!uv run python scripts/download_gdc.py --limit 3
!uv run python scripts/download_gdc.py
```
Kesilirse aynı komutu tekrar çalıştırın. Sonda `data/interim/gdc_failed.tsv` olmamalı.

**Hücre 4 — Hattı çalıştır**
```bash
!uv run python scripts/preprocess_omics.py build
!uv run python scripts/build_mutation_labels.py
!uv run python scripts/make_splits.py --version splits-v1
!uv run python scripts/make_splits.py --verify splits-v1
!uv run pytest tests/test_no_leakage.py -q
```
`--verify` GitHub'daki dondurulmuş özetle eşleşmiyorsa DURUN ve çıktıyı paylaşın.

**Hücre 5 — Tam temel model koşusu**
```bash
!PYTHONUTF8=1 OMP_NUM_THREADS=2 uv run python scripts/train_baselines.py --run tam1 --splits-version splits-v1 2>&1 | tee tam1_log.txt
```

**Hücre 6 — Sonuçları Drive'a kopyala (silmeden)**
```bash
!mkdir -p /content/drive/MyDrive/ONKOS_sonuclar
!cp -r docs/baselines docs/splits /content/drive/MyDrive/ONKOS_sonuclar/
!cp -r data/processed/baselines /content/drive/MyDrive/ONKOS_sonuclar/ 2>/dev/null
!cp tam1_log.txt /content/drive/MyDrive/ONKOS_sonuclar/
```
Sonra `docs/baselines/tam1_rapor.md` içeriğini ve verdict satırlarını Claude'a yapıştırın.
