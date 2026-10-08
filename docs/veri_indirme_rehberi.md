# TCGA Veri İndirme Rehberi (Mini sprint 1.3)

Sprint 1 için gereken veri: TCGA-LUAD, TCGA-BRCA, TCGA-COAD projelerinin **açık erişimli** RNA-seq
(STAR - Counts), maskeli somatik mutasyon (MAF) ve klinik verisi; ayrıca tanı slaytlarının yalnızca
**listesi** (slaytlar indirilmez, o Nisa'nın işi). Boyut: tahminen **8-12 GB**; kesin rakamı
`--dry-run` söyler (aşağıda).

> Betik GDC alan adlarıyla belgelerden yazıldı ve gerçek API'ye karşı **henüz denenmedi**. İlk
> komut daima `--dry-run`. Uyumsuzluk olursa hata hangi kaydın bozuk olduğunu söyler.

## Yol A (önerilen): Bulut oturumu indirir, senin internetin kullanılmaz
1. Claude oturumunun ortam menüsünden **Edit → Network access**; ya daha geniş erişim seç ya da
   **Allowed domains**'e `api.gdc.cancer.gov` ve `portal.gdc.cancer.gov` ekle (paket yöneticileri
   kutusu işaretli kalsın).
2. Yeni bir oturumda "1.3'ü çalıştır" de. `--dry-run`, sonra `--limit 3`, sonra tam indirme yapılır.

## Yol B: Okul bilgisayarı (Windows)
Önkoşul: ≥ 15 GB boş disk, bilgisayarın uyumaması. Bu yolda veri **o bilgisayarda** kalır; bulut
oturumuna taşıyamam, bu yüzden 1.4-1.8'i de aynı bilgisayarda çalıştırırsın.

```powershell
winget install --id=astral-sh.uv -e          # uv
winget install --id=Git.Git -e               # git
git clone https://github.com/Weli-byte/FUBiltagteam.git
cd FUBiltagteam
uv sync --group data
uv run python scripts/download_gdc.py --dry-run     # 1) sayıları ve toplam GB'ı gör
uv run python scripts/download_gdc.py --limit 3     # 2) her türden 3 dosya dene
uv run python scripts/download_gdc.py               # 3) tam indirme (kesilirse aynı komutu tekrar çalıştır)
```
> Kod `main` dalına girmediyse klonlarken `git clone -b claude/kind-euler-8kj48k ...` kullan.

## Çıktılar
| Dosya | İçerik | Git'e girer mi |
|---|---|---|
| `data/raw/gdc/<proje>/<tür>/<id>/<dosya>` | ham veri | **Hayır** (DVC) |
| `data/manifests/*.tsv`, `query_meta.json` | manifestler (`gdc-client` ile uyumlu), GDC sürümü ve sorgu tarihi | Evet |
| `data/interim/patient_map.parquet` | hasta ↔ örnek ↔ dosya eşlemesi | **Hayır** |
| `data/interim/gdc_failed.tsv` | başarısız dosyalar (varsa) | Hayır |
| `docs/veri_envanteri.md` | proje bazında toplu sayılar ve kesişimler | Evet (hasta düzeyi yok) |

**`git add .` veya `git add -A` ile `data/` eklenmez.** Veri DVC ile izlenir (`dvc add data/raw/gdc`);
ortak depo (Google Drive) kurulumu ayrı bir iş, ekip karar verince `docs/ortam_kurulumu.md`'ye eklenecek.

## Nasıl doğrularım? (bitti kriteri)
1. `data/interim/gdc_failed.tsv` yok ve md5 hatası sıfır (betik her dosyada md5 kontrol eder).
2. `docs/veri_envanteri.md`'deki rakamlardan **5 rastgele hastayı** GDC portalında
   (portal.gdc.cancer.gov, hasta barkodunu ara) elle kontrol et: RNA-seq/MAF/slayt var-yok durumu uyuşmalı.
3. `patient_map.parquet` içinde her `patient_barcode` bir kez geçmeli (testle de korunuyor).

## Kurallar
- Yalnızca açık erişimli (`access = open`) veri; kontrollü erişimli veriyi indirmeye çalışma.
- Hasta başına tek RNA-seq: birincil tümör (`Primary Tumor`), birden fazlaysa en küçük örnek barkodu
  (`-01A`, sonra `-01B`); aday sayısı `n_rnaseq_primary_candidates` sütununda görünür.
- Normal doku RNA-seq'i varsayılan olarak indirilmez (`--include-normal` ile).
- Hasta düzeyinde veri repoya girmez; yalnızca TCGA kimlikleri kullanılır.
- GDC/TCGA veri kullanım ve atıf koşullarını makale/rapor yazmadan önce resmi sayfadan doğrulayın.
