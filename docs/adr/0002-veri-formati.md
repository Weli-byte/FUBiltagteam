# ADR-0002: Veri formatları (Parquet + HDF5, DVC ile sürümleme)

- **Durum:** Önerildi (ekip onayı bekleniyor)
- **Tarih:** 2026-10-08

## Bağlam
Omics matrisleri (örnek x gen, GB düzeyi), etiket/bölme tabloları ve patch özellik vektörleri paylaşılacak.

## Karar
- Tablo verisi (etiketler, bölmeler, eşleştirme): **Parquet**.
- Büyük ifade matrisleri: **Parquet** (sıkıştırılmış) veya **h5ad/HDF5**; sütun/gen sözlüğü ayrı tutulur.
- Patch özellikleri (Nisa'dan): **HDF5** (`features` N×D float16, `coords` N×2 int, `slide_id`, `magnification`, `encoder_name`, `encoder_version`).
- Veri git'e girmez; **DVC** ile izlenir (uzak depo: ücretsiz S3 uyumlu depo veya Google Drive).
- Hasta kimliği olarak yalnızca anonim TCGA barkodu (hasta düzeyi: ilk 12 karakter).

## Alternatifler
CSV (yavaş, tipsiz), SQL'de ham matris (gereksiz yük), Zarr (ekipte deneyim yok).

## Sonuçlar
Hızlı okuma ve şema korunur. Risk: HDF5 sürüm uyumsuzluğu; `encoder_version` alanı ile izlenir.
