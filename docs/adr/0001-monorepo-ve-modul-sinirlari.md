# ADR-0001: Tek repo (monorepo) ve modül sınırları

- **Durum:** Önerildi (ekip onayı bekleniyor)
- **Tarih:** 2026-10-08

## Bağlam
4 kişilik ekip 12 ayda 5 modül geliştiriyor. Herkesin birbirini bloklamadan paralel çalışması gerekiyor.

## Karar
- Depo kökü monorepo kökü; Python kodu tek paket `onkos` altında alt paketler olarak yaşar
  (`onkos/histo`, `omics`, `gnn`, `rag_llm`, `diffusion`, `xai`, `api`, ortak: `common`, `contracts`).
  Sprint planındaki `models/omics` vb. yerleşimi yerine tek paket seçildi: tek `uv` ortamı, basit import.
- Özellik modülleri birbirinin iç koduna bağımlı olamaz; yalnızca `onkos.contracts` ve `onkos.common`.
  `tests/test_module_boundaries.py` bunu CI'da zorlar.
- Frontend (`apps/web`) Node projesi olarak ayrı kalır.

## Alternatifler
- Modül başına ayrı repo: sürüm ve CI yükü 4 kişi için fazla; reddedildi.
- `models/<modül>` ayrı paketler: çoklu `pyproject`, 12 ayda gereksiz karmaşa; reddedildi.

## Sonuçlar
Basit kurulum; sınır ihlalleri test ile yakalanır. Risk: paket büyürse sonradan bölmek gerekebilir.
