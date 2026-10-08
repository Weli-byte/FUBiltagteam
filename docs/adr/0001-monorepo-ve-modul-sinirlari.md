# ADR-0001: Tek repo (monorepo) ve modül sınırları

- **Durum:** Önerildi (ekip onayı bekleniyor)
- **Tarih:** 2026-10-08

## Bağlam
4 kişilik ekip 12 ayda 5 modül geliştiriyor. Herkesin birbirini bloklamadan paralel çalışması gerekiyor.

## Karar
- Depo kökü monorepo kökü (sprint planındaki yerleşim): `models/{histo,omics,gnn,rag_llm,diffusion,xai}`,
  `services/api`, `apps/web`, `data`, `infra`, `docs`, `notebooks`, `tests`. Ortak kod `onkos/common` ve
  `onkos/contracts` altındadır. Tek `uv` ortamı kullanılır.
- Özellik modülleri (`models.*`, `services.api`) birbirinin iç koduna bağımlı olamaz; yalnızca
  `onkos.contracts` ve `onkos.common` içe aktarılabilir.
  `tests/test_module_boundaries.py` bunu CI'da zorlar.
- Frontend (`apps/web`) Node projesi olarak ayrı kalır.

## Alternatifler
- Modül başına ayrı repo: sürüm ve CI yükü 4 kişi için fazla; reddedildi.
- Her modül için ayrı `pyproject`: 12 ayda gereksiz karmaşa; reddedildi.

## Sonuçlar
Basit kurulum; sınır ihlalleri test ile yakalanır. Risk: paket büyürse sonradan bölmek gerekebilir.
