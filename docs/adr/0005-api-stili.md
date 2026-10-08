# ADR-0005: API stili (FastAPI + REST/JSON, OpenAPI-first)

- **Durum:** Önerildi (ekip onayı bekleniyor)
- **Tarih:** 2026-10-08

## Karar
- **FastAPI** + Pydantic v2; REST/JSON. Uç noktalar: `/predict/mutation`, `/recommend/drugs`, `/report`, `/simulate`, `/explain`.
- Şemalar `onkos.contracts` içinde tek kaynak; OpenAPI buradan üretilir.
- Her şemada `schema_version`; geriye dönük uyumluluk kuralı. Hata gövdesi tek standart (RFC 7807 benzeri).
- Her yanıt araştırma prototipi uyarısı alanı taşır.

## Alternatifler
gRPC (frontend için ek yük), GraphQL (kapsam için gereksiz).

## Sonuçlar
Frontend ve backend aynı sözleşmeye karşı bağımsız geliştirilebilir; gerçekçi örnekler gerçek veri hattı çıktılarından gelir (uydurma veri kullanılmaz). Risk: uzun süren çıkarımlar için asenkron iş modeli sonra gerekebilir.
