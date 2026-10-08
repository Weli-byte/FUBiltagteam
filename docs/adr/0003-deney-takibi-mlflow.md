# ADR-0003: Deney takibi aracı olarak MLflow

- **Durum:** Önerildi (ekip onayı bekleniyor)
- **Tarih:** 2026-10-08

## Bağlam
Deneyler, hiperparametreler, metrikler ve veri sürümleri birbirine bağlanabilmeli. Bütçe: ücretsiz.

## Karar
**MLflow** (self-hosted, yerel dosya/SQLite veya küçük sunucu) kullanılır. Run adı standardı:
`<modül>-<deney>-<seed>`; her run config'i ve git commit hash'ini kaydeder.

## Alternatifler
- Weights & Biases: kullanımı rahat ama hesap/plan bağımlılığı var, veri üçüncü tarafta; ücretsiz-yerel
  öncelik nedeniyle seçilmedi.
- Sadece TensorBoard: deney karşılaştırma ve model kaydı zayıf.

## Sonuçlar
Ücretsiz ve verimiz bizde. Risk: bulut GPU ile yerel sunucu arasında takip URI'si ayarı gerekir (mini sprint 1.2).
