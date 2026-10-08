# Katkı Rehberi (1 sayfa)

## Dal (branch) akışı
- `main`: korumalı; yalnızca `dev`'den PR ile gelir, her zaman çalışır durumda.
- `dev`: entegrasyon dalı.
- `feature/<modül>-<kısa-ad>` (örn. `feature/omics-preprocess`): işler burada yapılır, `dev`'e PR açılır.
- Doğrudan `main` veya `dev`'e push yok.

## Pull Request ve code review
- Her PR **en az 1 onay** alır; kendi PR'ını kendin onaylayamazsın.
- PR küçük olsun (tek konu, ideal < 400 satır). Açıklamada ne/neden/nasıl test edildi yaz.
- Reviewer kontrol listesi: testler var mı, tip ipuçları var mı, modül sınırı ihlali yok mu,
  gizli anahtar yok mu, bilimsel iddia kaynaklı mı, sızıntı (leakage) riski yok mu.
- CI (ruff, black, mypy, pytest) yeşil olmadan birleştirme yok.

## Kod standartları
- Python 3.11, tip ipuçları zorunlu (`mypy --strict`), docstring açıklayıcı.
- Sihirli sayı ve gizli anahtar kodda olmaz: parametreler config dosyasından, anahtarlar `.env`'den.
- Seed her zaman sabitlenir (`onkos.common` içindeki yardımcı, mini sprint 1.2).
- Biçim: `ruff` + `black` (satır 100). `pre-commit install` yapılmalı.

## Modül sınırları (ADR-0001)
Bir modül başka bir modülün iç koduna **doğrudan bağımlı olamaz**. Konuşma yalnızca
`onkos.contracts` (sürümlü şemalar) ve `onkos.common` üzerinden olur.
`tests/test_module_boundaries.py` bunu otomatik kontrol eder.

## Sözleşme (contract) değişikliği
`onkos/contracts` değişiklikleri yalnızca PR ile ve ilgili modül sahibinin onayıyla yapılır;
`SCHEMA_VERSION` artırılır.

## Veri ve model dosyaları
Ham/işlenmiş veri ve model ağırlıkları git'e girmez (DVC). Hasta düzeyinde kişisel veri paylaşılmaz;
yalnızca anonim TCGA kimlikleri kullanılır.

## Bilimsel dürüstlük
Sonuçlar olduğundan iyi gösterilmez; sınırlılıklar yazılır; test kümesi hiperparametre
seçiminde kullanılmaz. "Simülasyon", "tahmin" ve "kanıt" kelimeleri ayrı tutulur.
