## Ne değişti?

## Nasıl test edildi?

## Kontrol listesi
- [ ] `uv run ruff check . && uv run black --check . && uv run mypy && uv run pytest` geçti
- [ ] Modül sınırı ihlali yok (başka modülün iç koduna import yok)
- [ ] Gizli anahtar / hasta verisi / büyük dosya yok
- [ ] **Sözleşme değişti mi?** Evetse: `scripts/export_contracts.py` çalıştırıldı, `SCHEMA_VERSION`
      uyumluluk kuralına göre güncellendi, etkilenen tüketicilerin sahipleri incelemeye eklendi
- [ ] Bilimsel iddialar kaynaklı; sınırlılıklar yazıldı
