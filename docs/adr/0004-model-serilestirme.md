# ADR-0004: Model serileştirme

- **Durum:** Önerildi (ekip onayı bekleniyor)
- **Tarih:** 2026-10-08

## Karar
- Eğitim checkpoint'leri: PyTorch `state_dict` + optimizer + config + git hash (en iyi, son model).
- Dağıtım/servis modelleri: `safetensors` ağırlıkları + model kartı (JSON/YAML: veri sürümü, metrik, sınırlılıklar).
- Ağırlıklar git'e girmez (DVC / MLflow artifact).

## Alternatifler
Ham `torch.save` pickle (güvenlik riski, sürüm kırılganlığı), ONNX (özel katmanlar için erken).

## Sonuçlar
Güvenli yükleme ve tekrarlanabilirlik. Risk: özel katmanlar için yükleme kodu sürüm bağımlıdır; model kartı bunu belgeler.
