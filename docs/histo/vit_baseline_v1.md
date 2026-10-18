# ViT Baseline v1: Patch Düzeyinde Tümör/Normal (Mini sprint 1.9)

> **Durum:** Eğitim/değerlendirme hattı yazıldı, test edildi ve sentetik pakette uçtan uca
> çalıştırıldı (MLflow kaydı dahil). **Gerçek veriyle ViT-S/16 ve ResNet-50 eğitimi
> yapılmadı** (GPU/veri makinesi bekleniyor); dolayısıyla "ViT, ResNet'ten %8-15 daha iyi" iddiası
> henüz **test edilmedi**; sonuç tablosu §5'te boş bırakıldı, uydurulmadı.

**Sade anlatım:** Yapay zekanın ilk sınavı: küçük doku karelerine bakıp "tümör mü, normal mi"
demek. İki aday yarışır: ViT (görüntüyü 16x16 parçalara bölüp parçalar arası ilişkiye bakan
"dikkat" tabanlı model) ve ResNet-50 (klasik evrişimli ağ). Aynı veri, aynı bölme, aynı tarif;
sonuçlar güven aralığıyla raporlanır.

## 1. Modeller ve lisanslar (kaynak: Hugging Face `timm/` model kartları, 13.10.2026)

| Kısa ad | timm adı | Ağırlık | Lisans |
|---|---|---|---|
| `vit_small` | `vit_small_patch16_224.augreg_in21k_ft_in1k` | ImageNet-21k ön eğitim, 1k ince ayar (AugReg) | Apache-2.0 |
| `vit_base` | `vit_base_patch16_224.augreg_in21k_ft_in1k` | aynı | Apache-2.0 |
| `resnet50` | `resnet50.a1_in1k` | "ResNet strikes back" A1 tarifi | Apache-2.0 |
| `tiny_cnn` | (yerel) | yok | testler için |

timm 1.0.30 (Apache-2.0). Ağırlıklar ilk kullanımda indirilir (HF cache; GPU makinesinde
`/mnt/persist/hf-cache`).

## 2. Tarif (`configs/histo/patch_classifier.yaml`)

- Girdi 224x224 (256 px patch INTER_AREA ile küçültülür), ImageNet normalizasyonu.
- AdamW, lr 1e-4, weight decay 0,05, **katman bazlı lr azalması 0,75** (ViT bloklarında;
  ResNet'te aşama başına), cosine + %10 warmup, gradient clipping 1.0.
- Sınıf ağırlıklı CrossEntropy (ters frekans, ortalama 1), AMP, gradient accumulation,
  erken durdurma (3 epoch), en iyi epoch doğrulama AUROC'una göre; `last.pt`/`best.pt`
  (`CheckpointManager`, weights_only yükleme).
- Augmentation: `HistoAugment` (1.8). TTA isteğe bağlı.
- Seed: `set_seed` (python/numpy/torch), DataLoader `seed_worker` + generator; CPU'da aynı seed
  aynı kayıp (`test_same_seed_same_losses`), GPU'da bit-bit garanti yok.

## 3. Değerlendirme protokolü (sızıntısız)

1. Yalnızca CAMELYON16 **eğitim** slaytları kullanılır; `val` = tam slaytlar (%20, slayt-majör
   etikete göre tabakalı) `split_rows_by_slide` ile ayrılır; aynı slaytın patch'leri iki kümeye
   giremez (`assert_slide_disjoint`).
2. Resmi test slaytları kilitlidir; `eval.final_evaluation=true` yalnızca ekip kararıyla, tek
   seferlik final koşuda.
3. Metrikler: AUROC, AUPRC, F1@0,5, doğruluk; **patch bootstrap %95 GA** (1.000 örnekleme) ve
   **3 seed ortalaması ± GA**. Not: patch'ler aynı slayttan geldiği için bağımsız değildir;
   bootstrap GA iyimser olabilir. Sprint 2'de slayt düzeyinde küme bootstrap eklenecek.
4. Grafikler: ROC, PR (`val_curves.png`), öğrenme eğrileri (`learning_curves.png`).
5. MLflow: `histo-patch_<model>-<seed>` run'ları, config + git hash + epoch metrikleri.

## 4. Çalıştırma

```bash
uv run --group histo python scripts/train_patch_classifier.py                 # 2 model x 3 seed
uv run --group histo python scripts/train_patch_classifier.py train.models=[vit_small] train.seeds=[0] train.epochs=3
```

Hattın kendisi `tests/test_histo_sample_pack_classifier.py` ile (sentetik test düzeneği, yalnızca
`tests/` altında) sınanır; betikler sentetik veri üretmez.

Beklenen süre (doğrulanmalı): ~10k patch, 10 epoch, ViT-S fp16, L4/A10G: 10-15 dk/seed;
ResNet-50 benzer; 2x3 koşu < 2 saat.

## 5. Sonuçlar (gerçek veri; doldurulacak)

| model | AUROC (ort, %95 GA) | AUPRC | F1 | doğruluk | seed |
|---|---|---|---|---|---|
| vit_small | — | — | — | — | 3 |
| resnet50 | — | — | — | — | 3 |

Hattın sentetik düzenekle sınanması yalnızca `tests/test_histo_sample_pack_classifier.py` içindedir;
sentetik veriden hiçbir sayı raporlanmaz.

## 6. "ViT ResNet'ten %8-15 daha iyi" iddiası

Proje dokümanındaki bu ifade kaynaksızdır ve bu deneyle sınanacak. Olası sonuçlar ve nasıl
yazılacağı: (a) fark GA içinde → "fark anlamlı değil"; (b) ViT üstün → yüzde ve GA ile; (c)
ResNet üstün → aynı dürüstlükle. Patch düzeyi sonuç slayt düzeyi (ABMIL) sonuca genellenmez;
Sprint 2 ABMIL deneyi ayrıca raporlanır. Patoloji temel modelleri (UNI/CONCH/Virchow;
erişim koşulları `docs/sprint1/nisa_sprint2_hazirlik.md`) ImageNet ViT'ten belirgin daha iyi
olabilir; bu baseline onların referans noktasıdır.

## 7. Doğrulama (bitti kriteri)

- [x] Eğitim ve doğrulama slaytları kesişmiyor (test + kod).
- [x] Test bölmesi kilitli (PermissionError).
- [ ] Gerçek veride 3 seed ortalaması ve GA ile sonuçlar (§5).
- [x] Aynı config + seed CPU'da aynı kayıp (tekrarlanabilirlik testi).
- [ ] Model kartı v0 gerçek sonuçlarla güncellendi (`MODEL_CARD_v0.md`).
