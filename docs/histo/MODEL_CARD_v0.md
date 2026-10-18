# Model Kartı v0: ONKOS Patch Sınıflandırıcı (taslak)

> Taslak; gerçek eğitim sonrası doldurulacak alanlar "—" ile işaretli. Araştırma prototipi;
> klinik tanı/tedavi kararı için kullanılmaz.

| Alan | Değer |
|---|---|
| Model adı / sürüm | `onkos-histo-patch-baseline` v0 (Sprint 1) |
| Görev | H&E patch (256 px @ 0,5 µm/px) → tümör / normal olasılığı |
| Mimari | ViT-S/16 (ImageNet-21k→1k, AugReg) ve ResNet-50 (A1); 2 sınıflı baş |
| Ön eğitim ağırlıkları | timm, Apache-2.0 (bkz. `vit_baseline_v1.md`) |
| Eğitim verisi | CAMELYON16 eğitim slaytları (RUMC + UMCU) patch'leri; etiket: anotasyon maskesi (≥ %50 tümör) — veri kartı `docs/data_cards/camelyon16.md` |
| Bölme | slayt bazlı train/val (%80/%20); resmi test kilitli |
| Tarif | `configs/histo/patch_classifier.yaml` (lr 1e-4, layer decay 0,75, AMP, HED augmentation) |
| Sonuçlar (val, 3 seed) — ÖN BULGU, NİHAİ DEĞİL | 18.10.2026, 2 eğitim + 2 doğrulama slaydı (400/400 patch): ViT-S AUROC 0,780 [0,729, 0,831], AUPRC 0,426; ResNet-50 AUROC 0,769 [0,659, 0,879], AUPRC 0,453; fark anlamlı değil (`docs/histo/vit_baseline_v1.md` §5) |
| Sonuçlar (resmi test, final) | — (yalnızca ekip kararıyla, tek sefer) |
| Kalibrasyon | yok (olasılıklar kalibre edilmedi; ABMIL aşamasında ele alınacak) |
| Bilinen sınırlılıklar | lenf nodu metastazı (meme) verisi; TCGA akciğer/kolon/meme dokusuna genelleme ölçülmedi; patch bağımsızlığı varsayımı bootstrap GA'yı iyimser yapar; merkez etkisi (2 merkez) |
| Adil kullanım / etik | Veri CC0, anonim; model yalnızca araştırma; klinik kullanım yok |
| Çıktı sözleşmesi | olasılık (0-1) `Probability`; slayt düzeyi karar ABMIL (Sprint 2) ile `onkos.contracts` |
| Tekrarlanabilirlik | seed, config hash, git hash MLflow'da; `best.pt` + `results.json` |
| Sorumlu | Nisa (Modül 1) |
