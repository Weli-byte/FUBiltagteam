# patch_v0 Örnek Veri Seti: Teslim Notu (Mini sprint 1.3)

> **Durum (13.10.2026):** Hat hazır ve sentetik slaytlarla uçtan uca çalıştırıldı; **gerçek
> slaytlarla paket henüz üretilmedi** çünkü bu makinede WSI yok (CAMELYON16 görüntüleri 0,75 TB,
> TCGA seçilenler 1,7 TB). Gerçek paket, GPU/veri makinesi kurulur kurulmaz (Sprint 1 sonu /
> Sprint 2 başı) aşağıdaki komutla üretilip ekibe DVC üzerinden verilecek. Ekip bu arada format
> ve DataLoader ile `--demo` çıktısı üzerinde çalışabilir.

**Sade anlatım:** Ekibin yapay zeka denemelerine başlaması için birkaç düzine slayttan kesilmiş
küçük doku kareleri (patch) ve her karenin "tümör / normal" etiketi. İlk yemek tadımlığı:
mükemmel değil, ama gerçek ve çalışır.

## 1. İçerik ve format

```
data/processed/patch_v0/
├── patches.csv        bir satır = bir patch (sütunlar aşağıda)
├── pack_meta.json     config, sayılar, koordinat kuralı, başarısız slaytlar
├── images/<slide_id>/<x>_<y>.png     256x256 RGB, hedef 0,5 µm/px (20x)
└── overlays/<slide_id>.png           thumbnail + tümör maskesi (kırmızı) + seçilen kutular (yeşil)
```

| Sütun | Anlam |
|---|---|
| `image` | PNG'nin paket köküne göre yolu |
| `slide_id` | anonim slayt anahtarı (CAMELYON: `tumor_001`; TCGA: GDC dosya UUID'si) |
| `source` | `camelyon16` / `tcga` (gerçek pakette); merkez bilgisi ayrıca `centre` olarak eklenecek |
| `split` | `train` / `test` (CAMELYON16 resmi bölmesi) / `unassigned` (TCGA; proje liderinin bölmesi uygulanacak) |
| `label`, `label_name` | 1 tümör, 0 normal, -1 bilinmiyor |
| `label_method` | `camelyon_xml_mask` (patch'in ≥ %50'si anotasyon içinde → tümör; hiç tümör pikseli yok → normal; arası atlanır), `slide_level` (TCGA: birincil tümör slaydı → zayıf "tümör" etiketi), `none` |
| `x`, `y` | seviye-0 sol-üst köşe (bkz. `koordinat_sistemi.md`) |
| `patch_level0_px`, `patch_px`, `target_mpp`, `slide_mpp` | geometri |
| `tissue_fraction`, `tumor_fraction` | doku ve tümör örtüşme oranları |

Sözleşmeyle ilişki: aynı `(x, y)` ve `slide_id` HDF5 koordinat/özellik dosyalarında kullanılır;
PNG paketi yalnızca hızlı deneme içindir, üretim hattı (1.7) PNG yazmaz.

## 2. Slayt listesi (gerçek paket için plan)

| Kaynak | Slayt | Seçim |
|---|---|---|
| CAMELYON16 eğitim | 20 tumor + 15 normal | `tumor_001..020` (tam anotasyonlu olanlar; `tumor_010/015/018/020` kısmi anotasyonlu olduğu için atlanır → `021..024`), `normal_001..015`; her iki merkez |
| CAMELYON16 test | 5 (`test_001..005`, etiketleri `reference.csv`'den) | yalnızca `split=test`; eğitime girmez |
| TCGA | 10 (LUAD/BRCA/COAD'dan 3-4'er; proje liderinin listesinde olan hastalar) | yalnızca slayt düzeyi etiket |
| Toplam | ~50 slayt, 200 patch/slayt → ~10.000 patch | tümör/normal dengeli (CAMELYON'da) |

Liste `data/manifests/patch_v0_slides.csv` olarak yazılacak (sütunlar `scripts/make_sample_pack.py`
başlığında).

## 3. Üretim

```bash
uv run --group histo python scripts/make_sample_pack.py                 # configs/histo/patch_v0.yaml
dvc add data/processed/patch_v0 && git add data/processed/patch_v0.dvc && dvc push
```

## 4. Beş dakikada kullanım (proje lideri ve Yusuf için)

```python
from pathlib import Path
from torch.utils.data import DataLoader
from models.histo.patch_classifier.data import load_labelled_rows, rows_for_split, split_rows_by_slide
from models.histo.patches.dataset import SamplePackDataset

pack = Path("data/processed/patch_v0")
rows = load_labelled_rows(pack)                          # label 0/1 olanlar
train_rows, val_rows = split_rows_by_slide(rows_for_split(rows, "train"), val_fraction=0.2, seed=0)
loader = DataLoader(SamplePackDataset(pack, train_rows, image_px=224), batch_size=32, shuffle=True)
batch = next(iter(loader))
batch["image"].shape, batch["label"][:8], batch["slide_id"][:3]   # (32, 3, 224, 224), 0/1, slayt kimlikleri
```

- `rows_for_split(rows, "test")` **kilitlidir** (`PermissionError`); yalnızca final değerlendirmede
  `final_evaluation=True`.
- Aynı slaytın patch'leri iki kümeye dağılmaz (`split_rows_by_slide`), bu patch düzeyi sızıntının
  ana kaynağıdır.
- Isı haritası için `x, y` → `koordinat_sistemi.md`.

## 5. Bilinen sınırlılıklar (dürüstlük)

- Küçük boyut (~50 slayt): model seçimi için yeterli, yayın için değil.
- Doku tespiti v0 basit Otsu'dur (morfoloji yok); kalem izi/katlanma filtrelenmez (1.6 bunu yapar).
- TCGA etiketleri slayt düzeyidir: "tümör slaydı" içindeki normal doku patch'leri de "tümör"
  etiketlidir (çoklu örnek öğrenme, ABMIL, bunun için var).
- CAMELYON16'da 20 tumor slaytı tam anotasyonlu değildir; bunlar "normal" patch kaynağı olarak
  kullanılmaz (`PARTIALLY_ANNOTATED`).
- CAMELYON16 TIFF'lerinde mpp etiketi olmayabilir; `mpp_override=0.243` (tarayıcı değeri
  doğrulanmalı: RUMC 3DHistech 0,243 / UMCU Hamamatsu 0,226 civarı, Litjens 2018).

## 6. Doğrulama (bitti kriteri)

- [ ] Rastgele 30 patch gözle kontrol edildi (doku var, etiket mantıklı) — gerçek pakette.
- [ ] Bir slaytta overlay (kutular + maske) doğru yerde — gerçek pakette; sentetikte test ediyor.
- [ ] Ekipten biri DataLoader'ı çalıştırdı (yukarıdaki kod).
- [x] Hat testleri: `tests/test_histo_sample_pack_classifier.py` (paket içeriği, etiketler, loader).
