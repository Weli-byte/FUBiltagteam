# Patch Koordinat Sistemi ve HDF5 Koordinat Dosyaları (Mini sprint 1.7)

> Yusuf (ısı haritası) ve proje lideri (özellik dosyası) için referans. Kural tek cümle:
> **`x, y` daima seviye-0 (tam çözünürlük) pikselidir ve patch'in sol-üst köşesidir.**

**Sade anlatım:** Dev doku fotoğrafını küçük karelere böleriz; her karenin fotoğraftaki yerini
"en büyük fotoğraftaki piksel adresi" olarak yazarız. Böylece model bir kare için "tümör" dese,
o kareyi orijinal slaytın üstünde doğru yere boyayabiliriz (ısı haritası).

## 1. Tanımlar

| Terim | Anlamı |
|---|---|
| Seviye 0 | Slaytın en ayrıntılı piramit katmanı; `SlideInfo.width/height` bu katmanın boyutudur |
| `mpp` | mikron/piksel; seviye-0'da 0,25 ≈ "40x", 0,5 ≈ "20x" (tarayıcıya göre 0,23-0,26 olabilir) |
| `target_mpp` | Patch'in **istenen** fiziksel çözünürlüğü (varsayılan 0,5 = 20x) |
| `patch_size_px` | Patch kenarı, hedef çözünürlükte piksel (varsayılan 256 → 128 µm) |
| `patch_level0_px` | Aynı patch'in seviye-0'daki kenarı = `round(patch_size_px * target_mpp / slide_mpp)` |
| `stride_level0_px` | Komşu patch'ler arası adım (seviye-0); örtüşme yoksa `= patch_level0_px` |

## 2. Örnekler

| Slayt | `slide_mpp` | `target_mpp` | `patch_size_px` | `patch_level0_px` | Okuma |
|---|---|---|---|---|---|
| 20x taranmış TCGA | 0,50 | 0,5 | 256 | 256 | seviye 0'dan 256x256 doğrudan |
| 40x taranmış TCGA | 0,25 | 0,5 | 256 | 512 | seviye 0'dan 512x512 okunup 256'ya küçültülür (ya da downsample=2 seviyesi varsa oradan 256) |
| CAMELYON16 (~0,243) | 0,243 | 0,5 | 256 | 527 | seviye 0'dan 527x527 → 256 (INTER_AREA) |

`SlideReader.level_for_mpp(target_mpp)` en kaba ama hâlâ yeterince ince seviyeyi seçer (asla
büyütme yapılmaz) ve artık ölçeği (`rescale`) döndürür; `read_patch(x, y, 256, 0.5)` bunu uygular.

## 3. Koordinat dönüşümleri (Yusuf için)

Isı haritası için patch'i herhangi bir seviyeye/thumbnail'e taşımak:

```python
ds = slide_width_level0 / thumbnail_width          # thumbnail'in seviye-0'a oranı
x_t, y_t = x / ds, y / ds                           # patch sol-üst köşesi thumbnail'de
w_t = patch_level0_px / ds                          # patch kenarı thumbnail'de
```

`onkos.contracts` içindeki `PatchRef.x/y` ve `ExplainResponse.tile_size` aynı kuralı kullanır:
`tile_size = patch_level0_px` (seviye-0). Sözleşme bunu "seviye-0 piksel" olarak zaten belirtir.

## 4. HDF5 koordinat dosyası (`data/processed/patches/coords_v1/<slide_id>.h5`)

| Ad | Şekil / tür | Açıklama |
|---|---|---|
| `coords` | (N, 2) int32 | `(x, y)` seviye-0, sol-üst, satır-majör ızgara sırası |
| `tissue_fraction` | (N,) float32 | kutunun doku maskesiyle örtüşme oranı (≥ `min_tissue_fraction`) |
| attr `schema_version` | str | sözleşme sürümü (`onkos.contracts.SCHEMA_VERSION`) |
| attr `slide_id` | str | anonim anahtar (TCGA'da GDC dosya UUID'si; CAMELYON'da `tumor_001`) |
| attr `magnification`, `target_mpp`, `slide_mpp` | float | `magnification` = **çıkarım** büyütmesi (`target_mpp`'den: 0,5 → 20); tarayıcının yerel çözünürlüğü `slide_mpp` |
| attr `patch_size_px`, `patch_level0_px`, `stride_level0_px` | int | yukarıdaki tanımlar |
| attr `slide_width`, `slide_height`, `level_count` | int | seviye-0 boyutlar |
| attr `tissue_mask_version`, `config_hash`, `pipeline_version`, `created_at` | str | tekrarlanabilirlik |

Kaynak dosya yolu **bilerek dosyada yoktur**: TCGA dosya adları hasta barkodu taşır ve bu
dosyalar diğer modüllerle paylaşılır (sözleşme kuralı: paylaşılan çıktıda hasta kimliği olmaz).
Slayt yolu yalnızca yerel `*_results.csv` (`source_path` sütunu) ve slayt listesinde tutulur.

Sprint 2'de aynı dosyaya `features (N, D) float16` ve `encoder_name/encoder_version` eklenir;
`tests/test_histo_patches.py::test_coords_file_becomes_contract_file_once_features_are_added`
bunun proje liderinin doğrulayıcısından (`validate_patch_feature_file`) geçtiğini gösterir.

Okuma:

```python
from models.histo.patches.h5 import read_coords_h5
from models.histo.wsi.reader import SlideReader
coords, fractions, attrs = read_coords_h5("data/processed/patches/coords_v1/tumor_001.h5")
slide_path = "data/raw/camelyon/CAMELYON16/images/tumor_001.tif"   # slayt listesinden (slides.csv)
with SlideReader(slide_path, mpp_override=0.243) as reader:
    x, y = coords[0]
    patch = reader.read_patch(int(x), int(y), attrs.patch_size_px, attrs.target_mpp)  # (256,256,3) uint8
```

PyTorch: `models.histo.patches.dataset.PatchCoordsDataset(slide_path, coords_h5)` patch'leri
ihtiyaç anında okur (her DataLoader işçisi slaytı kendisi açar).

## 5. Üretim hattı (`scripts/extract_patches.py`)

- Girdi: `data/manifests/slides_to_extract.csv` (`slide_id,path[,mpp_override]`).
- Her slayt: thumbnail (2048 px) → doku maskesi v1 → ızgara → doku filtresi → HDF5 (atomik yazım).
- Paralel (`run.workers`), **kesintiden devam** (geçerli dosya varsa atlanır; bozuksa yeniden),
  bozuk slayt loglanıp geçilir; `_summary.json` ve `*_results.csv` (slayt, patch sayısı, süre).
- Milyonlarca PNG yazılmaz; yalnızca koordinatlar saklanır (slayt başına ~10-50 KB).

### Performans (dürüst not)

Gerçek slaytlarda henüz ölçülmedi. Beklenti (literatür/CLAM deneyimi, doğrulanmalı): thumbnail +
maske + ızgara slayt başına 2-10 sn (koordinat üretimi I/O'ya bağlı değil); patch okuma özellik
çıkarımında (Sprint 2) darboğaz olur. Gerçek slaytlarla ölçüm yapıldığında sayılar
`_summary.json` çıktısından alınıp buraya yazılacak; sentetik test slaytlarının süreleri raporlanmaz.

## 6. Doğrulama (bitti kriteri)

- [ ] Rastgele 100 koordinatın slayt üzerinde doğru yere düştüğü görsel olarak kontrol edildi
      (`scripts/make_sample_pack.py` overlay'leri; `draw_boxes`).
- [ ] Hat yarıda kesilip yeniden başlatıldığında tamamlananlar atlandı (`test_extract_writes_validates_and_resumes`).
- [ ] 20x ve 40x slaytlar aynı fiziksel ızgarayı verdi (`test_same_physical_grid_for_20x_and_40x_scans`).
- [ ] Yusuf koordinat dönüşümünü kendi thumbnail'inde bir patch kutusuyla denedi.
