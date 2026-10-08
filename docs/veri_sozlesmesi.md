# ONKOS Veri Sözleşmesi (Mini sprint 1.7)

Modüller birbirinin koduna değil, bu sözleşmeye bağımlıdır (ADR-0001). Böylece Nisa (ViT/ABMIL),
proje lideri (Omics, RAG, Diffusion), İdil (GNN, backend) ve Yusuf (frontend, XAI) mock veriyle
birbirini beklemeden geliştirir.

- Şemalar: `onkos/contracts/schemas.py` (tek doğruluk kaynağı)
- Alan alan anlam/birim/aralık: [veri_sozlesmesi_alanlar.md](veri_sozlesmesi_alanlar.md) (otomatik üretilir)
- API taslağı: `onkos/contracts/openapi.yaml` (OpenAPI 3.1, otomatik üretilir)
- Örnek mesajlar: `onkos/contracts/examples/*.json`
- Mock veri: `onkos/contracts/mock.py`, mock sunucu: `scripts/mock_server.py`

> **Araştırma prototipi.** Her model çıktısı `research_use_only: true` ve `disclaimer` taşır.
> Simülasyon çıktısı ayrıca `is_conceptual: true` ve `limitations` taşır. Arayüz bu metinleri
> gizleyemez. Çıktılar klinik tanı/tedavi değildir.

## Veri akışı

| Kimden → Kime | Ne | Biçim | Sözleşme |
|---|---|---|---|
| Nisa → Omics, XAI, Diffusion | patch özellikleri, koordinatlar, ABMIL dikkat, slayt gömmesi | HDF5 | aşağıdaki "Patch özellik dosyası" |
| Omics → GNN, RAG, Frontend | mutasyon olasılığı, belirsizlik, güven aralığı, kalibrasyon sürümü, füzyon dikkati | JSON | `MutationPrediction` |
| GNN → RAG, Frontend | ilaç sıralaması, uyum skoru, hedef protein | JSON | `DrugRecommendation` |
| RAG → Frontend | rapor metni + kaynaklar | JSON | `ReportResponse` |
| Diffusion → Frontend | önce/sonra görüntü (kavramsal) | JSON + görüntü | `SimulationResponse` |
| XAI → Frontend | ısı haritası katmanları, bölgeler | JSON + görüntü | `ExplainResponse` |

Uç noktalar (hepsi `POST`, JSON; `GET /health` ayrıca): `/predict/mutation`, `/recommend/drugs`,
`/report`, `/simulate`, `/explain`.

## Patch özellik dosyası (HDF5)

| Ad | Şekil / tür | Zorunlu | Açıklama |
|---|---|---|---|
| `features` | (N, D) float16 | evet | ViT patch özellik vektörleri |
| `coords` | (N, 2) int32 | evet | patch sol-üst (x, y), seviye-0 piksel, >= 0 |
| `attention` | (N,) float32 | hayır | ABMIL dikkat ağırlığı, >= 0, toplam ~1 |
| `slide_embedding` | (D,) float32 | hayır | slayt düzeyi gömme |
| attr `schema_version` | str | evet | MAJOR uyuşmalı |
| attr `slide_id` | str | evet | anonim anahtar |
| attr `magnification` | float | evet | örn. 20.0 / 40.0 |
| attr `encoder_name`, `encoder_version` | str | evet | hangi ViT, hangi sürüm |
| attr `patch_size_px` | int | evet | patch kenarı (piksel) |

Doğrulama: `onkos.contracts.patch_features.validate_patch_feature_file(path)`.
Sahte dosya: `write_mock_patch_features(path)`. Modül 2/4/5 gerçek özellikleri beklemeden bununla başlar.

## Kimlik ve gizlilik kuralları
- Hasta kimliği taşıyan alan yoktur. Slaytlar anonim `slide_id` ile anılır (4-64 karakter).
- **TCGA barkodu (`TCGA-XX-XXXX…`) `slide_id` olarak reddedilir.** Veri hattı, barkodu `slide_id`'ye
  çeviren eşlemeyi yalnızca kendi içinde tutar; bu eşleme repoya girmez.
- Alan adları İngilizce `snake_case`; açıklamalar Türkçe.

## Sürümleme ve geriye dönük uyumluluk
- `schema_version = MAJOR.MINOR.PATCH` (şu an `onkos/contracts/__init__.py` içindeki `SCHEMA_VERSION`).
- **PATCH:** yalnızca açıklama/doğrulama mesajı gibi davranışı değiştirmeyen düzeltmeler.
- **MINOR:** yalnızca yeni **isteğe bağlı** alan veya yeni uç nokta eklenir. Mevcut alan silinmez,
  yeniden adlandırılmaz, zorunlu hale gelmez, türü/aralığı daraltılmaz.
- **MAJOR:** kırıcı değişiklik. Mesajlar MAJOR uyuşmazsa `422` ile reddedilir.
- Şemalar bilinmeyen alanı reddeder (`extra="forbid"`); bu yüzden şema tek yerde, tüm monorepo için
  birlikte sürümlenir.

## Hata standardı
Hatalar `application/problem+json` (RFC 7807 biçimi) ve `ProblemDetails` şemasıyla döner:
`title`, `status`, `code` (makine kodu), `detail`, `instance`, `errors[]` (alan düzeyi).
Kodlar: `validation_error`, `slide_not_found`, `features_incompatible`, `calibration_missing`,
`model_unavailable`, `rate_limited`, `internal_error`.

## Mock ile bağımsız geliştirme
```bash
uv run python scripts/mock_server.py 8000      # frontend bunu çağırır (CORS açık, yalnızca geliştirme)
curl -s -X POST localhost:8000/predict/mutation -H 'Content-Type: application/json' \
     -d @onkos/contracts/examples/mutation_request.json
```
- Mock değerler rastgeledir; **biyolojik/klinik anlamı yoktur**. Metinler `[MOCK]` ile işaretlidir;
  kaynakçalar yer tutucudur, gerçek atıf değildir. Mock'taki DrugBank kimlikleri gerçek kullanımdan
  önce doğrulanmalıdır.
- Backend (İdil) gerçek uç noktaları yazarken aynı Pydantic modellerini kullanır; Yusuf mock sunucuya
  karşı çalışır; aynı JSON şekli iki taraf için geçerlidir (testlerle doğrulanır).

## Sözleşme değişiklik süreci (kim onaylar, nasıl sürümlenir)
1. `onkos/contracts/schemas.py` (ve gerekirse `consumers.py`) değiştirilir.
2. `uv run python scripts/export_contracts.py` çalıştırılır; `openapi.yaml`, `examples/` ve alan
   referansı güncellenir. **Bu çıktılar commit'e girmezse CI kırılır.**
3. Uyumluluk kuralına göre `SCHEMA_VERSION` artırılır.
4. PR açılır. `.github/CODEOWNERS` gereği sözleşme sahibi, ayrıca etkilenen tüketicilerin sahipleri
   (`consumers.py` hangi alanlara dayandıklarını listeler) onaylar.
5. **CI neleri yakalar:** (a) üretilen dosyaların güncelliği, (b) örneklerin şemaya uyumu,
   (c) `consumers.py`'deki alanlardan biri silinir/adı değişirse hangi tüketicinin bozulduğunu
   adıyla söyleyen test.
6. Kırıcı değişiklik (MAJOR) için önce ekip toplantısında karar alınır ve ADR yazılır.

## Sınırlılıklar (dürüstlük notu)
- `consumers.py` yalnızca listelenen alanları korur; bir tüketici yeni bir alana bağımlı olursa o
  tüketicinin sahibi listeye eklemelidir.
- OpenAPI belgesi Pydantic modellerinden üretilir; henüz gerçek bir FastAPI uygulaması yoktur
  (İdil'in sprintinde yazılacak). Gerçek uygulama bu belgeyle uyumsuz çıkarsa belge değil, uygulama düzeltilir.
- Görüntü/ısı haritası içeriğinin dosya biçimi (PNG/NPY, renk skalası) henüz dondurulmadı; `uri`
  alanı biçimden bağımsız tutuldu. Yusuf ve proje lideri karar verince `HeatmapLayer`'a eklenir (MINOR).
