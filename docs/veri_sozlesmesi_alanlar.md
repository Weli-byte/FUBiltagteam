# Veri Sözleşmesi: Alan Referansı

> Sürüm 0.1.0. Bu dosya `scripts/export_contracts.py` ile `onkos/contracts/schemas.py`'den üretilir; elle düzenlemeyin.

## Citation

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `pmid` | string | null | hayır | default=None | PubMed kimliği. |
| `title` | string | evet | minLength=1 | Kaynak başlığı. |
| `year` | integer | null | hayır | default=None | Yayın yılı. |

## ConfidenceInterval

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `lower` | number | evet | minimum=0.0; maximum=1.0 | Güven aralığı alt sınırı (olasılık, 0-1). |
| `upper` | number | evet | minimum=0.0; maximum=1.0 | Güven aralığı üst sınırı (olasılık, 0-1). |
| `level` | number | hayır | exclusiveMinimum=0.0; exclusiveMaximum=1.0; default=0.95 | Aralığın güven düzeyi (örn. 0.95). |

## DrugCandidate

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `rank` | integer | evet | minimum=1 | Sıra (1 = en uyumlu). |
| `drug_name` | string | evet | minLength=1 | İlaç adı (etken madde). |
| `drugbank_id` | string | null | hayır | default=None | DrugBank kimliği (örn. DB00530). |
| `compatibility_score` | number | evet | minimum=0.0; maximum=1.0 | Mutasyon profili ile uyum skoru (0-1); göreli sıralama içindir, klinik etki olasılığı DEĞİLDİR. |
| `target_proteins` | list[string] | hayır |  | Hedef proteinler. |
| `rationale` | string | null | hayır | default=None | Kısa gerekçe metni (varsa). |

## DrugRecommendation

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `schema_version` | string | hayır | pattern=`^\d+\.\d+\.\d+$`; default='0.1.0' | Sözleşme sürümü (MAJOR.MINOR.PATCH). MAJOR uyuşmazsa mesaj reddedilir. |
| `research_use_only` | const True | hayır | default=True | Her zaman true: çıktı yalnızca araştırma amaçlıdır. |
| `disclaimer` | string | hayır | default="Araştırma prototipi çıktısıdır; klinik tanı veya tedavi kararı için kullanılamaz ve 'kesin tanı' ya da 'reçete' değildir." | Kullanıcıya gösterilecek uyarı metni. |
| `slide_id` | string | evet | pattern=`^[A-Za-z0-9_-]{4,64}$` | Anonim slayt anahtarı (4-64 karakter: harf, rakam, _ ve -). Hasta kimliği veya TCGA barkodu OLAMAZ. |
| `candidates` | list[DrugCandidate] | evet | minItems=1 | Uyum skoruna göre azalan sırada ilaç adayları. |
| `model_version` | string | evet | minLength=1 | GNN modelinin sürümü. |
| `knowledge_base_version` | string | null | hayır | default=None | Kullanılan ilaç veritabanı sürümü (örn. DrugBank sürümü). |

## DrugRecommendationRequest

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `schema_version` | string | hayır | pattern=`^\d+\.\d+\.\d+$`; default='0.1.0' | Sözleşme sürümü (MAJOR.MINOR.PATCH). MAJOR uyuşmazsa mesaj reddedilir. |
| `slide_id` | string | evet | pattern=`^[A-Za-z0-9_-]{4,64}$` | Anonim slayt anahtarı (4-64 karakter: harf, rakam, _ ve -). Hasta kimliği veya TCGA barkodu OLAMAZ. |
| `mutations` | list[GeneMutationScore] | evet | minItems=1 | Mutasyon profili (Modül 2 çıktısındaki skorlar). |
| `top_k` | integer | hayır | minimum=1; maximum=20; default=5 | İstenen en fazla ilaç sayısı (1-20). |

## ErrorCode

Machine-readable error codes returned in :class:`ProblemDetails`.

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|

## ExplainMethod

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|

## ExplainRequest

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `schema_version` | string | hayır | pattern=`^\d+\.\d+\.\d+$`; default='0.1.0' | Sözleşme sürümü (MAJOR.MINOR.PATCH). MAJOR uyuşmazsa mesaj reddedilir. |
| `slide_id` | string | evet | pattern=`^[A-Za-z0-9_-]{4,64}$` | Anonim slayt anahtarı (4-64 karakter: harf, rakam, _ ve -). Hasta kimliği veya TCGA barkodu OLAMAZ. |
| `method` | ExplainMethod | evet |  | Açıklama yöntemi. |
| `target_gene` | Gene | null | hayır | default=None | Açıklanacak gen (varsa). |

## ExplainResponse

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `schema_version` | string | hayır | pattern=`^\d+\.\d+\.\d+$`; default='0.1.0' | Sözleşme sürümü (MAJOR.MINOR.PATCH). MAJOR uyuşmazsa mesaj reddedilir. |
| `research_use_only` | const True | hayır | default=True | Her zaman true: çıktı yalnızca araştırma amaçlıdır. |
| `disclaimer` | string | hayır | default="Araştırma prototipi çıktısıdır; klinik tanı veya tedavi kararı için kullanılamaz ve 'kesin tanı' ya da 'reçete' değildir." | Kullanıcıya gösterilecek uyarı metni. |
| `slide_id` | string | evet | pattern=`^[A-Za-z0-9_-]{4,64}$` | Anonim slayt anahtarı (4-64 karakter: harf, rakam, _ ve -). Hasta kimliği veya TCGA barkodu OLAMAZ. |
| `slide_width` | integer | evet | exclusiveMinimum=0 | Slayt genişliği (seviye-0 piksel). |
| `slide_height` | integer | evet | exclusiveMinimum=0 | Slayt yüksekliği (seviye-0 piksel). |
| `tile_size` | integer | evet | exclusiveMinimum=0 | Isı haritası döşeme boyutu (piksel). |
| `layers` | list[HeatmapLayer] | evet | minItems=1 | Isı haritası katmanları. |
| `regions` | list[Region] | hayır |  | Vurgulanan bölgeler. |
| `summary_text` | string | null | hayır | default=None | Kısa açıklama metni (varsa). |

## FieldError

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `loc` | list[string | integer] | evet |  | Hatalı alanın yolu. |
| `msg` | string | evet |  | Hata açıklaması. |
| `type` | string | evet |  | Hata türü. |

## FusionAttention

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `image_weight` | number | evet | minimum=0.0; maximum=1.0 | Görüntü modalitesinin füzyon ağırlığı (0-1). |
| `omics_weight` | number | evet | minimum=0.0; maximum=1.0 | Omics modalitesinin füzyon ağırlığı (0-1). |
| `top_patches` | list[PatchAttention] | hayır | maxItems=50 | En yüksek dikkatli patch'ler (<= 50). |

## Gene

Genes with a mutation-prediction head (extend here when a new gene is added).

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|

## GeneMutationScore

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `gene` | Gene | evet |  | Hedef gen. |
| `probability` | number | evet | minimum=0.0; maximum=1.0 | Kalibre edilmiş mutasyon olasılığı (0-1). |
| `uncertainty` | number | evet | minimum=0.0 | Belirsizlik: tahminin standart sapması (olasılık biriminde, >= 0). |
| `ci` | ConfidenceInterval | evet |  | Olasılık için güven aralığı. |

## HealthResponse

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `schema_version` | string | hayır | pattern=`^\d+\.\d+\.\d+$`; default='0.1.0' | Sözleşme sürümü (MAJOR.MINOR.PATCH). MAJOR uyuşmazsa mesaj reddedilir. |
| `status` | const ok | hayır | default='ok' |  |

## HeatmapLayer

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `name` | string | evet | minLength=1 | Katman adı (arayüzde gösterilir). |
| `method` | ExplainMethod | evet |  |  |
| `uri` | string | evet | minLength=1 | Isı haritası görüntüsü/dizisi konumu. |
| `width` | integer | evet | exclusiveMinimum=0 | Katman genişliği (piksel). |
| `height` | integer | evet | exclusiveMinimum=0 | Katman yüksekliği (piksel). |
| `value_min` | number | hayır | default=0.0 | Renk skalasının alt değeri. |
| `value_max` | number | hayır | default=1.0 | Renk skalasının üst değeri. |

## MutationPrediction

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `schema_version` | string | hayır | pattern=`^\d+\.\d+\.\d+$`; default='0.1.0' | Sözleşme sürümü (MAJOR.MINOR.PATCH). MAJOR uyuşmazsa mesaj reddedilir. |
| `research_use_only` | const True | hayır | default=True | Her zaman true: çıktı yalnızca araştırma amaçlıdır. |
| `disclaimer` | string | hayır | default="Araştırma prototipi çıktısıdır; klinik tanı veya tedavi kararı için kullanılamaz ve 'kesin tanı' ya da 'reçete' değildir." | Kullanıcıya gösterilecek uyarı metni. |
| `slide_id` | string | evet | pattern=`^[A-Za-z0-9_-]{4,64}$` | Anonim slayt anahtarı (4-64 karakter: harf, rakam, _ ve -). Hasta kimliği veya TCGA barkodu OLAMAZ. |
| `predictions` | list[GeneMutationScore] | evet | minItems=1 | Gen başına mutasyon skorları (her gen en fazla bir kez). |
| `calibration_version` | string | evet | minLength=1 | Kalibrasyon modülünün sürümü (örn. 'isotonic-2026-01'). |
| `model_version` | string | evet | minLength=1 | Mutasyon modelinin sürümü. |
| `fusion_attention` | FusionAttention | null | hayır | default=None | Cross-attention füzyon dikkat ağırlıkları (varsa). |

## MutationPredictionRequest

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `schema_version` | string | hayır | pattern=`^\d+\.\d+\.\d+$`; default='0.1.0' | Sözleşme sürümü (MAJOR.MINOR.PATCH). MAJOR uyuşmazsa mesaj reddedilir. |
| `slide_id` | string | evet | pattern=`^[A-Za-z0-9_-]{4,64}$` | Anonim slayt anahtarı (4-64 karakter: harf, rakam, _ ve -). Hasta kimliği veya TCGA barkodu OLAMAZ. |
| `patch_features_uri` | string | evet | minLength=1 | Nisa'nın ürettiği patch özellik HDF5 dosyasının konumu (dosya yolu veya URI). |
| `omics_sample_id` | string | null | hayır | default=None | Varsa omics örneğinin anonim anahtarı; yoksa yalnızca görüntü. |
| `genes` | list[Gene] | hayır | minItems=1 | Tahmin istenen genler (varsayılan: hepsi). |

## PatchAttention

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `x` | integer | evet | minimum=0 | Patch sol-üst köşesi, seviye-0 piksel X koordinatı. |
| `y` | integer | evet | minimum=0 | Patch sol-üst köşesi, seviye-0 piksel Y koordinatı. |
| `weight` | number | evet | minimum=0.0; maximum=1.0 | Patch'e verilen normalize dikkat ağırlığı (0-1). |

## PatchRef

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `x` | integer | evet | minimum=0 | Patch sol-üst köşesi, seviye-0 piksel X koordinatı. |
| `y` | integer | evet | minimum=0 | Patch sol-üst köşesi, seviye-0 piksel Y koordinatı. |

## ProblemDetails

Error body (RFC 7807 style), served as ``application/problem+json``.

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `type` | string | hayır | default='about:blank' | Hata türünü tanımlayan URI. |
| `title` | string | evet | minLength=1 | Kısa, insan okuyabilir başlık. |
| `status` | integer | evet | minimum=100; maximum=599 | HTTP durum kodu. |
| `detail` | string | null | hayır | default=None | Bu hataya özgü açıklama. |
| `code` | ErrorCode | evet |  | Makine tarafından okunabilir hata kodu. |
| `instance` | string | null | hayır | default=None | Hatanın oluştuğu istek yolu. |
| `errors` | list[FieldError] | null | hayır | default=None | Alan düzeyinde hatalar. |

## Region

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `x` | integer | evet | minimum=0 | Sol-üst X (seviye-0 piksel). |
| `y` | integer | evet | minimum=0 | Sol-üst Y (seviye-0 piksel). |
| `width` | integer | evet | exclusiveMinimum=0 | Genişlik (piksel). |
| `height` | integer | evet | exclusiveMinimum=0 | Yükseklik (piksel). |
| `score` | number | evet | minimum=0.0; maximum=1.0 | Bölgenin önem skoru (0-1). |

## ReportRequest

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `schema_version` | string | hayır | pattern=`^\d+\.\d+\.\d+$`; default='0.1.0' | Sözleşme sürümü (MAJOR.MINOR.PATCH). MAJOR uyuşmazsa mesaj reddedilir. |
| `slide_id` | string | evet | pattern=`^[A-Za-z0-9_-]{4,64}$` | Anonim slayt anahtarı (4-64 karakter: harf, rakam, _ ve -). Hasta kimliği veya TCGA barkodu OLAMAZ. |
| `mutation_prediction` | MutationPrediction | evet |  |  |
| `drug_recommendation` | DrugRecommendation | null | hayır | default=None | Varsa ilaç önerisi; rapor bunu da özetler. |
| `language` | enum(tr, en) | hayır | default='tr' | Rapor dili. |

## ReportResponse

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `schema_version` | string | hayır | pattern=`^\d+\.\d+\.\d+$`; default='0.1.0' | Sözleşme sürümü (MAJOR.MINOR.PATCH). MAJOR uyuşmazsa mesaj reddedilir. |
| `research_use_only` | const True | hayır | default=True | Her zaman true: çıktı yalnızca araştırma amaçlıdır. |
| `disclaimer` | string | hayır | default="Araştırma prototipi çıktısıdır; klinik tanı veya tedavi kararı için kullanılamaz ve 'kesin tanı' ya da 'reçete' değildir." | Kullanıcıya gösterilecek uyarı metni. |
| `slide_id` | string | evet | pattern=`^[A-Za-z0-9_-]{4,64}$` | Anonim slayt anahtarı (4-64 karakter: harf, rakam, _ ve -). Hasta kimliği veya TCGA barkodu OLAMAZ. |
| `report_text` | string | evet | minLength=1 | Doktorun okuyacağı rapor metni. |
| `citations` | list[Citation] | hayır |  | Rapora dayanak kaynaklar (RAG). |
| `language` | enum(tr, en) | evet |  |  |
| `llm_model` | string | evet | minLength=1 | Raporu üreten dil modelinin adı/sürümü. |
| `retrieval_index_version` | string | null | hayır | default=None | Kullanılan literatür vektör indeksinin sürümü. |

## SimulatedImage

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `role` | enum(before, after) | evet |  | Tedavi öncesi mi sonrası mı. |
| `uri` | string | evet | minLength=1 | Görüntü konumu (dosya yolu veya URL). |
| `width` | integer | evet | exclusiveMinimum=0 | Genişlik (piksel). |
| `height` | integer | evet | exclusiveMinimum=0 | Yükseklik (piksel). |

## SimulationRequest

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `schema_version` | string | hayır | pattern=`^\d+\.\d+\.\d+$`; default='0.1.0' | Sözleşme sürümü (MAJOR.MINOR.PATCH). MAJOR uyuşmazsa mesaj reddedilir. |
| `slide_id` | string | evet | pattern=`^[A-Za-z0-9_-]{4,64}$` | Anonim slayt anahtarı (4-64 karakter: harf, rakam, _ ve -). Hasta kimliği veya TCGA barkodu OLAMAZ. |
| `drug_name` | string | evet | minLength=1 | Simüle edilecek ilaç. |
| `region` | PatchRef | null | hayır | default=None | Simüle edilecek patch; yoksa sistem seçer. |
| `seed` | integer | hayır | minimum=0; default=0 | Tekrarlanabilirlik için rastgelelik tohumu. |
| `num_inference_steps` | integer | hayır | minimum=1; maximum=1000; default=50 | Difüzyon örnekleme adım sayısı (1-1000). |

## SimulationResponse

| Alan | Tür | Zorunlu | Kısıtlar | Anlam |
|---|---|---|---|---|
| `schema_version` | string | hayır | pattern=`^\d+\.\d+\.\d+$`; default='0.1.0' | Sözleşme sürümü (MAJOR.MINOR.PATCH). MAJOR uyuşmazsa mesaj reddedilir. |
| `research_use_only` | const True | hayır | default=True | Her zaman true: çıktı yalnızca araştırma amaçlıdır. |
| `disclaimer` | string | hayır | default="Araştırma prototipi çıktısıdır; klinik tanı veya tedavi kararı için kullanılamaz ve 'kesin tanı' ya da 'reçete' değildir." | Kullanıcıya gösterilecek uyarı metni. |
| `slide_id` | string | evet | pattern=`^[A-Za-z0-9_-]{4,64}$` | Anonim slayt anahtarı (4-64 karakter: harf, rakam, _ ve -). Hasta kimliği veya TCGA barkodu OLAMAZ. |
| `drug_name` | string | evet |  |  |
| `images` | list[SimulatedImage] | evet | minItems=2 | En az bir 'before' ve bir 'after' görüntüsü. |
| `seed` | integer | evet | minimum=0 |  |
| `model_version` | string | evet | minLength=1 | Difüzyon modelinin sürümü. |
| `is_conceptual` | const True | hayır | default=True | Her zaman true: simülasyon kavramsaldır, doğrulanmamıştır. |
| `limitations` | string | hayır | default='Kavramsal dijital ikiz gösterimidir; tedavi öncesi/sonrası eşleşmiş görüntü ile doğrulanmamıştır ve klinik olarak geçerli bir tedavi tahmini değildir.' | Arayüzde gösterilecek sınırlılık metni. |
