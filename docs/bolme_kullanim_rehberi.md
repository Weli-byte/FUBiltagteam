# Veri Bölme Kullanım Rehberi (Mini sprint 1.6, 1 sayfa)

**Tek bir bölme dosyası vardır ve herkes onu kullanır.** Bölme **hasta** düzeyindedir: bir hastanın tüm
örnekleri, slaytları ve patch'leri aynı kümede kalır. Böylece aynı hasta hem eğitimde hem testte olamaz.

## Üretme (bir kez) ve doğrulama
```bash
uv run python scripts/make_splits.py --version splits-v1     # girdi: patient_map (1.3) + labels (1.5)
uv run python scripts/make_splits.py --verify splits-v1      # dondurulmuş özetle (hash) karşılaştırır
```
Var olan sürümün **üzerine yazılmaz**. Değişiklik gerekirse yeni sürüm açılır (`--version splits-v2`) ve
ekibe duyurulur. Hasta düzeyindeki dosyalar `data/processed/splits/<sürüm>/` altında kalır (git'e girmez);
git'e yalnızca toplu rapor ve `docs/splits/<sürüm>_manifest.json` (sayılar + SHA-256 özetleri) girer.

## Dosyalar (`data/processed/splits/<sürüm>/`)
| Dosya | İçerik |
|---|---|
| `splits.parquet` | hasta başına: `split` (train/val/test), `cv_fold` (0-4; test için -1), `split_site`, proje, TSS, modalite bayrakları, gen etiketleri |
| `slide_splits.parquet` | her tanı slaytı (`slide_file_id`) ↔ `patient_barcode` ↔ küme (**Nisa'nın patch verisi için kimlik sözleşmesi**) |
| `train_ids.txt`, `val_ids.txt`, `test_ids.txt` | hasta kimlik listeleri |
| `train_ids_omics.txt` | eğitim hastalarından RNA-seq'i olanlar: `preprocess_omics.py fit-transform --train-ids ...` girdisi |

## Kurallar (herkes için)
1. **Test kümesi kilitlidir.** Yalnızca **final değerlendirmede** açılır. Model seçimi, hiperparametre ayarı,
   eşik belirleme veya "bir bakayım" için kullanılmaz. Kodda `get_ids(splits, "test")` hata verir; yalnızca
   `final_evaluation=True` ile açılır ve bu bilinçli bir karardır.
2. **Model seçimi için CV:** train+val havuzunda `cv_fold` (5 katman) kullan. Tek seferlik doğrulama için `val`.
3. **Ön işleme parametreleri yalnızca eğitimden öğrenilir:** `fit-transform --train-ids train_ids_omics.txt`.
   CV içinde her katmanda yeniden öğren (katman dışı hastalar eğitimdeymiş gibi davranma).
4. **Ek senaryo `split_site`:** bazı kurumlar (TSS) tamamen test'tedir; görülmemiş kuruma genellemeyi ölçer.
   Ana sonuç `split` ile, bu sonuç ayrıca raporlanır; ikisi karıştırılmaz.
5. Her modül kendi veri kümesini bu dosyaya göre filtreler (`has_rnaseq`, `has_slide`, `labeled` sütunları);
   **kendi bölmesini üretmez**. Etiketi bilinmeyen hastalar (`labeled=False`) eğitim/değerlendirmeye alınmaz
   ama kümeleri belirlidir (görüntü modülü onları kullanabilir).

```python
import pandas as pd
from models.omics.splits import get_ids
splits = pd.read_parquet("data/processed/splits/splits-v1/splits.parquet")
train = get_ids(splits, "train", modality="labeled")        # test için final_evaluation=True gerekir
fold0_val = splits[splits["cv_fold"] == 0]["patient_barcode"]
```

## Nasıl bölünüyor
Çok etiketli tabakalama (iterative stratification): kanser türü, her hedef genin pozitifliği, etiket
bilgisinin varlığı ve modalite (RNA-seq, slayt) varlığı kümelere dengeli dağıtılır. Seed sabittir
(`configs/omics/splits.yaml`), sonuç deterministiktir ve girdi sırasından bağımsızdır.

## Sınırlılıklar (dürüstlük notu)
- **Nadir hedeflerde test kümesi küçüktür.** Örneğin %15'lik bir kümede çok az pozitif örnek düşebilir
  (bölme raporu sayıları gösterir). Bu durumda AUROC güven aralıkları çok geniştir; AUPRC ile birlikte
  güven aralığıyla raporlanmalı ve "bu gen için sonuç güvenilmezdir" diye yazılmalıdır.
- Alt kümede 5'ten az pozitif varsa veya pozitif oranı genelden 3 puandan fazla sapıyorsa bu **uyarı**
  olarak raporlanır (`--strict` ile hata verir), sessizce geçilmez.
- Tabakalama etiketleri kullanır; etiketler 1.5'teki MAF kurallarına bağlıdır. Etiket tanımı değişirse
  bölme yeniden üretilmez, **yeni sürüm** açılır.
- Aynı kurumdan (TSS) hastalar `split` içinde farklı kümelere dağılabilir; bu bilinçlidir. Kurum etkisini
  `split_site` ölçer.
