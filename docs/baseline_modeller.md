# Baseline Omics Modelleri ve Değerlendirme Çerçevesi (Mini sprint 1.8)

Amaç: karmaşık Transformer Omics Encoder ve Cross-Attention füzyonuna (Sprint 2) geçmeden önce **güçlü
ve dürüst referans noktaları** kurmak. "İleri modeller gerçekten fayda sağlıyor mu?" sorusu ancak bu
referanslara karşı yanıtlanabilir. Hedef: TP53, KRAS, EGFR, BRCA1 mutasyon durumu (ikili), girdi: 1.4'ün
ifade matrisi, 1.5'in etiketleri, 1.6'nın hasta bazlı bölmesi.

## Protokol (sızıntı önleme)
| Adım | Kural |
|---|---|
| Veri | Yalnızca **geliştirme havuzu** (train+val, `cv_fold` ≥ 0). Test hastaları daha dizinlenmeden dışarıda bırakılır; bu modülde test değerlendirmesi **yoktur** |
| Dış döngü | Hazır `cv_fold` (5 katman). Her katmanın tahmini bir kez, o katmanı görmemiş modelle yapılır (OOF) |
| Ön işleme | Gen filtresi + yüksek varyanslı gen seçimi + z-skor **her dış katmanda yalnızca o katmanın eğitim satırlarından** öğrenilir (1.4'teki `ExpressionPreprocessor`) |
| İç döngü | Eğitim satırlarında 3 katlı CV ile hiperparametre seçimi (ölçüt: AUPRC) |
| Sınıf dengesizliği | Sınıf ağırlıkları, her katmanın içinde. SMOTE vb. yapay örnekleme **yok** |
| F1 eşiği | Yalnızca eğitim tarafındaki iç-CV tahminlerinden seçilir; raporlanan eşik katman eşiklerinin medyanı |
| Güven aralığı | %95, hasta düzeyinde bootstrap, **≥ 1000** örnek (yapılandırma 1000'in altını reddeder) |

## Modeller
L1 ve L2 lojistik regresyon, Random Forest, gradyan artırma (`gbm`), iki gizli katmanlı MLP. `gbm`,
**XGBoost kuruluysa XGBoost, değilse scikit-learn `HistGradientBoosting`** kullanır; hangisinin kullanıldığı
sonuç tablosundaki `backend` sütununda yazar (XGBoost opsiyoneldir; kurulum büyük ve gerekli değildir).
**Trivial taban:** yalnızca kanser türü (one-hot) ile lojistik regresyon. "Omics, kanser türünün zaten
söylediğinden fazlasını söylüyor mu?" sorusunun cevabı budur.

## Çalıştırma
```bash
uv sync --group data --group omics
# 1) hızlı duman testi (SONUÇLAR NİHAİ DEĞİL): her gridden tek değer, 2 model
uv run python scripts/train_baselines.py --run quick-v1 --splits-version splits-v1 --quick
# 2) gerçek çalıştırma (uzun sürebilir)
uv run python scripts/train_baselines.py --run baseline-v1 --splits-version splits-v1 n_jobs=2
```
Girdiler: `data/processed/omics` (1.4 `build`), `data/processed/labels/labels.parquet` (1.5),
`data/processed/splits/<sürüm>/splits.parquet` (1.6). Çıktılar: `data/processed/baselines/<run>/`
(`results.csv`, `oof_predictions.parquet`; hasta düzeyinde, git'e girmez) ve `docs/baselines/<run>_rapor.md`
+ şekiller (toplu sayılar). **Var olan bir run'ın üzerine yazılmaz.**

## Raporun içeriği
Birleşik (pan-cancer) model tablosu (AUROC, AUPRC, F1, Brier, ECE ve %95 GA), kanser türüne özel
eğitim ve birleşik modelin tür bazında sonucu, **trivial tabanla eşleştirilmiş fark (ΔAUROC, ΔAUPRC
ve GA)**, atlanan kapsamlar, uyarılar ve ROC / precision-recall / kalibrasyon şekilleri.

## Nasıl okunmalı (dürüstlük notları)
- **AUPRC birincil metriktir** (nadir hedefler). Prevalans, bilgisiz bir sınıflandırıcının AUPRC'sidir ve
  yanında verilir.
- Pozitif sayısı < 30 olan kapsamlar eğitilmez, "atlandı" diye listelenir (EGFR/BRCA1'de kanser türüne özel
  modellerin çoğu bu gruba girmesi beklenir; birleşik model kullanılır).
- **En iyi model aynı OOF tahminlerinde seçilir**, bu yüzden sayısı hafif iyimserdir. Tarafsız tahmin, kilitli
  test kümesindeki tek seferlik final değerlendirmeden gelir (Sprint 2).
- `AUROC > 0.95` uyarı olarak işaretlenir (şişirilmiş metrik şüphesi: sızıntıyı araştır).
- "kolay / orta / zor" ve "omics katkısı var / belirsiz" yorumları **kural tabanlı okuma yardımcısıdır**
  (eşikler yazarların seçimi, veriye göre ayarlanmadı). Biyolojik yorumu ekip yapar.
- Kalibrasyon düzeltmesi yapılmaz (Sprint 2.9); Brier/ECE ham olasılıkları ölçer.
- Bayraklı (QC) örnekler `drop_flagged: false` ile **dahildir**; bu bir yapılandırma kararıdır.

## Sınırlılıklar
- Sürelerin gerçek veride ne kadar olduğu **bilinmiyor** (tahmin: `--quick` birkaç dakika-yarım saat, tam
  çalıştırma saatler). Yalnızca sentetik veriyle denendi.
- Özellik kümesi sabit (`protein_coding`, 2000 gen); farklı gen sayıları 1.4'teki çıktıyla ve yapılandırmayla
  denenebilir, ama seçim **doğrulama/CV'ye bakarak** yapılmalı, test'e bakarak değil.
- MLP sınıf ağırlığını örnek ağırlığıyla uygular; PyTorch tabanlı MLP bu sprintte gerekmedi.
