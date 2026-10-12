# Bulut GPU ve Maliyet Stratejisi (Mini sprint 1.1, Nisa)

> **Durum:** Taslak; ekip onayı bekleniyor. Fiyatlar 13 Ekim 2026'da resmi sayfalardan okundu
> (USD, vergi hariç) ve her satırda kaynak verildi. Fiyatlar değişir: harcama yapmadan önce
> bağlantıyı yeniden açıp doğrulayın. "Doğrulanamadı" yazan yerler hafızadan yazılmadı, resmi
> sayfada bulunamadı demektir.

**Bu belge ne işe yarar? (sade anlatım)** Yapay zeka eğitmek için güçlü ekran kartları (GPU)
gerekir; satın almak yerine saatlik kiralarız. Bu belge, telefon tarifesi seçer gibi, hangi
hizmeti ne zaman kullanacağımızı, aylık harcama sınırımızı ve "fatura şoku" yaşamamak için
alarm/otomatik kapatma düzenini belirler.

## 1. Ekibin 12 aylık GPU ihtiyacı (saat)

Tahminler iş yükü x epoch x batch süresi mantığıyla yapıldı ve **%30 güvenlik payı** eklendi.
Bellek sütunu, işin rahat çalışması için gereken en küçük GPU belleğidir.

| İş | Kişi | Dönem | Bellek | Saat (tahmin) | +%30 pay | Dayanak |
|---|---|---|---|---|---|---|
| ViT-S/16 ve ResNet-50 patch baseline (2 model x 3 seed, ~300k patch, 10 epoch) | Nisa | Sprint 1-2 | 8 GB | 20 | 26 | 224 px, batch 64, fp16: ~1-2 sa/koşu (T4'te 2-3x yavaş) |
| Patch özellik çıkarımı: ~2.000 TCGA tanı slaydı + 399 CAMELYON16 slaydı | Nisa | Sprint 2 | 8 GB | 100 | 130 | ViT-S/16 fp16 ~1.500 patch/sn; darboğaz OpenSlide okuma (CPU) → ~2-3 dk/slayt |
| ABMIL eğitimi (özellik düzeyi, 3 seed, hiperparametre taraması) | Nisa | Sprint 2 | 4 GB | 15 | 20 | Özellikler hazırsa slayt başına ms; çoğu zaman CPU'da bile olur |
| ViT iyileştirme / patoloji temel modeli (UNI/CONCH) ile yeniden çıkarım | Nisa | Sprint 3 | 16 GB | 60 | 78 | ViT-L boyutu: 3-4x daha yavaş çıkarım |
| Omics Encoder + Cross-Attention (küçük Transformer) | Proje lideri | Sprint 2 | 8 GB | 30 | 39 | Örnek sayısı ~1.600; dakikalar/epoch |
| LLM QLoRA ince ayarı (7B, 4-bit) | Proje lideri | Sprint 3 | 24 GB | 40 | 52 | 1-3 epoch, ~10k örnek; A10G/L4 24 GB sınırda, A100 rahat |
| Conditional Diffusion (256 px patch, ilk prototip + final) | Proje lideri | Sprint 3-4 | 40 GB | 200 | 260 | En belirsiz kalem: 100-400 sa aralığı; checkpoint zorunlu |
| GNN ilaç önerisi | İdil | Sprint 2-3 | 8 GB | 20 | 26 | Küçük graflar; CPU'da da çalışır |
| XAI (GradCAM / attention haritaları) | Yusuf | Sprint 3-4 | 8 GB | 10 | 13 | Çıkarım ağırlıklı |
| **Toplam** | | | | **495** | **~645** | |

Belirsizlik notu: Diffusion ve LLM kalemleri toplamın yarısından fazlasıdır ve en az
doğrulanmış tahminlerdir; Sprint 3 başında ilk 10 saatlik ölçümle güncellenecek.

## 2. Sağlayıcı karşılaştırması (13.10.2026)

Hepsi resmi fiyat sayfalarından; AWS/GCP sayfaları JavaScript ile yüklendiği için değerler
sayfanın beslendiği resmi fiyat listesi JSON'larından okundu (K3, K7, K24, K25).

| Sağlayıcı | GPU (bellek) | Saatlik (USD) | Avantaj | Risk / kısıt | Kaynak |
|---|---|---|---|---|---|
| Google Colab Pro | T4/L4/A100 (değişken) | 9,99/ay, 100 CU | Kurulumsuz, öğrenciye tanıdık | 12 sa oturum, GPU türü garantisiz, CU tükenir | K1, K2 |
| Google Colab Pro+ | T4/L4/A100/H100 (değişken) | 49,99/ay, 600 CU | Arka plan çalışma 24 sa, premium GPU önceliği | CU/saat oranı resmi değil (doğrulanamadı); uzun eğitim için checkpoint şart | K1, K2 |
| AWS EC2 g4dn.xlarge | T4 (16 GB) | 0,526 | Ucuz giriş | 16 GB: ViT-B/LLM için dar | K3 |
| AWS EC2 g5.xlarge | A10G (24 GB) | 1,006 | Dengeli; spot %56 indirim | **Kota varsayılan 0 vCPU**, talep gerekir | K3, K6, K8 |
| AWS EC2 g6.xlarge | L4 (24 GB) | 0,8048 | g5'ten ucuz, benzer bellek | Kota 0; spot kesinti >%20 | K3, K6 |
| AWS EC2 p4d.24xlarge | 8x A100 40 GB | 21,96 | Büyük iş | Tek GPU alınamaz; pahalı | K3 |
| AWS SageMaker ml.g5.xlarge | A10G | 1,408 | Yönetilen notebook | EC2'den %40 pahalı | K7 |
| GCP g2-standard-4 | L4 (24 GB) | 0,707 (spot 0,423) | Spot ucuz | GPU kotası bölge+global, talep gerekir; **ücretsiz deneme hesabı kota artıramaz** | K10, K11, K12 |
| GCP a2-highgpu-1g | A100 40 GB | 3,67 (spot 2,20) | Tek A100 alınabilir | Pahalı on-demand | K10, K11 |
| GCP a2-ultragpu-1g | A100 80 GB | 5,07 (spot 3,04) | Diffusion için bellek | | K10, K11 |
| Lambda | A100 40 GB (1x) | 1,99 | Dakika bazlı, egress ücretsiz | Stok dalgalı | K13 |
| Lambda | H100 PCIe (1x) | 3,29 | Hızlı | | K13 |
| RunPod | A100 80 GB | 1,79 | En ucuz A100 (on-demand) | Community/Secure ayrımı sayfada net değil (doğrulanamadı) | K14 |
| RunPod | L4 (24 GB) / RTX 4090 | 0,59 / 0,89 | Ucuz 24 GB | Konteyner tabanlı; kalıcı disk ayrı ücret | K14 |
| Vast.ai | RTX 3090 / 4090 / A100 80 | medyan 0,19 / 0,47 / 0,71-0,83 | En ucuz pazar yeri | Güvenilirlik host'a bağlı; hasta verisi için uygun değil (TCGA/CAMELYON anonim ama yine de dikkat) | K15 |

Ücretsiz/akademik seçenekler (K16-K23): Google Cloud Research Credits **5.000 USD** (öğretim
üyesi/doktora; Türkiye listede; proje danışmanı başvurmalı), Azure for Students 100 USD/kişi,
GitHub Student Pack (Azure 100 USD + servisler; DigitalOcean artık yok), AWS Educate (kredi yok,
yalnızca eğitim), AWS Activate (yalnızca şirketleşmiş startup), NVIDIA Academic Grant (şu an
başvuru almıyor), NVIDIA Inception (yalnızca startup).

## 3. Karar önerisi (ekip onaylayacak)

1. **Günlük geliştirme ve küçük deneyler:** her üyenin kendi bilgisayarı + **Colab Pro (9,99
   USD/ay, tek hesap: Nisa)**; Sprint 2'de özellik çıkarımı başlayınca **Pro+'a geçiş (49,99)**.
   Colab'da her koşu `CheckpointManager` ile 10 dakikada bir checkpoint alır (oturum kopar).
2. **Uzun/büyük işler (özellik çıkarımı, LLM, Diffusion):** saatlik kiralık tek GPU:
   birinci tercih **RunPod A100 80 GB (1,79 USD/sa)** veya **Lambda A100 (1,99)**; AWS/GCP
   yalnızca kredi gelirse (kota talebi şimdiden gönderilir, günler sürebilir).
3. **Kredi başvuruları (takvim §5):** Google Research Credits (danışman), Azure for Students
   (4 kişi), GitHub Student Pack (4 kişi).
4. **Veri depolama:** büyük WSI verisi bulut nesne depolamada tutulmaz (1,7 TB TCGA + 0,75 TB
   CAMELYON16 → S3'te ~56 USD/ay + egress 0,09 USD/GB; K24, K25). Veri **tek bir GPU
   makinesinin yerel/kalıcı diskinde** veya ekip NAS'ında tutulur; yalnızca patch özellikleri
   (HDF5, ~1 GB/100 slayt) ve küçük tablolar DVC uzak deposuna (Google Drive) gider.

## 4. Aylık maliyet tahmini ve alarm eşikleri

Varsayımlar: 645 GPU saatinin %35'i Colab'da (abonelik içinde), %65'i (~420 sa) kiralık A100
(ortalama 2 USD/sa) + depolama/egress payı 10 USD/ay.

| Kalem | Aylık (USD) | 12 ay (USD) |
|---|---|---|
| Colab Pro → Pro+ (3 ay Pro, 9 ay Pro+) | 10-50 | 480 |
| Kiralık A100 (420 sa x 2 USD, Sprint 2-4'e yayılmış) | 0-150 | 840 |
| Depolama / egress / küçük disk | 10 | 120 |
| **Toplam** | **~120 ort.** | **~1.440** |
| Kredi senaryosu (Google 5.000 USD + Azure 400 USD) | | nakit ihtiyacı → ~500 |

**Alarm eşikleri (kesin):** aylık üst sınır **200 USD**; e-posta uyarısı 50 / 100 / 150 USD'de.
AWS Budgets (ücretsiz, K27) ve GCP Budgets (K28) her hesapta kurulur; RunPod/Lambda'da ön ödemeli
bakiye kullanılır (kredi kartı limiti = doğal sınır). Her kiralık makinede `infra/gpu/auto_shutdown.sh`
(60 dk boşta → kapat) zorunludur; çalıştırılmayan makine **o gün** kapatılır.

## 5. Kredi/hibe başvuru takvimi

| Program | Kim başvurur | Ne zaman | Beklenen | Kaynak |
|---|---|---|---|---|
| Google Cloud Research Credits | Proje danışmanı (öğretim üyesi) | Sprint 1, hafta 2 | 5.000 USD, 365 gün | K16, K17 |
| Azure for Students | 4 üye, okul e-postası | Sprint 1, hafta 1 | 4 x 100 USD | K18 |
| GitHub Student Developer Pack | 4 üye | Sprint 1, hafta 1 | Azure 100 USD, Copilot, Codespaces | K19 |
| AWS kota artışı (G/VT on-demand, 4-8 vCPU) | Nisa | Sprint 1, hafta 1 (kredi gelirse kullanılır) | Onay 1-5 gün | K8, K9 |
| GCP GPU kotası (L4/A100, 1 adet, us-central1) | Nisa | Kredi onayından hemen sonra | Deneme hesabında talep edilemez | K12 |
| TÜBİTAK 2209-A bütçesi (GPU kalemi) | İdil (başvuru sahibi) | Takvime göre | Resmi bütçe | - |

## 6. Kota talebi şablonu (AWS/GCP formuna yapıştırılır, İngilizce)

```
Use case: Academic research prototype (university team of 4, project ONKOS: histopathology +
omics deep learning, non-clinical). We need ONE GPU instance (L4/A10G/A100) for intermittent
training runs of 2-10 hours, a few times per week over 12 months. Budget alarms and automatic
idle shutdown are configured. Requested quota: Running On-Demand G and VT instances = 8 vCPUs
(us-east-1) [GCP: NVIDIA_L4_GPUS = 1, GPUS_ALL_REGIONS = 1, region us-central1].
Data: public, de-identified datasets (TCGA, CAMELYON16). No PHI.
```

## 7. Ekip erişim ve kullanım planı

- Tek bulut hesabı (sahibi: Nisa), **IAM/alt kullanıcı** ile üyelere erişim; kök anahtar
  paylaşılmaz; anahtarlar `.env`'de, repoya girmez.
- GPU makinesi **SSH anahtarıyla** (parola kapalı), Jupyter/MLflow yalnızca localhost + SSH tüneli
  (`docs/gpu_ortam_rehberi.md`).
- Rezervasyon: ekip takviminde "GPU" etkinliği; aynı anda tek kullanıcı; iş bitince makine
  kapatılır ve takvime "kapatıldı" yazılır.
- Her koşu MLflow'a yazar (run adı `<modül>-<deney>-<seed>`), checkpoint'ler kalıcı diske gider.

## 8. Riskler ve yedek plan

| Risk | Önlem |
|---|---|
| Kota reddi (AWS/GCP) | RunPod/Lambda ön ödemeli; Colab Pro+ |
| Spot/preemptible kesintisi | Yalnızca on-demand uzun koşu; spot sadece 1 saatten kısa denemeler; her koşu resume edebilir |
| Kredi bitmesi / bütçe aşımı | Alarmlar + 200 USD üst sınır; Diffusion kapsamı küçültülür (daha az adım/çözünürlük) |
| Veri kaybı (makine silinir) | Ham veri yeniden indirilebilir (GDC/AWS Open Data); özellik dosyaları DVC'de |
| Colab'da oturum kopması | 10 dk checkpoint; `scripts/*` `--resume` ile devam |

## Doğrulama (bitti kriteri)

- [ ] Saat tahminleri ekip toplantısında kişi başı gözden geçirildi (özellikle Diffusion/LLM).
- [ ] Seçilen sağlayıcı ve 200 USD/ay sınırı onaylandı.
- [ ] Kota talepleri (AWS G/VT, GCP GPU) sprintin ilk haftasında gönderildi.
- [ ] Azure for Students / GitHub Pack / Google Research Credits başvuruları yapıldı.
- [ ] Her fiyat satırının kaynağı açılıp güncel değerle karşılaştırıldı.

## Kaynaklar (erişim: 13.10.2026)

- K1 https://colab.research.google.com/signup — K2 https://research.google.com/colaboratory/faq.html
- K3 AWS EC2 on-demand fiyat listesi JSON (us-east-1, Linux), sayfa: https://aws.amazon.com/ec2/pricing/on-demand/
- K6 https://aws.amazon.com/ec2/spot/instance-advisor/ — K7 https://aws.amazon.com/sagemaker/pricing/
- K8 https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html — K9 https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html
- K10 https://cloud.google.com/products/compute/pricing/accelerator-optimized — K11 https://cloud.google.com/spot-vms/pricing — K12 https://docs.cloud.google.com/compute/resource-usage
- K13 https://lambda.ai/instances — K14 https://www.runpod.io/pricing — K15 https://vast.ai/pricing
- K16 https://cloud.google.com/edu/researchers — K17 https://edu.google.com/programs/credits/research/ — K18 https://azure.microsoft.com/en-us/free/students/
- K19 https://education.github.com/pack — K20 https://aws.amazon.com/education/awseducate/ — K21 https://aws.amazon.com/startups/credits
- K22 https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/ — K23 https://www.nvidia.com/en-us/startups/
- K24 https://aws.amazon.com/s3/pricing/ — K25 AWS veri transferi fiyat listesi — K26 https://cloud.google.com/storage/pricing
- K27 https://aws.amazon.com/aws-cost-management/aws-budgets/pricing/ — K28 https://docs.cloud.google.com/billing/docs/how-to/budgets
