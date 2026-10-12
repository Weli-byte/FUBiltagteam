# 1.9 — Koşullu Difüzyon: literatür taraması

> **Durum: TASLAK, KAYNAKLAR DOĞRULANMALI.**
> Bu metin, geliştirme ortamında internete erişim olmadan, yazarın bilgisinden derlendi.
> Aşağıdaki her künye (yazar, yıl, yayın yeri, bulgu) **doğrulanmalı**: DOI/arXiv sayfasından
> açılıp okunmadan raporlarda, sunumlarda ve TEKNOFEST/TÜBİTAK metninde **alıntılanmamalıdır**.
> Doğrulandıkça "Doğrulama" sütunu `[x] ad, tarih` ile doldurulur.

## Terim ayrımı (projede zorunlu)
- **Tahmin (prediction):** gözlenen veriden etiket/olasılık üretmek (1.8 temel modelleri).
- **Simülasyon (simulation):** koşullu üretici modelle *varsayımsal* bir çıktı üretmek. Gerçek bir
  tedavi sonucu, hasta yanıtı veya biyolojik gerçek **değildir**.
- **Kanıt (evidence):** yalnızca bağımsız doğrulama (ör. klinik veri, deneysel çalışma) ile oluşur.
  Üretici model çıktısı tek başına kanıt sayılmaz.

Projede difüzyon çıktıları için "tedavi etkisi", "hasta yanıtı", "klinik" gibi ifadeler **kullanılmaz**;
doğru ifade: "koşullu örnek üretimi / veri artırma / hipotez üretimi (araştırma prototipi)".

## Kaynaklar (doğrulanmalı)

| # | Künye (hafızadan, DOĞRULANMALI) | Konu | Projeye katkısı | Doğrulama |
|---|---|---|---|---|
| 1 | Ho, Jain, Abbeel — *Denoising Diffusion Probabilistic Models* (NeurIPS 2020) | DDPM temeli | Temel eğitim hedefi (gürültü tahmini) | [ ] |
| 2 | Song, Meng, Ermon — *Denoising Diffusion Implicit Models* (ICLR 2021) | DDIM, hızlı örnekleme | Çıkarım süresi düşürme | [ ] |
| 3 | Song vd. — *Score-Based Generative Modeling through SDEs* (ICLR 2021) | Skor/SDE çerçevesi | Teorik çerçeve | [ ] |
| 4 | Nichol, Dhariwal — *Improved DDPM* (ICML 2021) | Öğrenilmiş varyans, kosinüs şeması | Eğitim kararlılığı | [ ] |
| 5 | Dhariwal, Nichol — *Diffusion Models Beat GANs on Image Synthesis* (NeurIPS 2021) | Sınıflandırıcı kılavuzluğu | Koşullama yöntemlerinden biri | [ ] |
| 6 | Ho, Salimans — *Classifier-Free Diffusion Guidance* (2022) | Sınıflandırıcısız kılavuzluk | Koşul (etiket) kullanımı için birincil aday | [ ] |
| 7 | Rombach vd. — *High-Resolution Image Synthesis with Latent Diffusion Models* (CVPR 2022) | Latent difüzyon | Düşük hesaplama maliyeti; **bizim durumda patch-özellik uzayı** | [ ] |
| 8 | Karras vd. — *Elucidating the Design Space of Diffusion-Based Generative Models* (NeurIPS 2022) | Tasarım uzayı | Hiperparametre seçimi | [ ] |
| 9 | Zhang, Rao, Agrawala — *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet, ICCV 2023) | Ek koşul girişi | Çoklu koşul (alt tip + mutasyon) | [ ] |
| 10 | Kazerouni vd. — *Diffusion Models in Medical Imaging: A Comprehensive Survey* (Medical Image Analysis, 2023) | Tıbbi görüntüde difüzyon taraması | Alanın genel durumu | [ ] |
| 11 | Müller-Franzes vd. — *Diffusion Probabilistic Models beat GANs on Medical Images* (2022, arXiv) | Tıbbi görüntü kıyası | GAN ile kıyas gerekçesi | [ ] |
| 12 | Pinaya vd. — *Brain Imaging Generation with Latent Diffusion Models* (2022) | Latent difüzyon, tıbbi | Latent yaklaşımın tıbbi örneği | [ ] |
| 13 | Packhäuser vd. — *Generation of Anonymous Chest Radiographs Using Latent Diffusion Models* (Sci Rep, 2023) | Gizlilik odaklı üretim | Mahremiyet / paylaşım tartışması | [ ] |
| 14 | Carlini vd. — *Extracting Training Data from Diffusion Models* (USENIX Security 2023) | Ezberleme | Hasta verisi sızıntısı riski testi | [ ] |
| 15 | Somepalli vd. — *Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models* (CVPR 2023) | Veri kopyalama | Ezberleme ölçümü | [ ] |
| 16 | Heusel vd. — *GANs Trained by a Two Time-Scale Update Rule* (NeurIPS 2017) | FID | Kalite metriği ve sınırlılığı | [ ] |
| 17 | Naeem vd. — *Reliable Fidelity and Diversity Metrics for Generative Models* (ICML 2020) | Precision/recall, density/coverage | FID'e tamamlayıcı metrik | [ ] |
| 18 | Chen vd. — *Towards a general-purpose foundation model for computational pathology* (UNI, Nature Medicine, 2024) | Patoloji temel modeli özellikleri | Patch özellik çıkarıcı adayı (histo modülü ile uyum) | [ ] |

> Kaynak 18 histoloji özellik uzayı içindir; projenin kullanacağı gerçek özellik çıkarıcı
> Sprint 2'de (histo modülü) seçilir. Burada yalnızca olası bağlantı olarak listelenmiştir.

## Sentez (kaynaklara dayanan, doğrulanmamış çıkarımlar)
1. Tıbbi görüntüde difüzyon modelleri GAN'lara güçlü bir alternatif olarak raporlanıyor (#10, #11);
   ancak bu *görüntü kalitesi* iddiasıdır, biyolojik/klinik geçerlilik iddiası değildir.
2. Hesaplama maliyeti için latent uzayda çalışmak standart çözüm (#7, #12). Bizde veri zaten
   düşük boyutlu (omics vektörü, patch özellikleri), bu yüzden **piksel difüzyonu gerekmez**.
3. Koşullama için sınıflandırıcısız kılavuzluk (#6) en basit seçenek; etiket bırakma (condition dropout) ile eğitilir.
4. Küçük, dengesiz hasta verisinde ezberleme riski gerçek (#14, #15) ve ölçülmeden paylaşım yapılmamalı.
5. Kalite metrikleri tek başına yetmez (#16, #17); **alt görev faydası** (downstream) ve ezberleme testi şart.

## Sınırlılıklar
- Tarama sistematik değildir (PRISMA yok); 18 kaynak başlangıç setidir.
- Künyeler hafızadan yazıldı; yanlış yıl/yayın yeri olabilir.
- TCGA gibi küçük kohortlarda difüzyon başarısına dair doğrulanmış bir kaynak bu listede yoktur;
  bu bir boşluktur ve ilk doğrulama turunda aranmalıdır.
