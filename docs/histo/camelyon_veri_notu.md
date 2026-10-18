# CAMELYON16/17 ve PAIP: Veri Notu, Maskeler, Resmi Bölme (Mini sprint 1.5)

> **Durum (13.10.2026):** Kova listesi **canlı olarak** okundu ve manifest yazıldı
> (`data/manifests/camelyon16_objects.csv`: 399 görüntü 752,5 GB, 160 XML, 399 maske, 3
> değerlendirme dosyası). Görüntüler indirilmedi (0,75 TB). Maske hattı ve bölme üretimi yalnızca
> birim testleriyle sınandı; gerçek slaytlarda maske üretimi ve 20 slaytlık görsel doğrulama
> indirme sonrasına kaldı.

**Sade anlatım:** CAMELYON, patoloji yapay zekasının "sınav kâğıdı": meme kanserinin lenf
düğümüne yayılmasını (metastaz) arayan, patologların tümör bölgelerini elle çizdiği slaytlar.
Çizimleri bilgisayarın anlayacağı "tümör burada" haritasına (maske) çeviririz; resmi
eğitim/test ayrımına sadık kalırız ki sonuçlarımız dünyadakilerle karşılaştırılabilsin.

## 1. CAMELYON16 olguları (kaynaklı)

| Olgu | Değer | Kaynak (erişim 13.10.2026) |
|---|---|---|
| Toplam | 400 WSI planlandı; **yayımlanan 399** (normal_086 → tumor_111 yeniden adlandırıldı, test_049 kopya olduğu için çıkarıldı) | resmi README (grand-challenge kullanıcı içeriği) ve S3 listesi |
| Eğitim | 270: 159 normal + 111 tumor (S3 sayımı) | S3 `CAMELYON16/images/` |
| Test | 129 (`test_001..130`, 049 yok) | S3 sayımı |
| Merkezler | Radboud UMC (3DHistech Pannoramic 250, ~0,243 µm/px) ve UMC Utrecht (Hamamatsu NanoZoomer-XR, ~0,226 µm/px) | Litjens ve ark. 2018 (PMC6007545); `reference.csv` `center` sütunu |
| Test etiketleri | `CAMELYON16/evaluation/reference.csv`: `image,type,class,center` (399 satır; tumor/normal, negative/micro/macro, merkez) | S3 |
| Anotasyon | ASAP XML; `PartOfGroup` değerleri S3 kopyasında **"Tumor"** (3.263) ve **"Exclusion"** (284); README `_0/_1/_2` anlatır (eski yarışma dosyaları). Koordinatlar seviye-0 piksel | 160 XML tarandı; camelyon17.grand-challenge.org/Data |
| Tam anotasyonlu olmayan | `test_114` ve 20 tumor slaydı (listesi `configs/histo/camelyon.yaml`) | resmi README |
| Format | BigTIFF (ASAP dönüştürücü), tile 512, JPEG; dosya 0,33-3,97 GB | README, S3 |
| Lisans | **CC0 1.0** (`CAMELYON16/license.txt`; AWS kaydı "License: CC0") | https://registry.opendata.aws/camelyon/ |
| İndirme | `s3://camelyon-dataset/CAMELYON16/` (us-west-2, anonim: `aws s3 ls --no-sign-request`); HTTPS: https://camelyon-dataset.s3.us-west-2.amazonaws.com/ ; GigaDB doi:10.5524/100439 | registry + gigadb.org/dataset/100439 |
| Makale | Ehteshami Bejnordi B ve ark., JAMA 2017;318(22):2199-2210, doi:10.1001/jama.2017.14585 | Crossref |

CAMELYON17: 1.000 WSI (500 eğitim / 500 test), 5 merkez (RUMC, CWZ, UMCU, Rijnstate, LPON), hasta
düzeyi pN evresi (`stages.csv`), 50 eğitim slaydında lezyon anotasyonu, 2,97 TB, CC0; makale Bandi
ve ark., IEEE TMI 2019;38(2):550-560, doi:10.1109/TMI.2018.2867350. Çok merkezli genelleme testi
(Sprint 3) için adaydır; şimdilik yalnızca `stages.csv` ve anotasyonlar indirilecek.

PatchCamelyon (PCam): CAMELYON16'dan türetilmiş 327.680 adet 96x96 patch (train 262.144 / valid
32.768 / test 32.768; etiket: merkez 32x32 bölgede tümör pikseli var mı), Zenodo 2546921 (8 GB),
`torchvision.datasets.PCAM`; lisans README'de CC0, Zenodo metadatasında MIT (**tutarsız; veri
kartına not düşüldü**). Baseline (1.9) için hızlı alternatif: slayt bazlı bölme bilgisi PCam'de
WSI düzeyinde verilmez (yalnızca CAMELYON16 train/test ayrımı korunur); bu yüzden ana baseline
kendi patch setimizle yapılır.

## 2. Komutlar

```bash
uv run --group histo python scripts/download_camelyon.py --list-only      # manifest (ağ, 10 sn)
uv run --group histo python scripts/download_camelyon.py                  # anotasyon + reference.csv (~80 MB)
uv run --group histo python scripts/download_camelyon.py download.limit=10 \
    download.include=[annotations,evaluation,images]                      # 10 görüntü dene
uv run --group histo python scripts/build_tumor_masks.py                  # maskeler + splits.json + 20 overlay
```

- İndirme HTTP Range ile kaldığı yerden devam eder; boyut her zaman, md5 yalnızca tek parçalı
  nesnelerde (çok parçalı ETag md5 değildir) doğrulanır. Resmi `checksums.md5` dosyası indirme
  sonrası ek kontrol için kullanılacak.
- Maskeler `masks.downsample=32` (seviye-0'ın 1/32'si; ~8 µm/px) PNG olarak yazılır; patch
  etiketi `patch_tumor_fractions` ile seviye-0 kutusundan hesaplanır (≥ %50 tümör → tümör).
  Resmi `masks/` klasöründeki maskeler (9,4 GB) alternatif olarak kullanılabilir; bizim hattımız
  XML'den üretir ki grup/istisna mantığı denetlenebilsin.
- `camelyon16_splits.json`: slayt adından resmi bölme; `reference.csv`'den etiket, lezyon sınıfı
  ve merkez; eğitim/test kesişimi sıfır (kod hata verir); kısmi anotasyonlu slaytlar işaretli.
- Kural: **test slaytları eğitimde, model seçiminde veya eşik ayarında kullanılmaz**
  (`rows_for_split(..., "test")` kilitli).

## 3. Doğrulama (bitti kriteri)

- [ ] Rastgele 20 slaytta maske bindirme görseli gözle kontrol edildi (`masks.overlay_sample=20`,
      `data/processed/camelyon16/mask_overlays/`); hizalama hatası yok. Sonuç: __/20.
- [x] Eğitim ve test listeleri kesişmiyor (kodda zorunlu; `test_build_splits_counts_overlap_and_flags`).
- [ ] Slayt ve maske sayıları resmi kaynakla tutarlı: 399 görüntü, 160 XML (111 tumor + 49 test)
      — manifestte doğrulandı, indirme sonrası dosya sayımıyla yinelenecek.

## 4. PAIP erişim notu

| Konu | Durum (kaynak: paip2019/2020/2021.grand-challenge.org, wisepaip.org; 13.10.2026) |
|---|---|
| Veri | PAIP2019 karaciğer HCC (50/10/40 WSI, Aperio AT2 20x, SVS + XML + maske); PAIP2020 kolon (47/31/40, 40x; MSI etiketi); PAIP2021 perinöral invazyon (240 WSI, kolon/prostat/pankreatobiliyer) |
| Erişim | **İmzalı Veri Kullanım Anlaşması (DUA) zorunlu**: `https://wisepaip.org/assets/files/PAIP_DUA.docx` indirilip e-posta ile gönderilir; grand-challenge hesabı + "Join"; şu an yalnızca 2019 ve 2020 setleri veriliyor |
| Lisans | CC BY-NC 4.0 (ticari kullanım yasak), zorunlu teşekkür metni (SNUH, KHIDI, grant HI18C0316); 2019 sitesinde açık lisans beyanı **doğrulanamadı** |
| Toplu indirme | Halka açık değil (Download sayfaları girişsiz 403) |
| Karar önerisi | ONKOS için **zorunlu değil**: meme/akciğer/kolon hedeflerimizde CAMELYON16/17 + TCGA yeterli. PAIP2020 (kolon, MSI) ileride kolon doğrulaması için değerli; DUA başvurusunu Sprint 2'de İdil'in TÜBİTAK başvurusuyla birlikte proje lideri değerlendirsin. Atıf: Kim YJ ve ark., Med Image Anal 2021;67:101854; Kim K ve ark., Med Image Anal 2023;89:102886 |
