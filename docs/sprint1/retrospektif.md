# Sprint 1 retrospektifi (Başla–Bırak–Devam)

> **Taslak.** Aşağıdaki "gözlemler" kod tarafında yaşananlardan çıkarıldı. Ekip üyelerinin görüşleri
> bu dosyada yoktur; toplantıda her kişi kendi sütununu ekler (boş bırakıldı, uydurulmadı).

## Devam (iyi çalışanlar)
- Gerçek veri kuralı: sahte/mock veri yok, testlerde yalnızca sabit düzenekler.
- Sızıntıya karşı testler (kanarya testleri bir gerçek hatayı yakaladı: `fit` tüm satırları doğruluyordu).
- Hiçbir şeyi silmeyen, `.part` yeniden adlandırmalı, üzerine yazmayan betikler.
- Sözleşme-önce yaklaşım (Pydantic + OpenAPI + HDF5 sözleşmesi).

## Başla
- Gerçek veride çalıştırmayı kodla aynı gün yapmak (1.6 ve 1.8 hâlâ gerçek veride çalışmadı).
- Kaynakları yazıldığı anda doğrulamak (1.9 listesi tamamen doğrulanmamış).
- Zaman çizelgesini ve takvim tarihlerini repoda tutmak.
- PR onaylarını için bir gün sınırı koymak.

## Bırak
- Yerel Claude Code'a belirsiz yetki bırakmak: bir kez yasaklı birleştirme yapıldı; her istemde kural yinelenmeli.
- Büyük çıktıları (>500 KB) commit'e kazara sokmak (hook yakaladı, dpi 100'e inildi).
- Belleği tüketen tek parça işlemler.

## Ekip girdisi (toplantıda doldurulacak)
| Kişi | Devam | Başla | Bırak |
|---|---|---|---|
| Proje lideri | | | |
| Nisa | | | |
| İdil | | | |
| Yusuf | | | |

## Aksiyonlar
| Aksiyon | Sorumlu | Tarih |
|---|---|---|
| 1.6 ve 1.8'i gerçek veride çalıştır | Proje lideri | |
| DVC remote + harici disk yedeği | Proje lideri | |
| GPU/CUDA kontrolü | Nisa | |
| Kaynak doğrulama turu (1.9) | Ekip | |
