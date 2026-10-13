"""Module 1: histopathology data pipeline and ViT + ABMIL (owner: Nisa).

Bu paket ne işe yarar? Dev boyutlu doku görüntülerini (WSI) okur, doku alanlarını bulur, kalite
kontrol yapar, küçük karelere (patch) böler, renk farklarını giderir ve ilk görüntü modelini
(ViT / ResNet) eğitir. Diğer modüllerle yalnızca ``onkos.contracts`` üzerinden konuşur.

Alt paketler
------------
- ``wsi``              : slayt okuma (OpenSlide), doku tespiti, kalite kontrol (1.3, 1.6)
- ``patches``          : koordinat üretimi, HDF5 çıktısı, örnek paket, PyTorch veri seti (1.3, 1.7)
- ``camelyon``         : CAMELYON16/17 anotasyon (XML) -> tümör maskesi, resmi bölme (1.5)
- ``tcga``             : TCGA barkod ayrıştırma, slayt seçimi ve envanter (1.4)
- ``stain``            : boya normalizasyonu (Macenko, Reinhard) ve histolojik augmentation (1.8)
- ``patch_classifier`` : patch düzeyi tümör/normal sınıflandırma baseline'ı (1.9)
"""
