# TCGA Tanı Slaytı Envanteri

> `scripts/download_tcga_slides.py` tarafından üretilir. Yalnızca toplu sayılar içerir;
> hasta düzeyi tablolar (`slide_inventory.parquet`, `patient_slide_map.parquet`)
> `data/interim/` altında kalır ve git'e girmez.

- GDC veri sürümü: `Data Release 46.0 - August 10, 2026`
- Seçim kuralı: hasta başına bir birincil tümör (01) tanı slaydı (DX); birden fazlaysa en
  düşük DX numarası; QC sonucu varsa `exclude` elenir, `use` > `suspect`.
- `selected_gb`: yalnızca seçilen slaytların toplam boyutu (indirme planı için).

| Proje | slides | dx_slides | patients | selected | total_gb | selected_gb | patients_in_leader_list | selected_in_leader_list |
|---|---|---|---|---|---|---|---|---|
| TCGA-BRCA | 1133 | 1133 | 1062 | 1061 | 1158.80 | 1085.07 | 0.00 | 0.00 |
| TCGA-COAD | 459 | 459 | 451 | 451 | 283.67 | 274.88 | 0.00 | 0.00 |
| TCGA-LUAD | 541 | 541 | 478 | 478 | 415.09 | 352.89 | 0.00 | 0.00 |
| TOTAL | 2133 | 2133 | 1991 | 1990 | 1857.55 | 1712.84 | 0.00 | 0.00 |

- İndirme planı: 1990 slayt, 1712.8 GB (only_in_leader_list=True, limit=None).
- **Not:** proje liderinin hasta listesi (`patient_map.parquet`) bu makinede yoktu; kesişim sütunları 0 görünür. Liste geldiğinde betik yeniden çalıştırılır.
