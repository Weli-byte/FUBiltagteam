# Risk kaydı (Sprint 1 sonu)

Sorumlular README'deki modül sahiplerine göredir (Nisa: histo; İdil: GNN + API; Yusuf: XAI + arayüz;
Proje lideri: omics, RAG/LLM, diffusion, veri hattı). **Atamalar öneridir, ekip onaylamalı.**
Olasılık/etki: D=düşük, O=orta, Y=yüksek (öznel; ekip gözden geçirmeli).

| # | Risk | Olas. | Etki | Önlem | Sorumlu | İzleme |
|---|---|---|---|---|---|---|
| R1 | Klinik doğruluk algısı (çıktı "tanı/reçete" sanılır) | O | Y | Her yanıtta araştırma prototipi uyarısı (ADR-0005), "simülasyon/tahmin/kanıt" ayrımı | Proje lideri + Yusuf (arayüz metni) | Sprint 2 incelemesi |
| R2 | Nadir sınıflarda (EGFR, BRCA1) güvenilmez sonuç | Y | O | AUPRC + bootstrap GA, tek-proje modeli yok, kanser türü tabanı ile kıyas | Proje lideri | 1.8 gerçek çalıştırma |
| R3 | Veri sızıntısı | D | Y | Hasta düzeyinde bölme, kilitli test, `test_no_leakage`, eğitim-bölmesinde fit | Proje lideri | Her PR (CI) |
| R4 | TCGA dışında doğrulama yok | Y | O | Genelleme iddiası yok; harici kohort adayı araştır | Proje lideri | Sprint 3 |
| R5 | Histoloji verisi/GPU hazır değil | O | Y | Nisa'nın GPU/CUDA kontrolü; WSI listesi hazır (indirme Nisa'da) | Nisa | Sprint 2 başı |
| R6 | Modüller arası sözleşme kayması | O | O | Pydantic/OpenAPI tek kaynak, tüketici sözleşme testi, CODEOWNERS | Proje lideri; tüketiciler İdil, Yusuf, Nisa | Her sözleşme PR'ı |
| R7 | Veri kaybı (data/ git'te değil) | O | Y | DVC Google Drive remote **henüz kurulmadı**; harici disk yedeği bekliyor | Proje lideri | Sprint 2 ilk hafta |
| R8 | Kaynak/literatür hataları | O | O | 1.9 listesi "doğrulanmalı"; doğrulanmadan alıntı yok | Proje lideri (+ ekip dağıtımı) | Sprint 2 sonu |
| R9 | Difüzyon çıktısı ezberleme / yanlış yorumlanma | O | Y | Ezberleme testi, `synthetic` bayrağı, paylaşım yok | Proje lideri | Sprint 3 PoC |
| R10 | Bilgisayar belleği / uzun çalıştırmalar | O | O | Memmap/bloklu işleme, PowerShell çalıştırıcı | Proje lideri | Süreklidir |
| R11 | Onay darboğazı (PR #3 onayları, #5 birleştirme) | O | D | Gözden geçirme sözü, gün sınırı | Tüm ekip | Haftalık |
| R12 | Etiket = somatik mutasyon; fonksiyonel anlam yanlış okunur | O | O | `docs/etiket_tanimi.md`, raporda açık ifade | Proje lideri | Sunum öncesi |
| R13 | TEKNOFEST/TÜBİTAK takvimi (tarihler bu repoda yok) | ? | Y | Tarihleri belgeden girin | Proje lideri | Hemen |
