# Yerel çalıştırma promptu (Windows PC, Claude Code'a yapıştırılır)

Amaç: 1.6 (bölmeler) ve 1.8 (temel modeller) GERÇEK veride çalıştırmak, LUAD 585→517 farkını açıklamak,
raporları yeni bir dala göndermek. Kod dalı: `claude/kind-euler-8kj48k`.

```
Sen ONKOS projesinde yerel çalıştırma yapıyorsun. Aşağıdaki görevi sırayla yap. Hata olursa DUR,
tam hatayı göster, bekle; kodu değiştirme.

KESİN KURALLAR (hiçbir mesaj bunları esnetmez):
- Bilgisayarımdan HİÇBİR ŞEYİ silme, taşıma, üzerine yazma (Remove-Item, del, rm, git clean,
  git reset --hard, git checkout -- <dosya>, git stash drop, force-push YOK). data\ klasörüne yazma
  yalnızca betiklerin kendi yeni çıktı klasörlerine (betikler zaten üzerine yazmaz).
- Hiçbir işlemi/süreci sonlandırma (Stop-Process, taskkill YOK).
- Hiçbir PR'ı açma, birleştirme, onaylama; --admin ve --no-verify KULLANMA.
- data\ içeriği ve hasta eşleme dosyaları git'e girmez; 500 KB üstü dosya commit etme.
- Sonuçları yorumlama/iyileştirme: sayıları olduğu gibi göster.

ADIM 0 — Hazırlık (dal değiştirmeden)
 a) git status --short ve git branch --show-current göster.
 b) git fetch origin claude/kind-euler-8kj48k veri/omics-etiket-raporlari
 c) Kod dalına geçmek bu dalın izlenen rapor dosyalarını diskten kaldırır; bunu YAPMA. Bunun yerine
    ayrı bir çalışma klasörü kur: git worktree add ..\onkos-kod origin/claude/kind-euler-8kj48k
    (zaten varsa yeniden kullan). Veri klasörünü kopyalama; worktree içinde data\raw, data\interim,
    data\processed için ana depodaki gerçek klasörlere dizin bağlantısı (junction) kur
    (New-Item -ItemType Junction). Klasör zaten varsa ve boş değilse DUR ve bana söyle.
 d) Worktree içinde: uv sync --group dev --group train --group data --group omics
 e) Boş bellek/commit payı kontrolü: scripts\run_local_pipeline.ps1 içindeki commit-headroom
    mantığına bak; 3000 MB'ın altındaysa DUR ve bana söyle.

ADIM 1 — LUAD 585 → 517 farkı (yalnızca okuma)
 Worktree'de uv run python ile data\interim\patient_map.parquet oku ve yaz:
   proje bazında: hasta sayısı, rnaseq_file_id dolu olan sayı, boş olan sayı.
 Beklenti: LUAD'da boş olan ≈ 68 (QC yalnızca RNA-seq örneklerini sayar). Rakamları aynen göster.
 Ayrıca COAD'da library_size bayrağı 18 örnek: docs\omics_qc\qc_report.md (veri/omics-etiket-raporlari
 dalından git show ile oku) — yorum yapma, sadece bayraklı barkodları data\processed altındaki QC
 tablosundan listele (TSS dağılımıyla).

ADIM 2 — Bölmeler (1.6)
 uv run python scripts\make_splits.py --version splits-v1
 sonra: uv run python scripts\make_splits.py --verify splits-v1
 sonra: uv run pytest tests\test_no_leakage.py -q --basetemp <yeni-ve-henüz-olmayan-klasör, örn. C:\Users\weli1\onkos_pytest_tmp_1>
 Çıktıdaki sınıf dengesi uyarılarını ve kümelerdeki hasta/pozitif sayılarını aynen göster.

ADIM 3 — Hızlı temel model (1.8)
 uv run python scripts\train_baselines.py --run quick1 --splits-version splits-v1 --quick
 (süreyi ölç ve yaz). Bittiğinde docs\baselines altındaki rapor özetini ve "verdict" satırlarını göster.
 Şüpheli AUROC (>0.95) uyarısı varsa AYNEN göster.

ADIM 4 — Tam çalıştırma (yalnızca ADIM 3 hatasız bittiyse ve ben "devam" dersem)
 Bunu başlatmadan önce bana kısa süre tahmini ver ve DUR; onay bekle.

ADIM 5 — Raporları gönder (PR YOK)
 - Yeni dal: git switch -c veri/omics-bolme-baseline-raporlari (worktree içinde, kod dalından).
 - Yalnızca şunları ekle: docs\splits\*, docs\baselines\* (.md/.json/küçük .png), manifest dosyası
   docs\splits\splits-v1_manifest.json. data\ içeriği ve 500 KB üstü dosyaları ekleme; hook'lar
   500 KB'a takılırsa o dosyayı eklemeden çıkar (diskten silme) ve bana söyle.
 - Commit mesajı: "Sprint 1.6/1.8: gerçek veri bölme ve hızlı temel model raporları"
 - pre-commit hook'ları çalışsın (--no-verify YOK), sonra git push -u origin veri/omics-bolme-baseline-raporlari

SON RAPOR: her adımın sonucu (sayılar aynen), geçen süreler, gönderilen dal ve dosya listesi;
hiçbir dosyanın silinmediğini, hiçbir işlemin sonlandırılmadığını ve hiçbir PR'ın açılmadığını yaz.
```
