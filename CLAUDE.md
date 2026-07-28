# CLAUDE.md

Bu dosya Claude Code'un bu repository'de nasıl çalışacağını tanımlar.

## Önce bunu oku

Proje bağlamının tek gerçek kaynağı [`AGENTS.md`](AGENTS.md) dosyasıdır: amaç,
teknoloji kararları, dosya haritası, korunması gereken davranışlar, güncel
eşikler ve roadmap oradadır. **Her göreve başlarken AGENTS.md'yi oku.** Bu dosya
onu tekrarlamaz; yalnızca ona ek olarak Claude'a özel çalışma kurallarını ve
kodda kolayca gözden kaçan noktaları tutar.

Ayrıntılı öğretici anlatım için [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md),
dış sunum için [`README.md`](README.md). `INSTRUCTIONS.md` ilk hedefleri tutar ve
güncel gerçekliği yansıtmaz; ona göre karar verme.

## Bu projenin asıl amacı öğrenmek

Kullanıcı bu projeyle RAG ve AI uygulama geliştirmeyi öğreniyor. Bu, çıktının
kalitesini düşürmek için bir mazeret değil, ama şunları değiştirir:

- Bir değişikliği anlatırken **ne** yaptığını değil **neden** yaptığını öne çıkar.
  Alternatifi neden seçmediğini bir cümleyle söyle.
- Sihirli sayı bırakma. Bir eşik, boyut veya sabit ekliyorsan nereden geldiğini
  yaz ve ölçülebilir hale getir.
- "Çalışıyor" yeterli değil; **ölçülebilir** olması gerekiyor. Retrieval veya
  cevap kalitesine dokunan her değişiklik bir eval vakasıyla veya öncesi/sonrası
  karşılaştırmasıyla gelmelidir.
- Kütüphane eklemek yerine mekanizmayı açıkça yazmak genelde daha öğreticidir.
  Bağımlılık eklemeden önce gerekliliğini gerekçelendir.

## Cevap dili ve üslup

- Kullanıcıya Türkçe cevap ver. Kod içi isimler İngilizce kalır (mevcut stil).
- Kullanıcıya gösterilen bütün terminal metinleri, hata mesajları, panel
  başlıkları ve çözüm önerileri Türkçe olmalıdır.
- Kodda yorum yoğunluğu düşüktür; bu stili koru. Kendini açıklayan kod tercih et.

## Her Python değişikliğinden sonra çalıştır

```bash
source .venv/bin/activate
python -m py_compile main.py eval.py app/*.py tests/*.py
python -m unittest discover -s tests
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python eval.py
```

Testler saniyeler sürer (embedding ve LLM mock'lanır); atlamak için bahane yok.
`eval.py` gerçek embedding modelini yükler, bu yüzden ilk çalıştırma yavaştır.
`HF_HUB_OFFLINE=1` gereksiz ağ retry loglarını önler.

Retrieval'a dokunan her değişiklikte metrik farkını da göster:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python eval.py --compare
```

Bu, Recall@k ve MRR'ı `eval_baseline.json` ile karşılaştırır. Baseline'ı yalnızca
değişikliğin **kasıtlı bir iyileşme** olduğu doğrulandıktan sonra
`--update-baseline` ile güncelle; gerilemeyi baseline'ı ezerek gizleme.

Ingestion, chunking veya embedding'e dokunduysan ek olarak `/reindex` çalıştır ve
chunk sayısı ile eval skorlarındaki değişimi kullanıcıya **sayıyla** bildir.

Prompt, LLM veya fallback davranışına dokunduysan bunlar deterministik test
edilemez. Kullanıcıya çalıştıracağı **tam** komutu ve soruları ver; sonucu
kendin ürettiğini varsayma.

## Kodu okurken kolayca gözden kaçanlar

- **Yollar runtime'da yeniden bağlanıyor.** `app/ingest.DOCS_DIR`,
  `app/database.DB_PATH` ve `app/health.DOCS_DIR` modül seviyesinde relative
  varsayılanlarla tanımlıdır; `main.py` içindeki `apply_project_paths()` bunları
  `app/project.py` sonuçlarıyla **ezer**. Yeni bir modüle yol sabiti eklersen bu
  fonksiyona da bağlamayı unutma, yoksa `--project` ve `LOCAL_RAG_HOME` o modül
  için sessizce çalışmaz. Yeni yol sabitini `tests/test_project.py` ile kapsa.

- **`CHUNK_SIZE = 110` karakter değil token.** Embedding tokenizer'ının
  offset mapping'i ile ölçülür ve özel tokenları da kapsar. Modelin sınırı 128
  tokendır. Bu değeri karakter sanıp büyütmek, gömülmeyen metin kuyruğu üretir.

- **Chunking testleri tokenizer'a bağımlıdır.** `split_long_text()` tokenizer'ı
  parametre olarak alır; testler kendi fake tokenizer'ını verir. Yeni chunking
  testinde gerçek modeli yükleme.

- **`replace_chunks()` atomiktir ve öyle kalmalı.** Chunklar ve
  `source_manifest` aynı transaction içinde değişir. Reindex akışında parça
  parça commit etme; hazırlama başarısız olursa eski indekse dokunma.

- **`KeyboardInterrupt` yutulmamalı.** `rag_service` içinde generation
  hatalarında fallback var, ama `KeyboardInterrupt` üst katmana çıkmalıdır.
  Aksi halde `Ctrl+C` yarım cevabı "kaynak fallback'i" sanır ve geçmişe yazar.

- **Veritabanı erişimi yalnızca `app/database.py` içinde.** SQLite sorgularında
  her zaman parametreli ifade kullan; kaynak filtresi dosya adını sorgu metnine
  eklemez.

- **Cevap modu isimleri iç kullanımdır.** `generative`, `extractive`,
  `fallback_extractive`, `no_evidence` kullanıcıya gösterilmez; Türkçe etiketler
  `app/cli_output.py` içindedir. Kaynak/skor/chunk bilgisi model cevabının içine
  değil ayrı kaynak tablosuna gider.

- **Türkçe metin karşılaştırmasında ham `casefold()` kullanma.**
  `"İ".casefold()` sonucu `"i"` değil `"i" + U+0307` olur ve `"I".casefold()`
  `"ı"` yerine `"i"` verir. Bu yüzden büyük harfli bir terim sessizce eşleşmez.
  Metin karşılaştıran her yerde `app/eval_metrics.normalize_text()` kullan;
  bu fonksiyon Türkçe eşlemeyi yapıp artakalan birleşen noktayı temizler.

- **Eval'de ground truth chunk ID ile etiketlenmez.** Chunk ID'leri her
  reindex'te değişir (`AUTOINCREMENT`) ve chunking ayarı değişince sınırlar da
  kayar. `eval_cases.json` içindeki `relevant_chunk_terms` bir **içerik
  imzasıdır**: bir imza terim listesidir, chunk o terimlerin hepsini içeriyorsa
  imzayı karşılar. Yeni vaka eklerken imzanın o chunk'a özgü olduğundan emin ol;
  `case_labels` kontrolü indekste karşılığı olmayan imzayı hata olarak gösterir.

- **`known_gap` bilinen boşluk demektir, görmezden gelme demek değildir.**
  Mevcut eşiklerle geçmesi beklenmeyen hard negative vakalar `GAP` olarak
  raporlanır ve baseline'a yazılır ama pass/fail kapısını düşürmez. Amaç eval'i
  kalıcı kırmızıda tutmamaktır. Bir `known_gap` vakası geçmeye başlarsa çıktı
  `FIXED` der; o zaman bayrağı kaldır.

- **`app/config.NO_EVIDENCE_ANSWER` tek kaynaktır.** "Bu bilgi verilen
  dokümanlarda yok." metnini prompt'a, servise veya teste elle yazma; sabiti
  içe aktar. Bu cümlenin birebir eşleşmesi yanlış-ret fallback'inin çalışması
  için gereklidir.

## Yapma

- Kullanıcı açıkça istemedikçe embedding modelini, varsayılan LLM'i, eşikleri
  veya depolama teknolojisini değiştirme.
- Kullanıcı açıkça istemedikçe commit, push, history rewrite veya destructive
  Git işlemi yapma. Çalışma ağacında commit edilmemiş kullanıcı değişikliği
  olabilir; başlamadan `git status` ve ilgili diff'i oku, mevcut değişiklikleri
  ezme.
- `data/`, `.venv/`, model cache veya büyük binary dosyaları Git'e ekleme.
- Küçük bir özellik için yeni framework ekleme; mevcut sade modüler yapıyı koru.
- Web arayüzü, API sunucusu veya conversation memory'yi kendi inisiyatifinle
  eklemeye başlama; bunlar şu an proje hedefinde değil.

## Mimari karar uyarısı

Kullanıcının global tercihi: birden fazla makul seçenek arasında seçim, veri
modeli, ölçekleme stratejisi veya önemli refactor yönü gibi noktalarda kısa bir
"burası mimari karar, daha yüksek model/effort düşünebilirsin" hatırlatması yap.
Bu projede özellikle şunlar bu kapsamdadır: hybrid search skor birleştirme
(RRF vs ağırlıklı normalizasyon), reranker seçimi ve latency bütçesi, eşiklerin
yeniden kalibrasyonu, conversation history için query rewriting tasarımı ve
vector database'e geçiş kararı. Rutin kod yazımında bu uyarıyı yapma.

## Bir görevi tamamlanmış saymak için

1. İlgili kodu ve commit edilmemiş kullanıcı değişikliklerini okumuş ol.
2. Yeni davranışı deterministik testle kapsa.
3. py_compile, unittest ve eval'i çalıştır; sonucu sayıyla bildir.
4. Gerekliyse kullanıcıya tam manuel model test komutunu ver.
5. Türkçe UX, kaynak doğruluğu ve fallback davranışının korunduğunu doğrula.
6. Ne değişti, neden değişti, test sonucu ne — kısa ve öğretici anlat.
7. Çalıştıramadığın bir doğrulama varsa açıkça söyle.
8. Davranış, dosya sorumluluğu veya roadmap değiştiyse AGENTS.md ve
   PROJECT_GUIDE.md'yi güncelle. Dokümanların koddan sapmasına izin verme.
