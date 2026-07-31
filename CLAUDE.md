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

`AGENTS.md` artık bölünmüştür: eşik/chunking/hybrid search/kelime kanıtı
kapısı kararlarının tam ölçüm geçmişi ve gerekçeleri `kalibrasyon-kaydi`
skill'indedir. Eşiklere, retrieval'a, chunking'e, eval setine veya kelime
kanıtı kapısına dokunmadan önce o skill'i çağır; `AGENTS.md` yalnızca özet
tutar.

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

- **`CHUNK_SIZE = 128` karakter değil token.** Embedding tokenizer'ının
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
  Metin karşılaştıran her yerde `app/term_evidence.normalize_text()` kullan;
  bu fonksiyon Türkçe eşlemeyi yapıp artakalan birleşen noktayı temizler.
  `app/eval_metrics.py` aynı fonksiyonu oradan alır; ikinci bir kopya açma.

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

- **Kelime kanıtı kapısı LLM'den önce çalışır ve iki işi birden yapar.**
  `RAGService.has_term_evidence()` sorunun kelimeleri seçilen context'te
  geçmiyorsa `no_evidence` döndürür ve model hiç yüklenmez. Bu kapı olmasaydı
  `generate_with_fallback` içindeki `false_no_evidence` koruması modelin doğru
  reddini ezerdi. Kapıyı kaldırır veya gevşetirsen o hata geri gelir; eval'deki
  hard negative vakalar bunu yakalar.

- **Türkçe kelime eşleştirmesi ortak kök temellidir, kesme temelli değildir.**
  Kelimeyi sabit N karaktere kesip metinde aramak `sayısı` -> `sayısal` gibi
  yanlış eşleşmeler üretir ve sinyali bozar (ölçüldü: boşluk 0.38'den 0.05'e
  düştü). `terms_match()` üç kural uygular: tam eşleşme, `min_prefix`
  uzunluğunda **ortak önek** (biri diğerinin öneki olmak zorunda değil, çünkü
  `korunulur` ve `korunmak` aynı köktendir) ve kısa kökler için kökün tamamen
  kapsanması (`avı` -> `avından`; bu kural olmadan üç karakterlik kökler
  korpusta hiç eşleşmiyordu). Türkçe ünsüz yumuşaması (`süreç` -> `süreci`)
  `common_prefix_length()` içinde ele alınır.

- **Kelime kanıtı kapsaması IDF ağırlıklıdır ve ağırlıklar retrieval'dan gelir.**
  `get_top_chunks()` her sonuca `question_term_weights` ekler; `RAGService`
  bunu `has_term_evidence()`'a taşır. Bağlantı kopsa kapı sessizce eşit sayma
  davranışına döner ve `hard_negative_firewall_rules` sızıntısı geri gelir.
  Kapı yalnızca seçilen context'i görür, ayırt ediciliği ise ancak korpusun
  tamamı söyleyebilir; bu yüzden ağırlığı üretebilecek tek katman retrieval'dır.

- **Birleşik skor sıralamada kullanılır, kapıda kullanılmaz.** Hybrid search
  sonuçları RRF skoruna göre sıralanır, ama `results[0]["score"]` artık en
  yüksek cosine değil. Eşik karşılaştırmasında ve kullanıcıya gösterimde
  `retrieval.gate_score()` kullan; listenin başından skor okumak
  `SIMILARITY_THRESHOLD`, `CONTEXT_RELATIVE_SCORE_MARGIN` ve eval'deki hard
  negative `max_score` kontrolünü sessizce kaydırır.

- **`terms_match()` değişikliği BM25'i de değiştirir.** `app/sparse_search.py`
  aynı eşleştiriciyi kullanır. Eşleştirmeyi gevşetmek sparse skorlara gürültü
  ekler ve sıralama metriklerini düşürebilir (ölçüldü: kısa kök kuralı
  `Recall@1`'i 0.85'ten 0.80'e indirdi). Eşleştiriciye dokunduysan hem
  `tools/term_evidence_analysis.py` hem `tools/hybrid_search_analysis.py`
  çalıştır; biri kapıyı, diğeri sıralamayı ölçer.

- **`QUESTION_STOPWORDS` Türkçe karakterlerle yazılmalıdır.** Liste
  `normalize_text()` çıktısıyla karşılaştırılır; ASCII yazılmış bir kelime
  (`nasil`) hiç eşleşmez ve listeyi sessizce etkisiz bırakır.
  `tests/test_term_evidence.py` bunu kontrol eder.

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
