# AGENTS.md

Bu dosya, bu repository üzerinde çalışan AI agentlar için proje bağlamı ve çalışma kurallarıdır. Repository içindeki bütün dosyalar için geçerlidir.

## 1. Projenin Amacı

Local RAG Assistant, kullanıcının `docs/` klasörüne koyduğu yerel dokümanlardan bilgi bulan ve yalnızca bulunan bağlama dayanarak Türkçe cevap üreten bir Python uygulamasıdır.

Bu proje:

- Bir fine-tuning veya model eğitimi projesi değildir.
- Local-first çalışmalıdır.
- Dokümanları chunklara ayırır, embedding üretir ve SQLite içinde saklar.
- Soruları semantic search ile ilgili chunklara yönlendirir.
- Güçlü kısa bir kaynak varsa extractive, sentez gerekiyorsa local LLM ile generative cevap verir.
- Yeterli kanıt yoksa tam olarak `Bu bilgi verilen dokümanlarda yok.` cevabını verir.

Ana kullanıcı bu projeyle RAG ve AI uygulama geliştirmeyi öğrenmektedir. Değişiklikleri açıklarken yalnızca ne yapıldığını değil, neden yapıldığını da kısa ve öğretici biçimde anlat.

## 2. Güncel Teknoloji Kararları

- Python: 3.11 virtual environment (`.venv`)
- Embedding modeli: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Embedding boyutu: 384
- Benzerlik: scikit-learn L2 normalization ve NumPy normalized dot product (cosine similarity)
- Veri deposu: `data/rag.db` içinde SQLite
- Embedding saklama biçimi: JSON string
- Local LLM çalışma zamanı: Microsoft Foundry Local
- Varsayılan chat modeli: `phi-4-mini`
- Doküman türleri: UTF-8 TXT ve metin tabanlı PDF
- PDF okuyucu: `pypdf`
- Kullanıcı arayüzü: Rich ve `prompt-toolkit` tabanlı interaktif terminal CLI

Kullanıcının açık isteği olmadan embedding modelini, varsayılan LLM'i, eşikleri veya depolama teknolojisini değiştirme.

## 3. Temel Çalışma Akışı

1. `docs/` içindeki TXT ve PDF dosyalarını oku.
2. Metni cümle/kelime sınırlarını gözeten, overlap içeren chunklara ayır.
3. Her chunk için embedding üret.
4. Yeni indeksin tamamını bellekte hazırla.
5. SQLite indeksini tek transaction ile atomik olarak değiştir.
6. Kullanıcı sorusunun embeddingini üret.
7. Chunk embeddingleriyle cosine similarity hesapla.
8. En iyi sonuçları skorlarına göre sırala ve zayıf context'i filtrele.
9. Ön kapı: soru bu korpusun konusu mu? Değilse `no_evidence`, model yüklenmez.
10. Uygun cevap modunu seç: `extractive`, `generative` veya `fallback_extractive`.
11. `generative` ise groundedness kapısı: üretilen cevap context'e dayanmıyorsa
    `ungrounded` döner ve cevap gösterilmez.
12. Cevabı, kaynakları ve performans sürelerini terminalde göster.

## 4. Dosya Haritası

Modül ve test dosyalarının tek tek ne yaptığı `ls app/ tests/` ve dosyaların
kendi docstring'lerinden okunur; burada yalnızca **koddan çıkarılamayan**
sorumluluk sınırları tutulur.

- `app/database.py`: Veritabanı erişiminin **tek** yeridir. SQLite sorgusunu
  başka modüle dağıtma.
- `app/term_evidence.py`: `normalize_text()`, tokenizasyon ve Türkçe morfoloji
  eşleştirmesinin tek kaynağıdır. `app/sparse_search.py`, `app/eval_metrics.py`
  ve `app/groundedness.py` bunu buradan alır; ikinci bir normalizasyon yolu
  açılmaz.
- `app/groundedness.py`: Cevabın context'e dayanıp dayanmadığını ölçer. Kelime
  kanıtı kapısı **soruyu** ölçer, bu modül **cevabı**. İkisi aynı eşleştiriciyi
  kullanır ama farklı sorulara bakar; birleştirme.
- `app/rag_service.py`: Cevap kararlarını sunumdan bağımsız çalıştırıp
  `RAGResult` döndürür. Terminal gösterimi `app/cli_output.py`'nin işidir; bu iki
  katman birbirine karışmamalıdır.
- `app/retrieval.py`: `rank_chunks()` saf sıralamadır ve veri erişimi yapmaz;
  `get_top_chunks()` onu veritabanına bağlar. Bu ayrım `tools/chunking_analysis.py`
  indekse dokunmadan ölçebilsin diye var.
- `app/project.py`: Aktif docs/data köklerini çözer. Yeni bir modüle yol sabiti
  eklersen `main.apply_project_paths()` içinde de bağla.
- `tools/*.py`: Keşif araçları; uygulamanın parçası **değildir** ve indekse
  dokunmaz. Eşik, eşleştirici ve chunking kararlarını beslemek için çalıştırılır.
- `eval_cases.json` / `eval_baseline.json`: Ground truth içerik imzalarıyla
  etiketlenir (chunk ID ile değil) ve son onaylanan metrikleri tutar.
- `docs/`: İndekslenecek kullanıcı dokümanları. `docs/LEARNING_NOTES.md`
  indekslenmez.
- `data/`: Üretilen yerel SQLite verisi; Git'e eklenmez.
- `INSTRUCTIONS.md`: İlk proje hedefleri; güncel gerçeklik için kodu ve bu
  dosyayı esas al.

## 5. Korunması Gereken Davranışlar

- Türkçe terminal deneyimini koru.
- LLM, dokümanlarda bulunmayan bilgiyi eklememeli.
- Kapsam dışı sorular cevap olarak gösterilmemeli. DİKKAT: bu artık "LLM'e
  gönderilmeden" demek değildir. Ön kapı alan filtresine indirildiği için hard
  negative sorular kasıtlı olarak modele ulaşır; kararı modelin reddi ve
  groundedness kapısı verir. Ön kapıyı "cevap var mı" kapısına geri çevirme,
  ölçüldü ve ayırt edemiyor.
- Modelin `NO_EVIDENCE_ANSWER` üretmesi NİHAİ cevaptır; kaynak metinle
  değiştirilmemeli. Eski `false_no_evidence` koruması bunu yapıyordu ve modelin
  doğru reddini siliyordu.
- `extractive` ve `fallback_extractive` yolları ne modelden ne groundedness'tan
  geçer; ikisi de aynı iddiayı yapar ("bu kaynak metni cevaptır") ve aynı güçlü
  kelime kanıtını ister. Groundedness onları koruyamaz: metin zaten context'ten
  gelir ve tanım gereği dayanaklıdır — dayanaklı olmak ALAKALI olmak değildir.
  `extractive`te kanıt yetersizse soru reddedilmez, üretken yola düşer;
  `fallback_extractive`te düşecek yol kalmadığı için `no_evidence` döner.
- Kaynak adı, sayfa, chunk ve skor bilgisi model cevabının içinde değil, ayrı kaynak bölümünde gösterilmeli.
- LLM boş, çok kısa, etiket ağırlıklı veya hatalı cevap verirse en iyi kaynak chunkına fallback yapılmalı.
- LLM yalnızca gerektiğinde lazy-load edilmelidir; uygulama açılışında zorunlu olarak yüklenmemelidir.
- İlk embedding ve ilk model çağrısının sonraki çağrılardan yavaş olması normaldir.
- `/reindex` başarısız olduğunda eski indeks korunmalıdır. Yeni indeks tamamen hazırlanmadan mevcut kayıtları silme.
- Chunklar ile `source_manifest` aynı transaction içinde değişmelidir; biri başarısızken diğeri commit edilmemelidir.
- Chunk başlangıçlarını mümkünse cümle, değilse kelime sınırına hizala. Uzun ve noktalamasız metinler dışında kelime ortasından chunk başlatma.
- PDF sayfa metadata'sını ve kaynak gösterimini koru.
- Bozuk PDF'lerde `pypdf` tarafından yazılan `Ignoring wrong pointing object` uyarısı, metin çıkarılıyorsa tek başına hata sayılmaz.

## 6. Güncel Ayarlar

```python
SIMILARITY_THRESHOLD = 0.20
CONTEXT_SCORE_THRESHOLD = 0.35
CONTEXT_RELATIVE_SCORE_MARGIN = 0.20
TOP_K = 3
NEIGHBOR_CHUNK_RADIUS = 1
MAX_CONTEXT_CHUNKS = 5

USE_HYBRID_SEARCH = True
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 2

TERM_EVIDENCE_THRESHOLD = 0.21          # ön kapı: yalnızca alan filtresi
TERM_EVIDENCE_MIN_PREFIX = 5
TERM_EVIDENCE_MIN_SHORT_ROOT = 3
TERM_EVIDENCE_MIN_TERM_LENGTH = 3

GROUNDEDNESS_THRESHOLD = 0.50           # cevabın dayanaklı cümle oranı
GROUNDEDNESS_SENTENCE_SUPPORT = 0.60    # bir cümlenin desteklenme oranı
GROUNDEDNESS_MIN_SENTENCE_TERMS = 2

USE_EXTRACTIVE_FALLBACK = True
EXTRACTIVE_SCORE_THRESHOLD = 0.50
EXTRACTIVE_TERM_EVIDENCE_MIN = 0.675    # kısayol her iki kapıyı da atlar
MAX_EXTRACTIVE_CHARS = 500
MIN_GENERATIVE_ANSWER_CHARS = 30

CHUNK_SIZE = 128
CHUNK_OVERLAP = 20
```

Bu değerler mevcut küçük veri seti ve regression testlerine göre seçildi. Değiştirilecekse önce gerekçeyi açıkla, eval vakası ekle ve eski/yeni sonucu karşılaştır.

## 7. Güncel Durum

Son doğrulanan durumda:

- 12 kaynak dosya ve 217 chunk bulunuyor.
- Unit testler `262/262` başarılı.
- Eval seti 125 vaka içerir (112'si etiketli, 13'ü `not_found`). Retrieval
  metrikleri: `Recall@1 = 0.8973`, `Recall@3 = 0.9911`, `Recall@5 = 0.9911`,
  `MRR = 0.9464`; `124/128` vaka geçiyor.
- **Kelime kanıtı kapısı ayırt ediciliğini kaybetti ve karar groundedness'a
  taşındı.** 112 etiketli vakayla ölçüldüğünde ayrım boşluğu yalnızca daralmadı,
  işaret değiştirdi: meşru soru `0.27`, tuzak soru `0.65`. Hiçbir eşik bu iki
  grubu ayıramaz; daha önce raporlanan `0.02`'lik boşluk 36 vakalık setin
  ürettiği bir yanılsamaymış. Kapı silinmedi, görevi daraltıldı
  (`0.675 -> 0.21`, artık yalnızca alan filtresi) ve asıl karar
  `app/groundedness.py`e geçti. Sonuç: kapının reddettiği 6 meşru vakanın 6'sı
  da geçiyor, 13 `not_found` vakasının 13'ü hâlâ reddediliyor.
- Kalan 4 başarısız vaka **sıralama** hatasıdır ve reranking'in konusudur;
  `known_gap` yapılmadı, ölçülür hâlde bırakıldı: `ds_stack_vs_queue`,
  `arch_monolith_vs_microservices`, `phrasing_high_coverage_sufficiency`,
  `phrasing_nlp_evidence_check`. Hepsinde doğru kaynak 2. sırada.
- **Kararın bir kısmı artık deterministik değil.** Hard negative sorular
  kasıtlı olarak modele ulaşır ve gerçek modelin reddedip reddetmeyeceğini eval
  ölçemez. Eval bizim tarafımızın sözleşmesini iki dalda sınar: model reddederse
  red nihai olmalı (`no_evidence`), model uydurursa groundedness kesmeli
  (`ungrounded`). Gerçek model davranışı manuel testin konusudur (bölüm 9).
- Tüm ölçüm geçmişi, kalibrasyon gerekçeleri (chunk boyutu, `RRF_K`, kelime
  kanıtı eşiği, hard negative bulguları, IDF ağırlıklandırma vb.) ve mimari
  karar notları artık burada değil: `kalibrasyon-kaydi` skill'ini çağır.

Çalışma ağacında commit edilmemiş kullanıcı/agent değişiklikleri bulunabilir. Her işe başlarken `git status` ve ilgili diff'i oku. Kullanıcının mevcut değişikliklerini geri alma veya ezme.

## 8. Kurulum ve Çalıştırma

```bash
cd /Users/erdemac/Developer/local-rag-assistant
source .venv/bin/activate
pip install -e .
local-rag
```

`python main.py` geriye dönük olarak aynı interaktif oturumu açmaya devam eder.

Alt komutların listesi `local-rag --help`, CLI içi komutlar `/help` ile alınır;
burada tekrarlanmaz.

Tahmin edilemeyecek tek nokta çalışma kökü önceliğidir: `--project` en yüksek,
`LOCAL_RAG_HOME` ikinci, kurulu repository kökü varsayılandır. Global seçenek
alt komuttan **önce** gelir:

```bash
local-rag --project /dosya/yolu/rag-calismasi stats
LOCAL_RAG_HOME=/dosya/yolu/rag-calismasi local-rag
```

Foundry Local modelinin cache'te bulunması gerekir. Güncel varsayılan model `phi-4-mini`dir.

## 9. Test ve Doğrulama

Çalıştırılacak komutlar ve baseline kuralı [`CLAUDE.md`](CLAUDE.md) içindedir;
tek kaynak orasıdır ve burada tekrarlanmaz.

Asgari manuel model testleri (deterministik olmayan tarafı kapsar):

```text
Veri madenciliği süreçleri nedir?
Veri madenciliğinde veri temizleme ne işe yarar?
Hava nasıl?
```

İlk iki generative soruyu aynı oturumda sorarak cold-start ve warm generation
sürelerini ayrı değerlendir.

## 10. Kodlama Kuralları

- Mevcut sade modüler yapıyı koru; küçük özellik için yeni framework ekleme.
- Veritabanı erişimini `app/database.py` dışında dağıtma.
- Embedding modelini sorgu başına yeniden yükleme.
- Structured veriyi string parçalama yerine sözlük, JSON ve SQLite parametreleriyle işle.
- SQLite sorgularında parametreli ifadeler kullan.
- Reindex yazımlarını transaction dışında parça parça commit etme.
- Yeni davranış için küçük ve deterministik test ekle.
- LLM çıktısına bağlı testleri ana regression paketine koyarken nondeterministic olmamasına dikkat et.
- Üretilen `data/`, `.venv/`, cache, model dosyaları ve büyük binary dosyaları Git'e ekleme.
- Kullanıcı istemedikçe commit, push, history rewrite veya destructive Git işlemi yapma.
- Bağımlılık eklemeden önce gerçekten gerekli olup olmadığını değerlendir ve kullanıcıya nedenini açıkla.

## 11. Bilinen Sınırlamalar

- Görüntü tabanlı PDF'ler için OCR yoktur.
- Türkçe gramer kalitesi otomatik olarak güvenilir biçimde ölçülmüyor.
- `phi-4-mini` zaman zaman küçük anlatım bozuklukları üretebilir.
- Bütün embeddingler SQLite'tan belleğe alınır; mevcut yaklaşım küçük/orta koleksiyonlara uygundur.
- SQLite içinde JSON embedding saklamak öğrenme ve V1 için uygundur, büyük ölçek için değildir.
- Çok uzun ve noktalamasız metinlerde chunk başlangıcı tam cümleye hizalanamayabilir.
- Eval seti 84 vaka (71'i etiketli) içerir; sekiz yeni doküman için doküman
  başına 6 vaka yazıldı. Vakaların bir bölümü, kelime kanıtı kapısı reddettiği
  için chunk metnine yaklaştırılarak yeniden yazıldı; bu, ölçümü gerçek
  kullanıcı sorularından bir miktar kolaylaştırır ve akılda tutulmalıdır.
- Oturum geçmişi listeleme, tekrar ve export içindir; konuşma context'i veya takip sorusu çözümleme amacıyla kullanılmaz ve uygulama kapanınca bellekten silinir.
- Terminal giriş geçmişi yerelde tutulur ancak konuşma context'i amacıyla kullanılmaz.

## 12. Öncelikli Roadmap

Tamamlanan özelliklerin listesi [`README.md`](README.md) içindedir; burada
yalnızca sıradaki hedefler tutulur.

Sıradaki hedefleri şu sırayla ele al:

0. ~~**Eval güçlendirmesi.**~~ **Tamamlandı.** Recall@k ve MRR metrikleri,
   içerik imzasıyla etiketlenmiş ground truth, 6 hard negative vaka, bozuk etiket
   tespiti (`case_labels`) ve `eval_baseline.json` üzerinden `--compare` akışı.
1. ~~**Chunking karşılaştırma deneyi.**~~ **Tamamlandı.**
   `tools/chunking_analysis.py` yedi ayarı indekse dokunmadan ölçtü;
   `CHUNK_SIZE` 110'dan 128'e çıkarıldı. Ayrıntı bölüm 7'de.
2. ~~**Yanlış pozitif savunması.**~~ **Tamamlandı.** Karar sorudan cevaba
   taşındı: kelime kanıtı kapısı alan filtresine indirildi (`0.675 -> 0.21`) ve
   `app/groundedness.py` üretilen cevabın context'e dayanıp dayanmadığını cümle
   bazlı ölçüyor. Birlikte kaldırılması zorunlu olan iki bağlantı da yapıldı:
   `false_no_evidence` koruması (modelin doğru reddini siliyordu) ve extractive
   kısayolunun kanıtsız çalışması. Sonuç: 6 yanlış ret çözüldü, 13 tuzak vaka
   hâlâ reddediliyor, eval `118/128 -> 124/128`.
   Kalan tek parça: **eşiklerin yeniden kalibrasyonu.** `SIMILARITY_THRESHOLD`,
   `CONTEXT_SCORE_THRESHOLD` ve `EXTRACTIVE_SCORE_THRESHOLD` hâlâ elle seçilmiş
   değerlerdir ve taşıdıkları yük bu değişiklikle yeniden değişti.
   Ölçüm geçmişi ve gerekçe: **`kalibrasyon-kaydi` skill'ini çağır**.
3. **Hybrid search.** ~~Yapıldı.~~ `app/sparse_search.py` (BM25) +
   `app/retrieval.py` (RRF). Karar: birleşik skor yalnızca sıralamada kullanılır,
   kapı skoru cosine kalır. Sonuç: `Recall@1` 0.7826 -> 0.9783, `MRR` 0.8551 -> 1.0000.
   Ayrıntı ve SQLite FTS5'in neden tercih edilmediği: `kalibrasyon-kaydi` skill.
4. **Reranking — sıradaki iş.** Geniş aday havuzunu cross-encoder ile yeniden
   sırala. Ön şart karşılandı: eval seti 112 etiketli vakaya çıktı ve metrikler
   tavandan indi (`MRR = 0.9464`). Ölçülecek boşluk somut: eval'de kalan 4
   başarısız vakanın dördünde de doğru kaynak 2. sırada. Aynı cross-encoder
   groundedness'ın kör noktasını da kapatabilir (aşağıya bak).
   Tam gerekçe için `kalibrasyon-kaydi` skill'ini çağır.
   *Mimari karar: aday sayısı ve kabul edilebilir latency bütçesi.*
5. **Conversation history.** Asıl iş takip sorusu değil, query rewriting; soruyu
   retrieval'a bağımsız (standalone) biçimde ver.
   *Mimari karar: rewriting tasarımı ve geçmiş bütçesi.*

Bu sıranın gerekçesi: 0–2 **ölçme yeteneği** kazandırır, 3–5 ise ancak o yetenek
varsa öğretici ve doğrulanabilir olur. Hybrid search ve reranking'in iddiası
"retrieval kalitesini artırmak"tır; iyi bir eval olmadan bu iddia ölçülemez.

Fırsat buldukça ele alınacaklar (öncelikli değil):

- **Groundedness'ın bilinen kör noktası.** Kontrol "cevap context'e dayanıyor
  mu" diye sorar, "cevap soruyu yanıtlıyor mu" diye değil. Retrieval alakasız
  ama gerçek bir metin getirir ve model onu özetlerse cevap DAYANAKLI çıkar.
  Bu durumda tek savunma modelin kendi reddidir. Cross-encoder'a geçilirse bu
  boşluk da kapanır; 4. maddeyle birlikte ele alınabilir çünkü aynı model iki
  işi birden yapar.
- **Incremental reindex.** Ancak `time local-rag reindex` rahatsız edici hale
  geldiğinde. Mevcut ölçekte (12 dosya, 217 chunk) çözülecek bir problem yok.
- **Ölçekleme deneyi.** Chunk sayısını sentetik olarak artırıp brute force
  cosine aramanın nerede kırıldığını ölç. Bu bir vector database migration'ı
  değildir; ANN/HNSW'nin neden var olduğunu ölçerek öğrenmek içindir.
- **OCR desteği.** Yalnızca gerçekten taranmış PDF ihtiyacı doğarsa.

Kapsam dışı bırakılanlar: vector database'e geçiş (mevcut ölçekte gereksiz,
öğrenme değeri kütüphane öğrenmekten ibaret) ve otomatik model karşılaştırma
raporu (`app/benchmark.py` bu ihtiyacı zaten karşılıyor). Web arayüzü ve API
sunucusu proje hedefinde değildir.

## 13. Bir Görevi Tamamlama Kriteri

Bir değişikliği tamamlanmış saymadan önce:

1. İlgili kodu ve mevcut kullanıcı değişikliklerini okumuş ol.
2. Davranışı mümkünse deterministik testle kapsa.
3. Unit test, eval ve sözdizimi kontrolünü çalıştır.
4. Gerekliyse kullanıcıya exact manuel model test komutunu ver.
5. Türkçe UX, kaynak doğruluğu ve fallback davranışını koruduğunu doğrula.
6. Ne değiştiğini, neden değiştiğini ve test sonucunu kısa biçimde kullanıcıya açıkla.
7. Çalıştıramadığın bir doğrulama varsa bunu açıkça belirt.
