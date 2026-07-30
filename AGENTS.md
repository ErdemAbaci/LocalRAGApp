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
9. Uygun cevap modunu seç: `extractive`, `generative` veya `fallback_extractive`.
10. Cevabı, kaynakları ve performans sürelerini terminalde göster.

## 4. Dosya Haritası

- `main.py`: İnteraktif ve argparse tabanlı terminal entrypoint'i; proje yolu seçimi, komut yönlendirme ve yapılandırılmış RAG sonucunun Rich ile gösterimini yönetir.
- `pyproject.toml`: `local-rag` console script'i, paket sürümü ve doğrudan Python bağımlılıkları.
- `app/__init__.py`: Paket sürümünü (`0.1.0`) tutar.
- `app/config.py`: Similarity, context, extractive ve cevap kalite eşikleri.
- `app/cli_output.py`: Rich konsolu, bordo renk teması, mini terminal maskotlu banner, tablolar, semantik cevap paneli, tek satırlı RAG aşama göstergesi, streaming cevap paneli, standart hata/uyarı ve Türkçe performans çıktısı.
- `app/cli_input.py`: `prompt-toolkit` ile çerçeveli giriş, giriş üstünde canlı slash menüsü, komut renklendirme, parametre ipuçları, model/kaynak/indeks/filtre durum satırı, proje bazlı kalıcı geçmiş, `Esc`/`Ctrl+L`/bağlamsal `Ctrl+C` ve bağlama duyarlı tamamlama; TTY olmayan kullanımda sade fallback.
- `app/benchmark.py`: Sabit RAG contextleriyle model yükleme, cold/warm generation, cevap geçerliliği ve terim kapsamı benchmark'ı.
- `app/database.py`: SQLite şeması, chunk/manifest okuma-yazma, istatistikler ve atomik `replace_chunks()` işlemi.
- `app/document_manager.py`: TXT/PDF doğrulama, üzerine yazmadan güvenli kopyalama ve `docs/` sınırında dosya silme işlemleri.
- `app/index_state.py`: Doküman SHA-256 manifestini üretir; eklenen, değişen ve silinen kaynakları indeksle karşılaştırır.
- `app/ingest.py`: TXT/PDF okuma, embedding tokenizer'ına göre 110/20 tokenlık cümle odaklı chunking, embedding hazırlama, doküman değişim koruması ve reindex akışı.
- `app/embeddings.py`: Yerel Hugging Face snapshot'ını tercih eden embedding lazy-load/cache yönetimi.
- `app/health.py`: `/doctor` için doküman, veritabanı, embedding ve Foundry/model cache sağlık kontrolleri.
- `app/retrieval.py`: Soru embeddingi, normalize edilmiş cosine hesabı, geçersiz vektör kontrolü, dense/sparse sıralamaların RRF ile birleştirilmesi, kapı skorunun sıralamadan ayrılması ve gerçek skorlu komşu chunk adayları.
- `app/prompts.py`: Türkçe, yalnızca context'e dayalı RAG promptu.
- `app/project.py`: `--project`, `LOCAL_RAG_HOME` ve varsayılan repository kökünden aktif docs/data/history/export yollarını çözer.
- `app/rag_service.py`: Retrieval, relevance/context seçimi, belge sıralı prompt, sınırlı komşu genişletme, cevap modu, streaming callback'i, fallback ve süre kararlarını sunumdan bağımsız çalıştırıp `RAGResult` döndürür.
- `app/llm.py`: Süre sınırlı Foundry başlangıcı, `LOCAL_RAG_MODEL`, streaming completion, cevap temizleme ve tekrar döngüsü dahil kalite doğrulaması.
- `app/session.py`: Başarılı RAG sonuçlarının oturum içi geçmişini, tekrar seçimini ve üzerine yazma korumalı Markdown/JSON export'unu yönetir.
- `benchmark_cases.json`: Modellerin aynı context ve beklenen kavramlarla karşılaştırıldığı üretken cevap vakaları.
- `eval.py`: İndeks sağlığı, imza doğrulaması, cevap kalite kararı, retrieval regression değerlendirmesi, Recall@k/MRR raporu ve `--compare`/`--update-baseline` baseline akışı.
- `app/sparse_search.py`: BM25 ile kelime örtüşmesi skoru ve soru kelimelerinin IDF ayırt edicilik ağırlıkları. Tokenizasyon ve morfoloji `app/term_evidence.py`'den gelir; ikinci bir normalizasyon yolu açılmaz.
- `app/term_evidence.py`: Soru kelimelerinin context'te bulunma oranı (IDF ağırlıklı); Türkçe normalizasyon, stopword listesi, ortak kök eşleştirmesi, kısa kök kuralı ve ünsüz yumuşaması. `normalize_text()` buranın tek kaynağıdır.
- `tests/test_term_evidence.py`: Türkçe normalizasyon, stopword bütünlüğü, önek/yumuşama eşleştirmesi ve kapsama/kapı davranışı testleri.
- `app/eval_metrics.py`: Retrieval metriklerinin saf hesabı; imza eşleştirme, Recall@k, MRR, baseline karşılaştırması ve Türkçe duyarlı metin normalizasyonu.
- `eval_cases.json`: Deterministik retrieval test soruları, içerik imzaları ve hard negative vakaları.
- `eval_baseline.json`: Son onaylanan Recall@k/MRR ve hard negative skorları; `--compare` bu dosyaya göre fark üretir.
- `tests/test_eval_metrics.py`: İmza eşleştirme, Türkçe normalizasyon, Recall@k, MRR, bozuk etiket tespiti ve baseline karşılaştırma testleri.
- `tests/test_benchmark.py`: Benchmark hazırlama, cold/warm ölçüm, kalite ve hata raporu testleri.
- `tests/test_eval.py`: Kaynak, skor, en iyi chunk ve genişletilmiş context kavramı değerlendirme testleri.
- `tests/test_ingest.py`: Token sınırı, cümle/kelime hizası, atomik reindex, `/sources` şema güvenliği ve CLI çıktı testleri.
- `tests/test_health.py`: `/doctor` başarı, uyarı, hata ve CLI çıktı testleri.
- `tests/test_document_manager.py`: Add/remove doğrulama, güvenlik, onay, CLI ve indeks güncelliği entegrasyon testleri.
- `tests/test_index_state.py`: Güncel, eski, manifestsiz ve eksik indeks senaryolarını test eder.
- `tests/test_cli_output.py`: Standart hata gösterimi, `/model`, `/config`, lazy-load ve CLI oturum dayanıklılığı testleri.
- `tests/test_embeddings.py`: Yerel embedding snapshot tercihi ve cache-miss fallback testleri.
- `tests/test_entrypoint.py`: Paket metadata'sı, `local-rag` alt komutları, exit code ve ortak soru akışı testleri.
- `tests/test_llm.py`: Parça etiketi temizliği, streaming/iptal bağlantı kapatma ve sessiz/debug Foundry servis başlangıcı testleri.
- `tests/test_retrieval.py`: Normalize edilmiş cosine skorunun sıralama/sonlu değer, RRF füzyonu, sıra numaralandırma, kapı skorunun sıralamadan bağımsızlığı ve belge sıralı komşu aday testleri.
- `tests/test_sparse_search.py`: BM25 tf doygunluğu, IDF, Türkçe ek eşleştirmesi ve IDF ağırlık testleri.
- `tests/test_rag_service.py`: Yapılandırılmış sonuç, relevance ile seçim, belge sıralı prompt, komşu sınırı/rolü, cevap modları, fallback, kaynak filtresi, streaming ve iptal callback'lerini test eder.
- `tests/test_source_tools.py`: Parametreli kaynak filtresi, `/show`, `/filter` ve `ask --source` davranışlarını test eder.
- `tests/test_project.py`: Proje yolu önceliğini ve runtime modüllerinin aynı docs/data köküne bağlanmasını test eder.
- `tests/test_cli_input.py`: Kalıcı giriş geçmişi, dosya izinleri, klavye kısayolları ve bağlama duyarlı Tab tamamlama testleri.
- `tests/test_session.py`: Oturum kaydı, tekrar filtresi, iptal davranışı ve güvenli Markdown/JSON export testleri.
- `README.md`: Kurulum, kullanım, mimari, benchmark, test sonuçları, sınırlamalar ve V2 yol haritasını özetleyen ana proje sunumu.
- `PROJECT_GUIDE.md`: Projenin uzun, öğretici teknik anlatımı ve roadmap'i.
- `INSTRUCTIONS.md`: İlk proje hedefleri; güncel gerçeklik için her zaman kodu ve bu dosyayı esas al.
- `CLAUDE.md`: Claude Code için çalışma kuralları ve kodda gözden kaçan noktalar; proje bağlamı için bu dosyaya yönlendirir.
- `docs/LEARNING_NOTES.md`: Projedeki RAG kavramlarının sıfırdan açıklaması ve ileriki kararlar için terim referansı. İndekslenmez.
- `tools/term_evidence_analysis.py`: Keşif aracı; uygulamanın parçası değildir. Soru kelimelerinin modele giden context'te geçme oranını hem eşit sayarak hem IDF ağırlıklı olarak, birden çok eşleştiriciyle vaka ve grup bazında ölçer. Eşik ve eşleştirici kararlarını beslemek için kullanılır.
- `tools/hybrid_search_analysis.py`: Keşif aracı; uygulamanın parçası değildir. Dense ve hybrid sıralamayı `RRF_K` adaylarıyla Recall@k/MRR üzerinden karşılaştırır ve sıra değişimlerini vaka bazında gösterir.
- `docs/`: İndekslenecek kullanıcı dokümanları (`.txt` ve `.pdf`).
- `data/`: Üretilen yerel SQLite verisi; Git'e eklenmez.

## 5. Korunması Gereken Davranışlar

- Türkçe terminal deneyimini koru.
- LLM, dokümanlarda bulunmayan bilgiyi eklememeli.
- Kapsam dışı sorular LLM'e gönderilmeden reddedilmeli.
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

TERM_EVIDENCE_THRESHOLD = 0.67
TERM_EVIDENCE_MIN_PREFIX = 5
TERM_EVIDENCE_MIN_SHORT_ROOT = 3
TERM_EVIDENCE_MIN_TERM_LENGTH = 3

USE_EXTRACTIVE_FALLBACK = True
EXTRACTIVE_SCORE_THRESHOLD = 0.50
MAX_EXTRACTIVE_CHARS = 500
MIN_GENERATIVE_ANSWER_CHARS = 30

CHUNK_SIZE = 110
CHUNK_OVERLAP = 20
```

Bu değerler mevcut küçük veri seti ve regression testlerine göre seçildi. Değiştirilecekse önce gerekçeyi açıkla, eval vakası ekle ve eski/yeni sonucu karşılaştır.

## 7. Güncel Durum

Son doğrulanan durumda:

- 5 kaynak dosya ve 47 chunk bulunuyor. En uzun chunk özel tokenlar dahil 109 tokendır; embedding modelinin 128 token sınırını aşan parça yoktur. `cybersecurity.txt` beş ayrı güvenlik konusu içeriyor.
- Retrieval, indeks ve cevap kararı değerlendirmesi `38/38` başarılı; bilinen boşluk (`GAP`) kalmadı.
- Retrieval metrikleri: `Recall@1 = 0.8636`, `Recall@3 = 0.9773`, `Recall@5 = 1.0000`, `MRR = 0.9318` (22 etiketli vaka).
  Bu değerler hybrid search sonrasıdır. Yalnızca dense ile ölçüm: `0.7273 / 0.8864 / 0.9545 / 0.8220`.
  `Recall@5` hybrid'den önce de `1.0` idi; yani sorun doğru parçayı **bulmak** değil
  **sıralamak**tı ve iyileşme tam olarak orada gerçekleşti.
- Hard negative ölçümü kritik bir sınırı ortaya çıkardı: cevabı dokümanda hiç
  bulunmayan `hard_negative_firewall_rules` sorusu `0.5985` alırken, cevabı
  bulunan `rag_definition` `0.5570` alıyor. Yani **hiçbir tek `SIMILARITY_THRESHOLD`
  değeri bu ikisini ayıramaz.** Cosine similarity konu benzerliğini ölçer, cevap
  içerip içermediğini değil. Bu boşluğun çözümü eşik ayarı değil; BM25 terim
  kanıtı, cross-encoder ve groundedness kontrolüdür.
- **Hybrid search eklendi ve sıralama sorununu ölçülebilir biçimde düzeltti.**
  `app/sparse_search.py` BM25 ile kelime örtüşmesini ölçer; `app/retrieval.py`
  dense ve sparse sıralamaları RRF (`1/(k + sıra)` toplamı) ile birleştirir.
  Ölçülen sonuç: `Recall@1` 0.60 -> 0.80, `MRR` 0.775 -> 0.90. Manuel testte
  bozuk cevaba yol açan "Kimlik avından nasıl korunulur?" sorusunda cevabı içeren
  chunk `4.` sıradan `1.` sıraya çıktı. Sıra değişen üç vaka:
  `phishing_protection` (4 -> 1), `data_transformation` (2 -> 1),
  `data_mining_process` (2,5 -> 2,4).
  Birleşik skor **yalnızca sıralama** için kullanılır. Kapı ve kullanıcıya
  gösterilen skor cosine kalır; `retrieval.gate_score()` bunu sıralamadan
  bağımsız olarak en yüksek cosine değerinden okur. Aksi halde dört eşiğin
  tamamı, hard negative `max_score` kontrolü ve gösterilen skor yeni bir ölçeğe
  göre yeniden kalibre edilmek zorunda kalırdı.
  `RRF_K` iki kez ölçüldü. İlk ölçümde (24 chunk, 11 vaka) 1 ile 60 arası
  ayırt edilemedi ve gelenek olan 60 seçildi. Korpus 47 chunk'a, set 22 vakaya
  çıkınca fark ortaya çıktı: k büyüdükçe sonuç monoton kötüleşiyor
  (k=1,2 -> MRR 0.9318; k=3,4 -> 0.9091; k=5..60 -> 0.9015). `RRF_K = 2`
  seçildi. Sebep mekanizmada: büyük k iki listede de ortalarda kalanı, küçük k
  tek listede tepe yapanı ödüllendirir; bu korpusta BM25'in birebir terim
  eşleşmesi cosine'den güvenilir, çünkü çok dilli embedding Türkçe'de zayıf.
- **Hybrid search kelime kanıtı kapısının kör noktasını açığa çıkardı ve kapı
  IDF ağırlıklarına geçirildi.** İki mekanizma da kelime örtüşmesine bakıyor;
  retrieval güçlenince kapı sızdırdı. `hard_negative_firewall_rules` sorusunda
  hybrid, yedekleme chunk'ını öne çekti ve o chunk'taki "3-2-1 kuralı" ifadesi
  sorunun `kuralları` kelimesiyle eşleşti; kapsama `0.33`'ten `0.67`'ye çıkıp
  eşiği geçti. Ayırt edici kelime olan `duvarı` dokümanlarda hiç yok, ama oran
  bütün kelimeleri eşit sayıyordu.
  Ölçüm sırasında ikinci ve daha temel bir hata bulundu: `avından` kelimesi
  korpusta hiçbir şeyle eşleşmiyordu, çünkü metindeki karşılığı `avı` yalnızca
  3 karakter ve ortak kök şartı 5. Yani "kelime gerçekten yok" ile "eşleştirici
  kaçırdı" aynı görünüyordu ve ağırlık tek başına yalnızca `0.02` boşluk verdi.
  `terms_match()`'e kısa kök kuralı eklendi (kök `min_short` karakterden uzunsa
  tamamen kapsanması yeterli). Ağırlıklı kapsamada ayrım boşluğu `0.02` -> `0.21`
  oldu; eşik `0.60` -> `0.70` olarak aralığın ortasına alındı.
  Bedeli: kısa kök kuralı BM25'i de gevşetti, bu yüzden hybrid'in `Recall@1`
  katkısı 0.85'ten 0.80'e indi. Kapı doğruluğu sıralama doğruluğuna tercih
  edildi; yanlış cevap üretmek, doğru cevabı 2. sıraya düşürmekten kötüdür.
- **Soru kalıbı kelimeleri IDF ile birlikte yeni bir yanlış ret ürettti.**
  Manuel testte "Çok faktörlü doğrulama neden önemli?" reddedildi. `önemli`
  dokümanlarda hiç geçmediği için en yüksek ağırlıklardan birini (`2.30`)
  alıyor ve kapsamayı `0.579`'a çekiyordu. Oysa bir şeyin önemini soran cevabın
  metinde "önemli" kelimesini içermesi gerekmez; bu bir içerik kelimesi değil
  soru kalıbıdır. `önemli`, `önemlidir`, `gerekli`, `gereklidir`
  `QUESTION_STOPWORDS`'e eklendi ve vaka (`multi_factor_importance`) eval'e
  girdi. Ayrım boşluğu 20 vakayla yeniden ölçüldü ve `0.21`'de kaldı; yani
  stopword eklemek kapıyı zayıflatmadı.
  Genel kural: IDF "nadir kelime = ayırt edici" varsayar. Bu varsayım soru
  kalıbı kelimeleri için geçersizdir, çünkü onlar soruda bulunup dokümanda
  bulunmamayı zaten doğal olarak yapar. Stopword listesi bu yüzden IDF'in
  yerine geçmez, ön koşuludur.
- **Korpus 24'ten 47 chunk'a çıkarıldı; kalibrasyonların korpusa bağlı olduğu
  ölçüldü.** `docs/versiyon_kontrol.txt` ve `docs/yazilim_testi.txt` eklendi;
  eval seti 20'den 35 vakaya, etiketli vaka 11'den 22'ye çıktı. Yeni dokümanlar
  bilinçli olarak mevcut hard negative konularını (yedekleme sıklığı, parola
  uzunluğu, fidye yazılımı aracı, k-means küme sayısı, min-max formülü, güvenlik
  duvarı) içermez; içerselerdi o vakalar sessizce geçersizleşirdi.
  Genel ders: IDF ağırlıkları korpustan gelir, bu yüzden **doküman eklemek eşik
  kalibrasyonunu değiştirir.** Reindex sonrası `tools/term_evidence_analysis.py`
  ve `tools/hybrid_search_analysis.py` yeniden çalıştırılmalıdır.
- ~~**Açık sorun: `RRF_K` tepe sinyali cezalandırıyor.**~~ **Çözüldü.** Manuel
  testte "Yedekleme neden gereklidir?" sorusunda BM25 doğru chunk'ı `1.` sıraya
  koymuşken füzyon olay müdahalesi chunk'ını seçiyordu: o chunk iki listede de
  ortalarda (dense 2., sparse 3.), doğrusu ise yalnızca birinde tepe (dense 5.,
  sparse 1.). `RRF_K = 60` istikrarı ödüllendirdiği için doğru chunk eleniyordu.
  Korpus büyüdükten sonra tarama tekrarlandı, fark ölçülebilir hale geldi ve
  `RRF_K = 2` seçildi; vaka artık `1.` sırada.
  Not: dense skorun bu soruda `0.1972` kalması ayrı bir zayıflıktır; embedding
  modeli birebir konu eşleşmesini bile yakalayamadı. Hybrid search'ün bu
  korpusta neden gerekli olduğunun en net kanıtı budur.
- **Kelime kanıtı eşiği üçüncü kez kalibre edildi: `0.70` -> `0.67`.**
  47 chunk'ta ayrım boşluğu `0.21`'den `0.09`'a düştü (tuzak max `0.63`, alakalı
  min `0.72`). Boşluk daralıyor çünkü korpus büyüdükçe soru kelimelerinin bir
  kısmı kaçınılmaz olarak başka dokümanlarda da geçiyor. Ayrıca `arasındaki`,
  `fark`, `yazılmalıdır` gibi soru kalıbı kelimeleri stopword listesine eklendi;
  dokümanlarda hiç geçmedikleri için en yüksek IDF ağırlığını alıp meşru
  soruları reddediyorlardı (`stub_vs_mock`, `commit_message_guidance`,
  `unit_vs_integration_test`). Bu, `önemli` bulgusunun aynısıdır ve listenin
  elle bakım gerektiren zayıf bir nokta olduğunu gösterir. Boşluk daralmaya
  devam ederse oran tabanlı kapıdan groundedness kontrolüne geçilmelidir.
- **Düzeltilen hata: context seçimi sıralamayla çelişiyordu.**
  `select_matched_context_chunks()` context'i cosine eşiğine göre seçerken liste
  hybrid sıraya göre geliyordu. Ölçümde birinci sıradaki doğru chunk elenip
  üçüncü sıradaki alakasız chunk context'e girdi ("Saplama ile taklit nesne
  arasındaki fark nedir?": doğru chunk cosine `0.3147` ile eşiği geçemedi,
  alakasız chunk `0.3623` ile geçti). Artık `chunks[0]` her zaman context'e
  girer. Sıralamanın birincisini elemek, hybrid search'ün kazandırdığını geri
  vermektir.
- **Eval açığı kapatıldı: alakalı vakalar artık kapıyı da kontrol ediyor.**
  Önceden `evaluate_relevant_case()` yalnızca kaynak, skor ve sırayı
  doğruluyordu; retrieval doğru parçayı bulup kelime kanıtı kapısı onu
  reddettiğinde vaka PASS görünüyor, kullanıcı ise "Bu bilgi verilen
  dokümanlarda yok." cevabı alıyordu. `stub_vs_mock` tam olarak böyle geçti.
  Kontrol LLM yüklemeden yalnızca kapıyı çalıştırır.
- **Tekrar döngüsü artık akış sırasında kesiliyor.** Manuel testte
  "Çakışma nasıl çözülür?" sorusunda `phi-4-mini` aynı cümleyi yaklaşık 20 kez
  üretti. Doğrulama bunu yakaladı ve `fallback_extractive`e düştü, yani sonuç
  doğruydu; ama kontrol generation bittikten sonra çalıştığı için işlem
  `31.5` saniye sürdü ve kullanıcı bu süre boyunca tekrarı canlı izledi.
  `has_repeating_trigram()` akış döngüsünde çalışır ve üçüncü tekrarda akışı
  keser. Erken kesmede yalnızca trigram kuralı kullanılır çünkü o **monotondur**;
  kelime oranı kuralı metin uzadıkça yeniden altına düşebildiği için meşru bir
  cevabı yarıda bırakabilirdi. Sonuç değişmez, süre değişir.
- **Açık sorun: context kirlenmesi.** "Çok faktörlü doğrulama neden önemli?"
  sorusunda doğru chunk (216) hem cosine hem BM25'te açık ara birinci, ama
  `datamining.pdf`'ten üç parça da mutlak eşiği geçtiği için context'e girdi.
  Model karışık metinden bozuk cevap üretti, `get_answer_validation_error`
  yakaladı ve `fallback_extractive`e düşüldü. Koruma çalıştı ama context seçimi
  hâlâ mutlak cosine eşiğine bakıyor; sparse kanıt şartı veya füzyon sırasına
  dayalı bir üst sınır değerlendirilmeli.
- **Kelime kanıtı kapısı eklendi ve yukarıdaki hatayı kapattı.**
  `app/term_evidence.py`, sorunun ayırt edici kelimelerinin seçilen context'te
  geçme oranını hesaplar; `RAGService.has_term_evidence()` bunu LLM çağrısından
  **önce** uygular. Kanıt yoksa cevap `no_evidence` olur ve model hiç yüklenmez.
  Kapı hem extractive hem generative yolu kapsar.
  Tek kapı iki sorunu birden çözüyor: kanıtsız soruya uydurma cevap üretilmiyor
  ve LLM'e ancak gerçekten kanıt varken gidildiği için `false_no_evidence`
  koruması artık modelin doğru reddini ezmiyor.
  Eval sonucu: 6 hard negative vakanın 6'sı da `no_evidence` veriyor, hiçbirinde
  LLM yüklenmiyor. Recall@k ve MRR değişmedi; 9 alakalı vaka aynen çalışıyor.
- Türkçe ünsüz yumuşaması `soften_final_consonant()` ile ele alınır
  (`süreç` -> `süreci`, `kitap` -> `kitabı`). Bu olmadan meşru eşleşmeler
  kaçıyordu; düz önek karşılaştırması son harfin değiştiğini göremez.
- **Aşağıdaki hata artık düzeltilmiştir; kayıt olarak tutuluyor.**
  Doğrulanmış hata: yanlış ret koruması tuzak sorularda ters çalışıyordu.
  `app/llm.get_answer_validation_error()` LLM tam olarak `NO_EVIDENCE_ANSWER`
  ürettiğinde bunu `false_no_evidence` sayıp geçersiz kılıyor ve
  `rag_service.generate_with_fallback()` kaynak metnine dönüyor. Bu koruma
  "arama doğru, LLM inatçı" varsayımına dayanıyor. Hard negative sorularda arama
  yanlış olduğu için varsayım tersine dönüyor: manuel testte
  `hard_negative_ransomware_tool` sorusunda model doğru biçimde
  "Bu bilgi verilen dokümanlarda yok." dedi, sistem bu **doğru** cevabı silip
  alakasız bir yedekleme cümlesi gösterdi. Uyarı metni
  "LLM bulunan kanıtı kullanmadı" ile bu yol tanınabilir.
- Kelime kanıtı kapısı eklenmeden önceki manuel LLM testi (6 hard negative,
  `phi-4-mini`): 5 soruda üretken cevap üretildi, 1 soruda yanlış ret koruması
  devreye girdi. Hiçbirinde kullanıcıya kapsam dışı cevabı gösterilmiyordu.
  Kapı eklendikten sonra altısı da LLM'e hiç gitmiyor.
- Retrieval bazı hard negative'lerde tamamen alakasız context seçiyor:
  "Parola en az kaç karakter olmalıdır?" sorusunun en iyi eşleşmesi, kategorik
  verilerin 0/1'e dönüştürülmesini anlatan `datamining.pdf` chunk'ı.
- **Kelime kanıtı ölçümü (17 eval sorusu).** Sorunun içerik kelimelerinin modele
  giden context'te geçme oranı, cosine skorunun ayıramadığı iki grubu net
  ayırıyor:

  | | Cosine | Kelime kanıtı (tam eşleşme) |
  |---|---|---|
  | Alakalı | 0.5570 – 0.8761 | 0.71 – 1.00 |
  | Hard negative | 0.2403 – 0.5985 | 0.00 – 0.33 |
  | Örtüşme | var | yok (0.33 → 0.71 boşluğu) |

  Kaba kök alma (kelimenin ilk 4-5 harfi) bu sinyali **bozuyor**: alakalı en
  düşük 0.80'e çıkarken hard negative en yüksek 0.75'e fırlıyor ve boşluk
  0.38'den 0.05'e düşüyor. Sebep yanlış eşleşmeler: `sayısı`→`sayısal`,
  `yapılandırılmalıdır`→`yapılır`. Bu iş için kesinlik, kapsayıcılıktan
  önemlidir; Türkçe kök alma tasarımı bu kısıtla yapılmalıdır.
- Unit testler `205/205` başarılı (token-aware chunking, göreli context seçimi, soru odaklı extractive fallback, komşu/context sırası, yanlış LLM ret fallback'i, context eval, RAG servis katmanı, streaming/iptal callback'leri, oturum geçmişi/export, kaynak filtreleme/görüntüleme, proje yolu, canlı slash menüsü, klavye kısayolları, terminal durum satırı/renklendirme/ipucu, giriş geçmişi/tamamlama, benchmark, retrieval, atomik reindex/manifest, güvenli TXT/PDF yönetimi ve LLM cevap temizliği).
- `/sources` indeksteki dosya, tür, sayfa ve chunk sayılarını gösterir; boş indeks, eksik `chunks` tablosu ve eski şema senaryolarında çökmez.
- `/doctor` dokümanları, indeks güncelliğini, veritabanını, 384 boyutlu embeddingleri, Foundry kurulumunu ve `phi-4-mini` cache dosyalarını model yüklemeden kontrol eder.
- CLI hataları kullanıcı mesajı ve çözüm önerisi gösterir; teknik exception yalnızca debug modunda görünür.
- CLI Rich tabanlı panel, tablo, yumuşak bordo tema ve TTY işlem göstergeleri kullanır; piped/renksiz çıktıda okunabilir kalır. Açılış panelinde özgün mini terminal robotu bulunur. RAG ilerlemesi arama, model hazırlama ve yanıt üretimini tek satırda günceller.
- TTY üretken cevaplarında ilk tokenla geçici streaming paneli açılır. `Ctrl+C` response'u kapatır, kısmi cevabı kaydetmez ve CLI oturumunu açık tutar; normal bitişte temizlenmiş nihai cevap standart panelde gösterilir.
- Cevap başlığı modun Türkçe adını ve retrieval skorunu gösterir. Üretken cevap yumuşak bordo, doğrudan kaynak cevabı yeşil, fallback amber, kanıt bulunamaması düşük vurgulu gri gösterilir.
- Performans satırı kullanıcıya `Arama`, `Yanıt` ve `Toplam` adlarıyla gösterilir; cevap, kaynak ve süre blokları aynı sol hizayı kullanır.
- LLM'in üretebildiği `[Parça 1]`, `(Parça 1)`, aralık/listeli atıflar ve cevap sonundaki çıplak `Parça 1.` etiketleri temizlenir; kaynak bilgisi yalnızca ayrı kaynak tablosunda gösterilir.
- Foundry servisi normal modda terminale ham başlangıç logu yazmadan başlatılır; `/debug on` açıkken SDK çıktısı korunur.
- `/model` chat/embedding modeli, cache ve oturumdaki lazy-load durumunu inference yapmadan gösterir.
- `/config` aktif eşikleri, cevap kalite ayarlarını, chunk değerlerini ve yolları salt okunur gösterir.
- Proje `pyproject.toml` ile editable kurulabilir; `local-rag` interaktif oturumu, `local-rag ask` tek soruyu ve diğer alt komutlar mevcut ortak fonksiyonları çalıştırır.
- Alt komutlar başarıda `0`, operasyonel hatada `1` exit code döndürür; argparse kullanım hataları `2` döndürür.
- Reindex, desteklenen dokümanların boyut ve SHA-256 özetini `source_manifest` tablosuna chunklarla atomik kaydeder.
- Soru akışı, `/stats` ve `/doctor`; eklenen, değişen veya silinen dokümanları algılayıp reindex önerir. Eski indeks uyarıdan sonra kullanılmaya devam eder.
- Dokümanlar indeksleme devam ederken değişirse yeni kayıtlar yazılmaz ve önceki indeks korunur.
- `/add` ve `local-rag add`; okunabilir, boş olmayan UTF-8 TXT veya metin tabanlı PDF'yi mevcut dosyanın üzerine yazmadan `docs/` içine kopyalar.
- `/remove` ve `local-rag remove`; yalnızca `docs/` içindeki tek dosya adını kabul eder. Varsayılan olarak onay ister; terminal alt komutu otomasyon için `--yes` destekler.
- Add/remove işlemleri pahalı embedding sürecini otomatik başlatmaz; kullanıcıya reindex gerektiğini bildirir ve indeks güncelliği kontrolü değişikliği görünür kılar.
- `local-rag benchmark --models ...` model yükleme, ilk/sıcak üretim, geçerli cevap ve beklenen terim kapsamını karşılaştırıp `data/model_benchmark.json` raporu üretir.
- `RAGService`, cevabı `RAGResult` içinde cevap modu, kaynaklar, skorlar, süreler ve fallback uyarısıyla döndürür; terminal sunumu bu sonucu ayrı katmanda işler.
- Retrieval chunkları sabit `0.35` eşiği ve en iyi skordan en fazla `0.20` uzaklıkla seçer. Üretken akış yalnızca context eşiğini geçen bir önceki/sonraki komşuyu, toplam 5 context chunkı sınırıyla ekler; LLM'e belge sırası verilirken kaynak tablosu relevance sırasını ve `Eşleşme`/`Komşu` rolünü korur. Extractive cevaplar komşuyla genişletilmez.
- Retrieval yeterli kanıt bulduğu halde LLM tam olarak kapsam dışı cevabını üretirse cevap geçersiz sayılır; fallback soru terimleriyle en çok örtüşen kaynak cümlelerini seçer.
- `/show <chunk_id>` ve `local-rag show <chunk_id>` tam kaynak metnini metadata ile gösterir.
- `/filter <kaynak>` interaktif oturum filtresini yönetir; `/ask --source` ve `local-rag ask --source` tek soruluk filtre sağlar. SQL filtreleri parametrelidir.
- `local-rag` çalışma dizininden bağımsızdır. `--project` en yüksek, `LOCAL_RAG_HOME` ikinci, kurulu repository kökü varsayılan önceliktedir.
- İnteraktif CLI, `prompt-toolkit` ile çerçeveli giriş kullanır. `/` yazıldığında açıklamalı komut menüsü girişin üstünde açılır; komut ve argüman ayrı renklendirilir, parametre isteyen komutlarda kullanım ipucu gösterilir. Alt durum satırı model, kaynak sayısı, indeks ve filtreyi prompt başına bir kez yeniler. Ok tuşları menü/geçmiş gezinmesini, Tab bağlama duyarlı tamamlamayı, `Esc` menü kapatmayı ve `Ctrl+L` ekran temizlemeyi yönetir. Girişte `Ctrl+C` dolu satırı temizler, boş satırda oturumu kapatır. Geçmiş seçilen projenin `data/cli_history` dosyasında düz metin ve `0600` izniyle tutulur. TTY olmayan kullanım sade input/readline fallback'ine döner.
- `/history` yalnızca o süreçte başarıyla tamamlanan yapılandırılmış cevapları listeler; `/repeat [id]` orijinal kaynak filtresini korur. `/export markdown|json [yol]` kayıtları varsayılan olarak `data/exports/` altına yazar, mevcut dosyanın üzerine yazmaz ve tam chunk metnini export etmez.
- `LOCAL_RAG_MODEL` aktif chat modelini kod değiştirmeden seçer; boş veya tanımsızsa varsayılan `phi-4-mini` kullanılır.
- Gerçek benchmark'ta `phi-4-mini` 3/3 geçerli cevap ve %89 terim kapsamı; `phi-3.5-mini` 2/3 geçerli cevap ve %56 kapsam verdi. Varsayılan model bu nedenle `phi-4-mini` olarak korundu.
- LLM kalite kontrolü aşırı kelime/üçlü ifade tekrarını reddeder; böyle cevaplar normal RAG akışında kaynak fallback'ine yönelir.
- Foundry servis durumunda 15 saniye, HTTP/model çağrılarında 120 saniye timeout vardır; takılı alt süreç sonsuza kadar beklemez.
- Embedding modeli yerel snapshot mevcutsa ağ kontrolü yapmadan yüklenir.
- `RAG nedir?` ve `Embedding nedir?` soruları `example.txt` kaynağını buluyor.
- `Veri madenciliği süreçleri nedir?` sorusu `datamining.pdf` kaynağını buluyor.
- `Hava nasıl?` sorusu threshold altında kalıyor ve kapsam dışı kabul ediliyor.
- Token-aware reindex sonrasında veri madenciliği retrieval skoru yaklaşık `0.6174`; kimlik avı `0.6038`; çok faktörlü doğrulama `0.7048`; 3-2-1 yedekleme `0.7051` oldu. Skorlar tek başına hedef değildir; eval doğru kaynak ve seçilmiş context kavramlarını birlikte doğrular.
- `phi-4-mini` ile yapılan gerçek generative test doğru ve kaynakla uyumlu cevap verdi. İlk model yüklemeli generation yaklaşık 39 saniye sürdü; bu beklenen bir cold-start davranışıdır.
- Yeni chunking, daha önce cümle ortasında başlayan fallback chunkını tam cümle başlangıcına taşıdı.
- Reindex artık hazırlama hatasında eski indekse dokunmuyor; SQLite yazma hatasında rollback yapıyor.

Çalışma ağacında commit edilmemiş kullanıcı/agent değişiklikleri bulunabilir. Her işe başlarken `git status` ve ilgili diff'i oku. Kullanıcının mevcut değişikliklerini geri alma veya ezme.

## 8. Kurulum ve Çalıştırma

```bash
cd /Users/erdemac/Developer/local-rag-assistant
source .venv/bin/activate
pip install -e .
local-rag
```

`python main.py` geriye dönük olarak aynı interaktif oturumu açmaya devam eder.

Terminal alt komutları:

```text
local-rag ask "RAG nedir?"
local-rag ask --source example.txt "RAG nedir?"
local-rag add "/dosya/yolu/notlar.pdf"
local-rag remove "notlar.pdf"
local-rag remove "notlar.pdf" --yes
local-rag benchmark --models phi-4-mini phi-3.5-mini
local-rag reindex
local-rag stats
local-rag sources
local-rag show 156
local-rag doctor
local-rag model
local-rag config
local-rag --help
```

Repository dışından varsayılan projeyi kullanmak için doğrudan `local-rag`
çalıştırılabilir. Farklı çalışma kökü için global seçenek alt komuttan önce gelir:

```bash
local-rag --project /dosya/yolu/rag-calismasi stats
LOCAL_RAG_HOME=/dosya/yolu/rag-calismasi local-rag
```

CLI içindeki temel komutlar:

```text
/help
/stats
/model
/config
/sources
/show <chunk-id>
/filter <dosya-adı|off>
/ask [--source <dosya-adı>] <soru>
/history
/repeat [kayıt-id]
/export <markdown|json> [dosya-yolu]
/doctor
/add <dosya-yolu>
/remove <dosya-adı>
/benchmark [model ...]
/reindex
/debug on
/debug off
/exit
```

Foundry Local modelinin cache'te bulunması gerekir. Güncel varsayılan model `phi-4-mini`dir.

## 9. Test ve Doğrulama

Her Python değişikliğinden sonra en az:

```bash
python -m py_compile main.py eval.py app/*.py tests/*.py
python -m unittest discover -s tests -v
python eval.py
```

Hugging Face ağı kapalıysa ve model daha önce cache'e indirilmişse:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python eval.py
```

Retrieval'a dokunan değişikliklerde metrik farkını göster:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python eval.py --compare
```

Baseline'ı yalnızca değişikliğin kasıtlı bir iyileşme olduğu doğrulandıktan
sonra güncelle. Gerilemeyi baseline'ı ezerek gizleme:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python eval.py --update-baseline
```

Ingestion, chunking veya embedding değişikliğinde:

1. Unit testleri çalıştır.
2. `/reindex` çalıştır.
3. `python eval.py` ile retrieval skorlarını kontrol et.
4. Chunk sayısındaki veya skorlardaki değişimi kullanıcıya bildir.

Prompt, LLM, context seçimi veya fallback değişikliğinde kullanıcıdan gerçek model testi istemek gerekebilir. Kullanıcıya çalıştıracağı komutu ve soruları açıkça ver. Asgari manuel testler:

```text
Veri madenciliği süreçleri nedir?
Veri madenciliğinde veri temizleme ne işe yarar?
Hava nasıl?
```

İlk iki generative soruyu aynı oturumda sorarak cold-start ve warm generation sürelerini ayrı değerlendir.

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
- Eval seti 9 retrieval vakası içerir; yeni doküman ve özelliklerle birlikte büyütülmeye devam edilmelidir.
- Oturum geçmişi listeleme, tekrar ve export içindir; konuşma context'i veya takip sorusu çözümleme amacıyla kullanılmaz ve uygulama kapanınca bellekten silinir.
- Terminal giriş geçmişi yerelde tutulur ancak konuşma context'i amacıyla kullanılmaz.

## 12. Öncelikli Roadmap

Tamamlanan yakın özellikler:

- `/sources`: İndeksteki dosya, tür, sayfa ve chunk sayılarını gösterir; şema güvenli hazırlanır.
- `/doctor`: Sistem bileşenlerini ve model cache'ini inference yapmadan kontrol eder.
- Standart hata mesajları: Hatalar ve uyarılar çözüm önerisiyle gösterilir; oturum korunur.
- Rich terminal görünümü: Sade banner, semantik cevap türleri, hizalı tablolar, Türkçe performans satırı ve tek satırlı retrieval/model/generation ilerlemesi.
- Sessiz Foundry başlangıcı: Normal modda servis logu spinner'a karışmaz; debug modu ham çıktıyı korur.
- `/model` ve `/config`: Model/cache/lazy-load durumu ile aktif RAG ayarlarını değiştirmeden gösterir.
- Kurulabilir CLI: `local-rag`, `ask`, `add`, `remove`, `reindex`, `stats`, `sources`, `doctor`, `model` ve `config` entrypoint'leri ortak uygulama akışını kullanır.
- İndeks güncelliği: SHA-256 manifestiyle eklenen, değişen ve silinen dokümanları algılar; soru akışı, `/stats` ve `/doctor` reindex gerektiğini bildirir.
- Güvenli dosya yönetimi: `/add` ve `/remove` ile doğrulama, üzerine yazma koruması, dizin sınırı ve silme onayı sağlar.
- Genişletilmiş eval: `cybersecurity.txt` ile doğru dosya yanında beklenen chunk kavramlarını da doğrular.
- Model benchmark: Phi modellerini yükleme, cold/warm süre, cevap geçerliliği ve terim kapsamıyla karşılaştırır.
- Model yapılandırması: `LOCAL_RAG_MODEL` ile kod düzenlemeden model seçer; varsayılan `phi-4-mini` kalır.
- Ana README: Kurulumdan ilk soruya kadar kullanım akışını, mimariyi, gerçek benchmark sonuçlarını, testleri ve proje sınırlarını sunar.
- Yapılandırılmış RAG servisi: cevap üretimini Rich terminal sunumundan ayırır.
- Kaynak denetimi: `/show`, `/filter` ve `ask --source` ile chunk inceleme ve dosya bazlı retrieval sağlar.
- Taşınabilir CLI: `--project` ve `LOCAL_RAG_HOME` ile her çalışma dizininden doğru docs/data kökünü kullanır.
- Terminal ergonomisi: çerçeveli giriş, giriş üstünde canlı slash menüsü, komut renklendirme, parametre ipuçları, model/kaynak/indeks/filtre durum satırı, proje bazlı kalıcı ok tuşu geçmişi ve bağlama duyarlı Tab tamamlama sağlar.
- Klavye ve streaming ergonomisi: `Esc`, `Ctrl+L`, bağlamsal `Ctrl+C`, token geldikçe güncellenen cevap ve güvenli generation iptali sağlar.
- Oturum araçları: `/history`, kaynak filtresini koruyan `/repeat` ve üzerine yazma korumalı Markdown/JSON `/export` sağlar.
- Token-aware chunking: embedding modelinin 128 token sınırına uygun 110/20 parçalama, cümle/kelime hizası ve kesilmeyen embedding girdisi sağlar.
- Context hazırlama: relevance ile seçim, belge düzeninde prompt, skorlu komşu genişletme, 5 parça üst sınırı ve kaynak rolü sağlar.

Sıradaki hedefleri şu sırayla ele al:

0. ~~**Eval güçlendirmesi.**~~ **Tamamlandı.** Recall@k ve MRR metrikleri,
   içerik imzasıyla etiketlenmiş ground truth, 6 hard negative vaka, bozuk etiket
   tespiti (`case_labels`) ve `eval_baseline.json` üzerinden `--compare` akışı.
1. **Chunking karşılaştırma deneyi.** 110/20 dışındaki konfigürasyonları aynı
   eval setinde ölç; seçimi ölçümle gerekçelendir. Artık `--compare` ile
   Recall@k/MRR farkı doğrudan görülebilir.
2. **Yanlış pozitif savunması — büyük kısmı tamamlandı.** Kelime kanıtı kapısı
   eklendi; 6 hard negative vakanın 6'sı artık `no_evidence` alıyor ve LLM'e
   gitmiyor. Kalan iki parça:
   - **Groundedness kontrolü.** Kapı retrieval context'ine bakar, üretilen
     cevabın o context'e sadık kalıp kalmadığına bakmaz. Kanıt varken model
     yine de context dışına çıkabilir.
   - **Eşiklerin yeniden kalibrasyonu.** `SIMILARITY_THRESHOLD`,
     `CONTEXT_SCORE_THRESHOLD` ve `EXTRACTIVE_SCORE_THRESHOLD` hâlâ elle
     seçilmiş değerler. Kelime kanıtı kapısı eklendikten sonra bu eşiklerin
     hangi yükü taşıdığı değişti; yeniden ölçülmeli. Hybrid search bunu
     etkilemedi, çünkü birleşik skor kapıda kullanılmıyor.
   - **Kelime kanıtı eşiğinin kırılganlığı.** `TERM_EVIDENCE_THRESHOLD = 0.67`
     20 vakayla ölçüldü ve boşluk `0.21`. Bu üçüncü kalibrasyon; ilk ikisi
     (`0.50`, `0.60`) gerçek kullanımda sızdırdı. Set her büyüdüğünde yeniden
     ölç; `tools/term_evidence_analysis.py` bunu tek komutla yapar.
     Her kalibrasyonu tetikleyen şey ölçüm seti değil gerçek bir kullanıcı
     sorusu oldu; bu, setin küçüklüğünün en somut kanıtıdır.
3. **Hybrid search.** ~~Yapıldı.~~ `app/sparse_search.py` (BM25) +
   `app/retrieval.py` (RRF). Karar: birleşik skor yalnızca sıralamada kullanılır,
   kapı skoru cosine kalır. Sonuç: `Recall@1` 0.60 -> 0.80, `MRR` 0.775 -> 0.90.
   Yan etkisi kelime kanıtı kapısını IDF ağırlıklarına taşımayı zorunlu kıldı;
   ayrıntı bölüm 7'de. SQLite FTS5 tercih edilmedi: `unicode61` tokenizer'ı
   Türkçe stemming yapmaz ve `remove_diacritics` seçenekleri `ı/i`, `ş/s`
   ayrımını bozar; ayrıca `normalize_text()`'ten sapan ikinci bir normalizasyon
   yolu açardı.
4. **Reranking.** Geniş aday havuzunu cross-encoder ile yeniden sırala. Hybrid
   sonrası kalan boşluk: `Recall@1 = 0.80`, yani beş vakadan birinde doğru parça
   hâlâ en üstte değil.
   *Mimari karar: aday sayısı ve kabul edilebilir latency bütçesi.*
5. **Conversation history.** Asıl iş takip sorusu değil, query rewriting; soruyu
   retrieval'a bağımsız (standalone) biçimde ver.
   *Mimari karar: rewriting tasarımı ve geçmiş bütçesi.*

Bu sıranın gerekçesi: 0–2 **ölçme yeteneği** kazandırır, 3–5 ise ancak o yetenek
varsa öğretici ve doğrulanabilir olur. Hybrid search ve reranking'in iddiası
"retrieval kalitesini artırmak"tır; iyi bir eval olmadan bu iddia ölçülemez.

Fırsat buldukça ele alınacaklar (öncelikli değil):

- **Cevap groundedness kontrolü.** `is_valid_answer()` şu an cevabın biçimini
  denetliyor, context'e sadakatini değil. n-gram örtüşmesi gibi basit bir ölçüt
  bile gerçek bir güvenlik katmanı olur. 0. adımın hard negative bulgusundan
  sonra bu madde önceliğini artırdı; 2. adımla birlikte ele alınabilir.
- **Incremental reindex.** Ancak `time local-rag reindex` rahatsız edici hale
  geldiğinde. Mevcut ölçekte (3 dosya, 24 chunk) çözülecek bir problem yok.
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
