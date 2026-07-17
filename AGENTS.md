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
- Kullanıcı arayüzü: Şimdilik interaktif terminal CLI

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

- `main.py`: İnteraktif ve argparse tabanlı terminal entrypoint'i, ortak `answer_question()` akışı, komutlar, retrieval kararları ve çıktı gösterimi.
- `pyproject.toml`: `local-rag` console script'i, paket sürümü ve doğrudan Python bağımlılıkları.
- `app/__init__.py`: Paket sürümünü (`0.1.0`) tutar.
- `app/config.py`: Similarity, context, extractive ve cevap kalite eşikleri.
- `app/cli_output.py`: Rich konsolu, banner, tablolar, semantik cevap paneli, spinner, standart hata/uyarı ve Türkçe performans çıktısı.
- `app/benchmark.py`: Sabit RAG contextleriyle model yükleme, cold/warm generation, cevap geçerliliği ve terim kapsamı benchmark'ı.
- `app/database.py`: SQLite şeması, chunk/manifest okuma-yazma, istatistikler ve atomik `replace_chunks()` işlemi.
- `app/document_manager.py`: TXT/PDF doğrulama, üzerine yazmadan güvenli kopyalama ve `docs/` sınırında dosya silme işlemleri.
- `app/index_state.py`: Doküman SHA-256 manifestini üretir; eklenen, değişen ve silinen kaynakları indeksle karşılaştırır.
- `app/ingest.py`: TXT/PDF okuma, chunking, embedding hazırlama, doküman değişim koruması ve reindex akışı.
- `app/embeddings.py`: Yerel Hugging Face snapshot'ını tercih eden embedding lazy-load/cache yönetimi.
- `app/health.py`: `/doctor` için doküman, veritabanı, embedding ve Foundry/model cache sağlık kontrolleri.
- `app/retrieval.py`: Soru embeddingi, normalize edilmiş cosine hesabı, geçersiz vektör kontrolü ve sıralama.
- `app/prompts.py`: Türkçe, yalnızca context'e dayalı RAG promptu.
- `app/llm.py`: Süre sınırlı Foundry başlangıcı, `LOCAL_RAG_MODEL`, cevap temizleme ve tekrar döngüsü dahil kalite doğrulaması.
- `benchmark_cases.json`: Modellerin aynı context ve beklenen kavramlarla karşılaştırıldığı üretken cevap vakaları.
- `eval.py`: İndeks sağlığı, cevap kalite kararı ve retrieval regression değerlendirmesi.
- `eval_cases.json`: Deterministik retrieval test soruları ve beklentileri.
- `tests/test_benchmark.py`: Benchmark hazırlama, cold/warm ölçüm, kalite ve hata raporu testleri.
- `tests/test_eval.py`: Kaynak, skor ve beklenen chunk kavramı değerlendirme testleri.
- `tests/test_ingest.py`: Chunk sınırı, atomik reindex, `/sources` şema güvenliği ve CLI çıktı testleri.
- `tests/test_health.py`: `/doctor` başarı, uyarı, hata ve CLI çıktı testleri.
- `tests/test_document_manager.py`: Add/remove doğrulama, güvenlik, onay, CLI ve indeks güncelliği entegrasyon testleri.
- `tests/test_index_state.py`: Güncel, eski, manifestsiz ve eksik indeks senaryolarını test eder.
- `tests/test_cli_output.py`: Standart hata gösterimi, `/model`, `/config`, lazy-load ve CLI oturum dayanıklılığı testleri.
- `tests/test_embeddings.py`: Yerel embedding snapshot tercihi ve cache-miss fallback testleri.
- `tests/test_entrypoint.py`: Paket metadata'sı, `local-rag` alt komutları, exit code ve ortak soru akışı testleri.
- `tests/test_llm.py`: Parça etiketi temizliği ile sessiz/debug Foundry servis başlangıcı testleri.
- `tests/test_retrieval.py`: Normalize edilmiş cosine skorunun sıralama ve sonlu değer testleri.
- `README.md`: Kurulum, kullanım, mimari, benchmark, test sonuçları, sınırlamalar ve V2 yol haritasını özetleyen ana proje sunumu.
- `PROJECT_GUIDE.md`: Projenin uzun, öğretici teknik anlatımı ve roadmap'i.
- `INSTRUCTIONS.md`: İlk proje hedefleri; güncel gerçeklik için her zaman kodu ve bu dosyayı esas al.
- `docs/`: İndekslenecek kullanıcı dokümanları.
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
TOP_K = 3

USE_EXTRACTIVE_FALLBACK = True
EXTRACTIVE_SCORE_THRESHOLD = 0.50
MAX_EXTRACTIVE_CHARS = 500
MIN_GENERATIVE_ANSWER_CHARS = 30

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
```

Bu değerler mevcut küçük veri seti ve regression testlerine göre seçildi. Değiştirilecekse önce gerekçeyi açıkla, eval vakası ekle ve eski/yeni sonucu karşılaştır.

## 7. Güncel Durum

Son doğrulanan durumda:

- 3 kaynak dosya ve 16 chunk bulunuyor. Yeni `cybersecurity.txt` beş ayrı güvenlik konusu içeriyor.
- Retrieval ve indeks değerlendirmesi `11/11` başarılı.
- Unit testler `81/81` başarılı (benchmark, eval, retrieval, chunking, atomik reindex/manifest, indeks güncelliği, güvenli TXT/PDF yönetimi, CLI, model yapılandırması ve LLM cevap temizliği).
- `/sources` indeksteki dosya, tür, sayfa ve chunk sayılarını gösterir; boş indeks, eksik `chunks` tablosu ve eski şema senaryolarında çökmez.
- `/doctor` dokümanları, indeks güncelliğini, veritabanını, 384 boyutlu embeddingleri, Foundry kurulumunu ve `phi-4-mini` cache dosyalarını model yüklemeden kontrol eder.
- CLI hataları kullanıcı mesajı ve çözüm önerisi gösterir; teknik exception yalnızca debug modunda görünür.
- CLI Rich tabanlı panel, tablo, renk ve TTY spinner'ları kullanır; piped/renksiz çıktıda okunabilir kalır.
- Cevap başlığı modun Türkçe adını ve retrieval skorunu gösterir. Üretken cevap cyan, doğrudan kaynak cevabı yeşil, fallback sarı, kanıt bulunamaması düşük vurgulu gri gösterilir.
- Performans satırı kullanıcıya `Arama`, `Yanıt` ve `Toplam` adlarıyla gösterilir; cevap, kaynak ve süre blokları aynı sol hizayı kullanır.
- LLM'in üretebildiği `[Parça 1]`, `(Parça 1)` ve aralık/listeli parça atıfları cevap metninden temizlenir; kaynak bilgisi yalnızca ayrı kaynak tablosunda gösterilir.
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
- `LOCAL_RAG_MODEL` aktif chat modelini kod değiştirmeden seçer; boş veya tanımsızsa varsayılan `phi-4-mini` kullanılır.
- Gerçek benchmark'ta `phi-4-mini` 3/3 geçerli cevap ve %89 terim kapsamı; `phi-3.5-mini` 2/3 geçerli cevap ve %56 kapsam verdi. Varsayılan model bu nedenle `phi-4-mini` olarak korundu.
- LLM kalite kontrolü aşırı kelime/üçlü ifade tekrarını reddeder; böyle cevaplar normal RAG akışında kaynak fallback'ine yönelir.
- Foundry servis durumunda 15 saniye, HTTP/model çağrılarında 120 saniye timeout vardır; takılı alt süreç sonsuza kadar beklemez.
- Embedding modeli yerel snapshot mevcutsa ağ kontrolü yapmadan yüklenir.
- `RAG nedir?` ve `Embedding nedir?` soruları `example.txt` kaynağını buluyor.
- `Veri madenciliği süreçleri nedir?` sorusu `datamining.pdf` kaynağını buluyor.
- `Hava nasıl?` sorusu threshold altında kalıyor ve kapsam dışı kabul ediliyor.
- Yeni chunking sonrasında veri madenciliği retrieval skoru son testte yaklaşık `0.6342` oldu.
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
local-rag add "/dosya/yolu/notlar.pdf"
local-rag remove "notlar.pdf"
local-rag remove "notlar.pdf" --yes
local-rag benchmark --models phi-4-mini phi-3.5-mini
local-rag reindex
local-rag stats
local-rag sources
local-rag doctor
local-rag model
local-rag config
local-rag --help
```

CLI içindeki temel komutlar:

```text
/help
/stats
/model
/config
/sources
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
- Konuşma geçmişi ve takip sorusu çözümleme henüz yoktur.
- `docs/` ve `data/` yolları mevcut çalışma dizinine göredir; `local-rag` şimdilik repository kökünde çalıştırılmalıdır.

## 12. Öncelikli Roadmap

Tamamlanan yakın özellikler:

- `/sources`: İndeksteki dosya, tür, sayfa ve chunk sayılarını gösterir; şema güvenli hazırlanır.
- `/doctor`: Sistem bileşenlerini ve model cache'ini inference yapmadan kontrol eder.
- Standart hata mesajları: Hatalar ve uyarılar çözüm önerisiyle gösterilir; oturum korunur.
- Rich terminal görünümü: Sade banner, semantik cevap türleri, hizalı tablolar, Türkçe performans satırı ve işlem spinner'ları.
- Sessiz Foundry başlangıcı: Normal modda servis logu spinner'a karışmaz; debug modu ham çıktıyı korur.
- `/model` ve `/config`: Model/cache/lazy-load durumu ile aktif RAG ayarlarını değiştirmeden gösterir.
- Kurulabilir CLI: `local-rag`, `ask`, `add`, `remove`, `reindex`, `stats`, `sources`, `doctor`, `model` ve `config` entrypoint'leri ortak uygulama akışını kullanır.
- İndeks güncelliği: SHA-256 manifestiyle eklenen, değişen ve silinen dokümanları algılar; soru akışı, `/stats` ve `/doctor` reindex gerektiğini bildirir.
- Güvenli dosya yönetimi: `/add` ve `/remove` ile doğrulama, üzerine yazma koruması, dizin sınırı ve silme onayı sağlar.
- Genişletilmiş eval: `cybersecurity.txt` ile doğru dosya yanında beklenen chunk kavramlarını da doğrular.
- Model benchmark: Phi modellerini yükleme, cold/warm süre, cevap geçerliliği ve terim kapsamıyla karşılaştırır.
- Model yapılandırması: `LOCAL_RAG_MODEL` ile kod düzenlemeden model seçer; varsayılan `phi-4-mini` kalır.
- Ana README: Kurulumdan ilk soruya kadar kullanım akışını, mimariyi, gerçek benchmark sonuçlarını, testleri ve proje sınırlarını sunar.

V1 hedefleri tamamlandı. Sonraki kararı şu sırayla ele al:

1. V2 arayüzü olarak FastAPI veya Streamlit yönünü seç.
2. Seçilen arayüzün ilk küçük dikey dilimini mevcut ortak uygulama akışını yeniden kullanarak geliştir.

Daha sonraki V2 seçenekleri:

- Streamlit kullanıcı arayüzü
- FastAPI `ask`, `reindex` ve `stats` endpointleri
- OCR desteği
- Kaynak filtresi ve `/show <chunk_id>`
- Conversation history ve takip soruları
- Neighbor chunk genişletme veya reranking
- Daha büyük koleksiyonlar için vector database değerlendirmesi

## 13. Bir Görevi Tamamlama Kriteri

Bir değişikliği tamamlanmış saymadan önce:

1. İlgili kodu ve mevcut kullanıcı değişikliklerini okumuş ol.
2. Davranışı mümkünse deterministik testle kapsa.
3. Unit test, eval ve sözdizimi kontrolünü çalıştır.
4. Gerekliyse kullanıcıya exact manuel model test komutunu ver.
5. Türkçe UX, kaynak doğruluğu ve fallback davranışını koruduğunu doğrula.
6. Ne değiştiğini, neden değiştiğini ve test sonucunu kısa biçimde kullanıcıya açıkla.
7. Çalıştıramadığın bir doğrulama varsa bunu açıkça belirt.
