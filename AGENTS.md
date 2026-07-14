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
- Benzerlik: NumPy ve scikit-learn cosine similarity
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

- `main.py`: CLI döngüsü, `/model` ve `/config` dahil komutlar, retrieval kararları, cevap modu ve çıktı gösterimi.
- `app/config.py`: Similarity, context, extractive ve cevap kalite eşikleri.
- `app/cli_output.py`: Rich konsolu, banner, tablolar, semantik cevap paneli, spinner, standart hata/uyarı ve Türkçe performans çıktısı.
- `app/database.py`: SQLite şeması, okuma/yazma, istatistikler, `get_indexed_sources()` ve atomik `replace_chunks()` işlemi.
- `app/ingest.py`: TXT/PDF okuma, chunking, embedding hazırlama ve reindex akışı.
- `app/embeddings.py`: Yerel Hugging Face snapshot'ını tercih eden embedding lazy-load/cache yönetimi.
- `app/health.py`: `/doctor` için doküman, veritabanı, embedding ve Foundry/model cache sağlık kontrolleri.
- `app/retrieval.py`: Soru embeddingi, cosine similarity, geçersiz vektör kontrolü ve sıralama.
- `app/prompts.py`: Türkçe, yalnızca context'e dayalı RAG promptu.
- `app/llm.py`: Sessiz/debug Foundry Local başlangıcı, `phi-4-mini`, cevap temizleme ve kalite doğrulaması.
- `eval.py`: İndeks sağlığı, cevap kalite kararı ve retrieval regression değerlendirmesi.
- `eval_cases.json`: Deterministik retrieval test soruları ve beklentileri.
- `tests/test_ingest.py`: Chunk sınırı, atomik reindex, `/sources` şema güvenliği ve CLI çıktı testleri.
- `tests/test_health.py`: `/doctor` başarı, uyarı, hata ve CLI çıktı testleri.
- `tests/test_cli_output.py`: Standart hata gösterimi, `/model`, `/config`, lazy-load ve CLI oturum dayanıklılığı testleri.
- `tests/test_embeddings.py`: Yerel embedding snapshot tercihi ve cache-miss fallback testleri.
- `tests/test_llm.py`: Parça etiketi temizliği ile sessiz/debug Foundry servis başlangıcı testleri.
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

- 2 kaynak dosya ve 11 chunk bulunuyor.
- Retrieval ve indeks değerlendirmesi `6/6` başarılı.
- Unit testler `38/38` başarılı (chunking, atomik reindex, CLI komutları, Rich çıktıları, model yükleme ve LLM cevap temizliği).
- `/sources` indeksteki dosya, tür, sayfa ve chunk sayılarını gösterir; boş indeks, eksik `chunks` tablosu ve eski şema senaryolarında çökmez.
- `/doctor` dokümanları, veritabanını, 384 boyutlu embeddingleri, Foundry kurulumunu ve `phi-4-mini` cache dosyalarını model yüklemeden kontrol eder.
- CLI hataları kullanıcı mesajı ve çözüm önerisi gösterir; teknik exception yalnızca debug modunda görünür.
- CLI Rich tabanlı panel, tablo, renk ve TTY spinner'ları kullanır; piped/renksiz çıktıda okunabilir kalır.
- Cevap başlığı modun Türkçe adını ve retrieval skorunu gösterir. Üretken cevap cyan, doğrudan kaynak cevabı yeşil, fallback sarı, kanıt bulunamaması düşük vurgulu gri gösterilir.
- Performans satırı kullanıcıya `Arama`, `Yanıt` ve `Toplam` adlarıyla gösterilir; cevap, kaynak ve süre blokları aynı sol hizayı kullanır.
- LLM'in üretebildiği `[Parça 1]`, `(Parça 1)` ve aralık/listeli parça atıfları cevap metninden temizlenir; kaynak bilgisi yalnızca ayrı kaynak tablosunda gösterilir.
- Foundry servisi normal modda terminale ham başlangıç logu yazmadan başlatılır; `/debug on` açıkken SDK çıktısı korunur.
- `/model` chat/embedding modeli, cache ve oturumdaki lazy-load durumunu inference yapmadan gösterir.
- `/config` aktif eşikleri, cevap kalite ayarlarını, chunk değerlerini ve yolları salt okunur gösterir.
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
python main.py
```

CLI içindeki temel komutlar:

```text
/help
/stats
/model
/config
/sources
/doctor
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
- Eval seti küçüktür; yeni doküman ve özelliklerle birlikte genişletilmelidir.
- Konuşma geçmişi ve takip sorusu çözümleme henüz yoktur.
- CLI henüz kurulan bir `local-rag` terminal komutu olarak paketlenmemiştir.

## 12. Öncelikli Roadmap

Tamamlanan yakın özellikler:

- `/sources`: İndeksteki dosya, tür, sayfa ve chunk sayılarını gösterir; şema güvenli hazırlanır.
- `/doctor`: Sistem bileşenlerini ve model cache'ini inference yapmadan kontrol eder.
- Standart hata mesajları: Hatalar ve uyarılar çözüm önerisiyle gösterilir; oturum korunur.
- Rich terminal görünümü: Sade banner, semantik cevap türleri, hizalı tablolar, Türkçe performans satırı ve işlem spinner'ları.
- Sessiz Foundry başlangıcı: Normal modda servis logu spinner'a karışmaz; debug modu ham çıktıyı korur.
- `/model` ve `/config`: Model/cache/lazy-load durumu ile aktif RAG ayarlarını değiştirmeden gösterir.

Yakın hedefleri şu sırayla ele al:

1. Uygulamayı `local-rag`, `local-rag ask`, `local-rag reindex`, `local-rag stats` komutlarıyla çalışacak şekilde paketle.
2. Doküman değişikliklerini algılayıp reindex gerektiğini bildir.
3. Güvenli `/add` ve onaylı `/remove` dosya yönetimi ekle.
4. Phi modellerini süre ve Türkçe cevap kalitesiyle karşılaştıran benchmark akışı oluştur.

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
