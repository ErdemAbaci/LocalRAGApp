# Local RAG Assistant - Proje Rehberi

Bu belge, projeye birkaç gün veya birkaç ay ara verdikten sonra geri döndüğümüzde "hangi dosya ne yapıyordu?", "sistem nasıl çalışıyordu?" ve "sırada ne vardı?" sorularına hızlıca cevap vermek için hazırlandı.

Belgenin anlattığı durum: **10 Temmuz 2026**.

## 1. Projenin amacı

Local RAG Assistant, kullanıcının kendi bilgisayarındaki TXT ve PDF dokümanlardan bilgi bulan ve bu bilgiye dayanarak Türkçe cevap üreten yerel bir soru-cevap uygulamasıdır.

Bu proje bir model eğitimi veya fine-tuning projesi değildir. Modelin ağırlıkları değiştirilmez. Dokümanlar ayrı bir bilgi kaynağı olarak işlenir ve kullanıcı soru sorduğunda ilgili parçalar bulunarak cevap üretiminde kullanılır.

RAG, "Retrieval-Augmented Generation" ifadesinin kısaltmasıdır. Bu projede iki ana sistem birlikte çalışır:

1. **Retrieval:** Soruyla ilgili doküman parçalarını bulur.
2. **Generation:** Bulunan parçaları context olarak kullanıp cevap üretir.

Sistem gerektiğinde LLM kullanmadan doğrudan güçlü bir kaynak parçasını da cevap olarak gösterebilir. Bu yaklaşım küçük local modellerin Türkçe üretim hatalarını ve hallucination riskini azaltır.

## 2. Sistemin genel akışı

Dokümanların hazırlanması:

```text
docs/ içindeki TXT ve PDF dosyaları
        ↓
Metin çıkarma
        ↓
Chunk oluşturma (800 karakter, 100 karakter overlap)
        ↓
Her chunk için embedding üretme
        ↓
Chunk + metadata + embedding bilgilerini SQLite'a kaydetme
```

Soru cevaplama:

```text
Kullanıcı sorusu
        ↓
Soru embedding'i
        ↓
SQLite'taki chunk embedding'leriyle cosine similarity
        ↓
En alakalı 3 chunk
        ↓
Eşik ve cevap modu kararı
        ├── Alakasız soru → "Bu bilgi verilen dokümanlarda yok."
        ├── Tek güçlü/kısa chunk → extractive cevap
        └── Birden fazla chunk → local LLM ile generative cevap
                                   ↓
                         Cevap başarısızsa fallback_extractive
```

## 3. Proje klasör yapısı

```text
local-rag-assistant/
├── app/
│   ├── config.py
│   ├── database.py
│   ├── embeddings.py
│   ├── ingest.py
│   ├── llm.py
│   ├── prompts.py
│   └── retrieval.py
├── data/
│   └── rag.db                 # Üretilen yerel veritabanı, Git'e eklenmez
├── docs/
│   ├── example.txt
│   └── datamining.pdf
├── eval.py
├── eval_cases.json
├── embedding_test.py
├── foundry_test.py
├── main.py
├── requirements.txt
└── PROJECT_GUIDE.md
```

## 4. Dosyalar ne yapıyor?

### `main.py`

Uygulamanın ana giriş noktasıdır. Terminal arayüzünü ve bütün RAG karar akışını yönetir.

Başlıca sorumlulukları:

- Açılış banner'ını ve `rag>` prompt'unu gösterir.
- `/help`, `/stats`, `/sources`, `/doctor`, `/reindex`, `/debug on`, `/debug off` ve `/exit` komutlarını işler.
- Kullanıcı sorusu için retrieval çalıştırır.
- En iyi similarity skorunu kontrol eder.
- Context'e girecek chunkları filtreler.
- Extractive veya generative cevap arasında karar verir.
- LLM cevabı başarısızsa en iyi chunk ile fallback yapar.
- Kaynakları, skorları ve süreleri ekrana yazdırır.

LLM uygulama açılır açılmaz yüklenmez. `get_llm()` fonksiyonu sayesinde yalnızca ilk generative cevap gerektiğinde yüklenir ve aynı oturumda tekrar kullanılır. Buna lazy loading denir.

### `app/config.py`

Uygulamanın davranış ayarlarını tek yerde tutar.

Mevcut ayarlar:

```python
SIMILARITY_THRESHOLD = 0.20
CONTEXT_SCORE_THRESHOLD = 0.35
TOP_K = 3

USE_EXTRACTIVE_FALLBACK = True
EXTRACTIVE_SCORE_THRESHOLD = 0.50
MAX_EXTRACTIVE_CHARS = 500

MIN_GENERATIVE_ANSWER_CHARS = 30
```

Anlamları:

- `SIMILARITY_THRESHOLD`: En iyi skor bunun altındaysa soru dokümanla alakasız kabul edilir.
- `CONTEXT_SCORE_THRESHOLD`: LLM'e yalnızca bu skoru geçen chunklar gönderilir.
- `TOP_K`: Retrieval aşamasında en iyi kaç chunk'ın alınacağını belirler.
- `USE_EXTRACTIVE_FALLBACK`: Tek güçlü chunk'ın LLM kullanılmadan cevap olmasına izin verir.
- `EXTRACTIVE_SCORE_THRESHOLD`: Extractive cevap için gereken minimum skor.
- `MAX_EXTRACTIVE_CHARS`: Çok uzun chunkların doğrudan cevap olarak dönmesini engeller.
- `MIN_GENERATIVE_ANSWER_CHARS`: Bundan kısa LLM cevapları başarısız kabul edilir.

Bu değerler rastgele seçilmemiştir; mevcut küçük eval seti ve manuel testlerle başlangıç değerleri olarak belirlenmiştir. Doküman sayısı büyüdüğünde yeniden kalibre edilebilir.

### `app/cli_output.py`

Rich tabanlı terminal sunumunu tek merkezden yönetir:

- Sade açılış paneli
- `/help`, `/stats`, `/sources` ve `/doctor` tabloları
- Cevap başlığında Türkçe cevap modu ve en iyi retrieval skoru
- Kaynak tablosu ve kompakt performans satırı
- Reindex, retrieval ve generation spinner'ları
- Standart hata, uyarı, başarı ve bilgi mesajları

Cevap renkleri dekorasyon için değil, durum bilgisini hızlı okutmak için kullanılır:

| İç mod | Kullanıcı etiketi | Renk | Anlam |
| --- | --- | --- | --- |
| `generative` | Üretken | Cyan | Yerel LLM birden fazla kaynağı sentezledi. |
| `extractive` | Doğrudan | Yeşil | Güçlü ve kısa kaynak metni doğrudan kullanıldı. |
| `fallback_extractive` | Kaynak metni | Sarı | LLM yerine güvenli kaynak metnine dönüldü. |
| `no_evidence` | Kanıt bulunamadı | Gri | Soru için yeterli doküman kanıtı bulunamadı. |

Teknik mod adları normal kullanıcı görünümünde gösterilmez. Panel başlığı `Cevap · Üretken · Skor 0.6342`, süre satırı ise `Arama · Yanıt · Toplam` biçimindedir. Cevap paneli, kaynak tablosu ve süreler aynı sol hizayı kullanır.

Normal modda kullanıcı yalnızca anlaşılır mesajı ve çözüm önerisini görür:

```text
HATA  Dokümanlarda arama yapılamadı.
      Çözüm  /doctor çalıştır; indeks sorunu varsa /reindex ile yenile.
```

`/debug on` açıkken bunlara exception türü ve teknik ayrıntı eklenir. Reindex, retrieval, LLM fallback, `/stats`, `/sources` ve `/doctor` hata yolları bu ortak gösterimi kullanır. Böylece hata, retrieval veya komut sırasında oluşsa bile CLI oturumu mümkün olduğunca açık kalır.

### `app/database.py`

SQLite veritabanı işlemlerinden sorumludur. Veritabanı yolu `data/rag.db` şeklindedir.

`chunks` tablosunda şu bilgiler saklanır:

| Kolon | Açıklama |
| --- | --- |
| `id` | Otomatik artan chunk kimliği |
| `source_name` | Kaynak dosyanın adı |
| `source_type` | `txt` veya `pdf` |
| `page_number` | PDF sayfa numarası; TXT için `None` |
| `chunk_index` | Kaynak/sayfa içindeki chunk sırası |
| `chunk_text` | Chunk'ın gerçek metni |
| `embedding` | JSON string olarak saklanan embedding vektörü |

Önemli fonksiyonlar:

- `init_db()`: Tabloyu oluşturur ve eksik metadata kolonlarını ekler.
- `insert_chunk()`: Bir chunk ve metadata'sını kaydeder.
- `replace_chunks()`: Eski indeksi silme ve yeni chunkları ekleme işlemini tek transaction içinde yapar. Ekleme başarısız olursa rollback ile eski indeks korunur.
- `get_all_chunks()`: Retrieval için bütün chunkları okur.
- `get_chunk_stats()`: `/stats` komutuna chunk ve kaynak sayısını verir.
- `get_indexed_sources()`: `/sources` komutuna dosya bazında tür, sayfa ve chunk özetini verir. Veritabanı dosyası varsa sorgu öncesi şemayı güvenli şekilde hazırlar.

`ensure_chunk_metadata_columns()` küçük bir migration görevi görür. Eski `rag.db` dosyasında yeni kolonlar yoksa tabloyu silmeden kolonları ekler.

### `app/health.py`

`/doctor` komutunun sağlık kontrollerini terminal gösteriminden bağımsız olarak yürütür:

- `docs/` klasörü ile TXT/PDF varlığını kontrol eder.
- SQLite indeksinin okunabildiğini, kaynak ve chunk sayılarını doğrular.
- Embeddinglerin 384 boyutlu ve sonlu sayılardan oluştuğunu denetler.
- `foundry` terminal aracının ve model cache dizininin varlığını kontrol eder.
- `phi-4-mini` model dosyalarının cache içinde gerçekten bulunduğunu doğrular.

Bu kontrol model yüklemez, model indirmez ve inference yapmaz. Her sonuç `ok`, `warning` veya `error` durumuyla birlikte gerektiğinde çözüm önerisi taşır.

### `app/ingest.py`

Dokümanları RAG sisteminin arayabileceği hale getirir. Bu işleme ingestion denir.

Akış:

1. `docs/` içindeki `.txt` ve `.pdf` dosyalarını bulur.
2. TXT dosyasını UTF-8 metin olarak okur.
3. PDF dosyasını `pypdf.PdfReader` ile sayfa sayfa okur.
4. Her sayfa/paragraf metnini chunklara böler.
5. Chunkları toplu olarak embedding'e çevirir.
6. Bütün yeni indeks kayıtlarını bellekte hazırlar.
7. Hazır kayıtları tek transaction ile SQLite'a yazar.

Chunk ayarları:

```python
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
```

Overlap, iki ardışık chunk arasında bir miktar ortak metin bırakır. Chunk sınırlarında önce cümle sonu tercih edilir. Overlap başlangıcı cümle sınırına hizalanamıyorsa kelime sınırı kullanılır; önceki chunk tam cümlede bittiyse yeni chunk bir sonraki tam cümleden başlatılır.

`read_pdf_file()` her PDF sayfasını ayrı bir document kaydı olarak üretir. Bu sayede cevap kaynaklarında `page=2` gibi sayfa bilgisi gösterilebilir.

Metni başarıyla çıkarılan bozuk PDF'lerdeki bilinen `Ignoring wrong pointing object` mesajları kullanıcı terminalini kirletmemesi için filtrelenir. Gerçek okuma hataları exception olarak görünmeye devam eder.

`/reindex` komutu doğrudan `ingest_documents()` fonksiyonunu çağırır. Doküman okuma veya embedding üretme başarısız olursa veritabanına dokunulmaz. SQLite yazımı sırasında hata oluşursa transaction geri alınır ve önceki indeks kullanılmaya devam eder.

### `app/embeddings.py`

Embedding modelini tek merkezden yönetir.

Kullanılan model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Bu model Türkçe dahil çok dilli metinleri 384 boyutlu sayısal vektörlere dönüştürür.

Önemli fonksiyonlar:

- `get_local_model_path()`: Hugging Face cache'indeki yerel snapshot yolunu bulur.
- `get_embedding_model()`: Önce yerel snapshot'ı yükler; cache yoksa model kimliğiyle normal indirme yoluna döner ve instance'ı bellekte tutar.
- `embed_text()`: Tek bir metni embedding'e çevirir.
- `embed_texts()`: Birden fazla metni batch halinde embedding'e çevirir.

Ingestion ve retrieval aynı model instance'ını kullanır. Yerel snapshot tercihi, model daha önce indirilmişken gereksiz ağ kontrolünü ve retry loglarını engeller. İlk çağrı yine modelin belleğe alınması nedeniyle sonraki çağrılardan yavaş olabilir.

### `app/retrieval.py`

Kullanıcı sorusuna en yakın chunkları bulur.

Akış:

1. SQLite'tan bütün chunkları alır.
2. Sorunun embedding'ini üretir.
3. Chunk embeddinglerini NumPy `float32` matrisine dönüştürür.
4. Bozuk, `NaN` veya sonsuz embeddingleri filtreler.
5. `cosine_similarity` ile benzerlik skorlarını hesaplar.
6. Sonuçları yüksek skordan düşük skora sıralar.
7. En iyi `TOP_K` sonucu döndürür.

Retrieval yalnızca ilgili metni bulur; cevap üretmez.

### `app/prompts.py`

Local LLM'e gönderilen system ve user mesajlarını hazırlar.

Prompt'un temel kuralları:

- Yalnızca verilen bağlamı kullan.
- Bağlam dışı bilgi ekleme.
- Sade ve doğal Türkçe yaz.
- Süreç sorularında kısa maddeler kullan.
- Kaynak adı, skor ve chunk numarası yazma.
- Bağlam yetersizse yalnızca `Bu bilgi verilen dokümanlarda yok.` de.

Retrieval sonucu gelen chunklar `[Parça 1]`, `[Parça 2]` şeklinde ayrılarak modele gönderilir. Bu etiketler yalnızca modelin context'i ayırt etmesi içindir; kullanıcıya gösterilmemelidir.

### `app/llm.py`

Microsoft Foundry Local ile local chat modelini çalıştırır.

Mevcut varsayılan model:

```python
MODEL_ALIAS = "phi-4-mini"
```

`LocalLLM` sınıfı:

1. `FoundryLocalManager` oluşturur.
2. Seçilen modeli yerel cache'ten yükler.
3. Foundry Local'ın OpenAI-compatible endpoint'ine bağlanan client'ı oluşturur.
4. Chat completion çağrısı yapar.
5. Ham cevabı `clean_answer()` ile temizler.

`clean_answer()` modelin ekleyebildiği `Cevap:`, `Kaynak:`, `[Parça 1-3]`, `(Parça 1)` ve `(Parça 3)` gibi istenmeyen etiketleri temizler. Etiket kaldırılırken noktalama önündeki gereksiz boşluklar da düzeltilir. Parça numaraları yalnızca modelin context'i ayırmasına yardım eder; kullanıcı kaynak bilgisini aşağıdaki kaynak tablosundan görür.

`is_valid_answer()` cevabın boş, aşırı kısa veya yalnızca kaynak etiketi olup olmadığını kontrol eder. Geçersiz cevaplar `main.py` tarafından fallback'e yönlendirilir.

### `eval.py`

Projenin hızlı kalite kontrol programıdır. LLM'i başlatmadan index, embedding, retrieval ve cevap kalite kurallarını test eder.

Kontroller:

- Index boş mu?
- Embeddingler 384 boyutlu mu?
- Embeddingler geçerli ve sonlu sayılardan mı oluşuyor?
- Boş/kısa/etiketli cevaplar başarısız kabul ediliyor mu?
- Bilinen sorular doğru kaynak dosyayı buluyor mu?
- Doküman dışı soru similarity eşiğinin altında kalıyor mu?

Çalıştırma:

```bash
python eval.py
```

Son doğrulanan sonuç:

```text
PASS  index_health
PASS  answer_quality
PASS  rag_definition
PASS  embedding_definition
PASS  data_mining_process
PASS  out_of_scope_weather

6/6 test başarılı
```

### `eval_cases.json`

Eval senaryolarını koddan ayrı, okunabilir veri halinde tutar.

Her vaka şu bilgilerin bir kısmını içerir:

- Test adı
- Kullanıcı sorusu
- Beklenen davranış (`relevant` veya `not_found`)
- Beklenen kaynak dosya
- Minimum veya maksimum skor

Yeni bir doküman veya konu eklendiğinde bu dosyaya yeni sorular eklenmelidir.

### `embedding_test.py`

Projenin ilk aşamasında embedding ve cosine similarity mantığını anlamak için yazılmış bağımsız deneme dosyasıdır.

Örnek dokümanları ve `RAG ne işe yarar?` sorusunu embedding'e çevirerek en yakın metni bulur. Ana uygulama bu dosyayı kullanmaz; eğitim ve basit smoke test amacı taşır.

### `foundry_test.py`

Foundry Local ve OpenAI-compatible endpoint entegrasyonunu ana RAG uygulamasından bağımsız test eder.

Modelin cache'te bulunması gerekir. Ana uygulama bu dosyayı çağırmaz.

### `docs/`

RAG bilgi kaynağıdır. Şu anda örnek TXT ve veri madenciliği PDF'i bulunur.

Yeni bir TXT veya metin tabanlı PDF eklendikten sonra `/reindex` çalıştırılmalıdır.

Tarama/görüntü şeklindeki PDF'lerde `pypdf` metin çıkaramaz. Böyle dosyalar için ileride OCR desteği gerekir.

### `data/rag.db`

Ingestion sonucu üretilen SQLite veritabanıdır. Tekrar üretilebilir yerel state olduğu için Git'e eklenmez.

Silinirse dokümanlar kaybolmaz; `/reindex` ile yeniden oluşturulur.

### `requirements.txt`

Python bağımlılıklarını ve sürümlerini tutar.

Ana paketler:

- `sentence-transformers`: Embedding üretimi
- `scikit-learn`: Cosine similarity
- `numpy`: Vektör doğrulama ve matris işlemleri
- `pypdf`: PDF metin çıkarma
- `foundry-local-sdk`: Local model yönetimi
- `openai`: Foundry Local endpoint'ine chat completion çağrısı

### `.gitignore`

Git'e eklenmemesi gereken yerel/üretilen dosyaları tanımlar:

- `.venv/`
- `__pycache__/`
- `.env`
- SQLite dosyaları ve `data/`
- Log ve test cache dosyaları
- Eski Git geçmişi yedeği

## 5. Cevap modları

### Dokümanda bilgi yok

En iyi similarity skoru `0.20` altında kalırsa LLM çağrılmaz:

```text
Bu bilgi verilen dokümanlarda yok.
```

### `extractive`

Tek bir context chunk'ı varsa, skoru en az `0.50` ise ve metin 500 karakterden kısa ise chunk doğrudan cevap olarak gösterilir.
CLI bu modu kullanıcıya `Doğrudan` etiketi ve yeşil çerçeveyle gösterir.

Avantajları:

- Generation süresi sıfırdır.
- Kaynak metin bozulmaz.
- Hallucination riski düşer.

### `generative`

Birden fazla chunk'ın sentezlenmesi gerektiğinde Foundry Local modeli context'e göre cevap üretir.
CLI bu modu kullanıcıya `Üretken` etiketi ve cyan çerçeveyle gösterir.

İlk generative cevapta model yükleme süresi de `generation` süresine dahildir. Aynı oturumdaki sonraki cevaplar daha hızlı olabilir.

### `fallback_extractive`

Generative cevap şu durumlarda başarısız kabul edilir:

- Model yükleme veya completion hatası
- Boş cevap
- 30 karakterden kısa cevap
- Yalnızca kaynak/parça etiketi içeren cevap

Bu durumda uygulama çökmez; en güçlü retrieval chunk'ını cevap olarak gösterir.
CLI güvenli geri dönüşü `Kaynak metni` etiketi ve sarı çerçeveyle görünür kılar.

Fallback ham chunk döndürür. Chunker cümle sınırını tercih eder; yalnızca çok uzun ve noktalamasız metinlerde cevap cümlenin ortasından başlayabilir.

## 6. Kurulum ve çalıştırma

Projeye geç:

```bash
cd /Users/erdemac/Developer/local-rag-assistant
```

Virtual environment'ı etkinleştir:

```bash
source .venv/bin/activate
```

Bağımlılıkları yüklemek gerekirse:

```bash
pip install -r requirements.txt
```

Uygulamayı başlat:

```bash
python main.py
```

CLI komutları:

```text
/help       Komutları gösterir
/stats      Index, model ve threshold bilgilerini gösterir
/sources    İndeksteki dosya, tür, sayfa ve chunk sayılarını gösterir
/doctor     Doküman, indeks, embedding ve Foundry/model sağlığını kontrol eder
/reindex    docs/ klasörünü yeniden işler
/debug on   Retrieval, context ve hata detaylarını açar
/debug off  Debug çıktısını kapatır
/exit       Uygulamadan çıkar
```

Önerilen ilk kullanım:

```text
/reindex
/stats
/sources
/doctor
RAG nedir?
Hava nasıl?
/exit
```

## 7. Foundry Local model yönetimi

Katalogdaki modelleri gör:

```bash
foundry model list
```

İndirilmiş modelleri gör:

```bash
foundry cache list
```

Model indir:

```bash
foundry model download phi-4-mini
```

Yüklü modelleri gör:

```bash
foundry service ps
```

Modeli bellekten çıkar:

```bash
foundry model unload qwen3-4b
```

Modeli disk cache'inden sil:

```bash
foundry cache remove qwen3-4b
```

Uygulamanın kullandığı chat modelini değiştirmek için `app/llm.py` içindeki `MODEL_ALIAS` değeri değiştirilir. Chat modeli değiştiğinde `/reindex` gerekmez; embedding modeli ve veritabanı aynı kalır.

Model katalogda bulunup cache'te bulunmuyorsa mevcut kodun `load_model()` çağrısı hata verir ve fallback devreye girer. Önce `foundry model download <alias>` çalıştırılmalıdır.

## 8. Test yaklaşımı

Her ingestion, retrieval, chunking veya threshold değişikliğinden sonra:

```bash
python eval.py
```

çalıştırılmalıdır.

Her prompt, LLM veya fallback değişikliğinden sonra eval'e ek olarak `python main.py` ile en az şu manuel kontroller yapılmalıdır:

1. Dokümanda açıkça bulunan kısa bir soru
2. Birden fazla chunk gerektiren generative soru
3. Dokümanla alakasız bir soru
4. Mümkünse aynı oturumda iki generative soru ile sıcak model süresi

`eval.py` retrieval davranışını deterministik biçimde test eder; LLM'in Türkçe akıcılığını tam olarak ölçmez. Model cevapları manuel olarak da değerlendirilmelidir.

## 9. Bilinen sınırlamalar

- `phi-4-mini` Türkçe cevaplarda önceki Phi-3.5 denemelerine göre daha iyi sonuç vermiştir, ancak dil kalitesi hâlâ manuel kontrol edilmelidir.
- İlk model yükleme ve ilk embedding çağrısı yavaştır.
- Generation ölçümü ilk çağrıda model yükleme süresini de içerir.
- `pypdf`, bazı bozuk PDF nesnelerinde `Ignoring wrong pointing object` uyarısı verebilir; metin yine çıkarılabilir.
- Görüntü tabanlı PDF'ler için OCR yoktur.
- Bütün embeddingler SQLite'tan belleğe alınır; bu yapı küçük/orta doküman koleksiyonları içindir.
- Embeddingler SQLite içinde JSON olarak saklanır; özel bir vector database kullanılmaz.
- Çok uzun ve noktalamasız metinlerde chunker kelime sınırına döner; bu durumda fallback tam bir cümlenin ortasından başlayabilir.
- Türkçe gramer kalitesi otomatik olarak güvenilir biçimde ölçülmez.

## 10. Şu ana kadar verilen önemli mimari kararlar

- Sistem local-first olacak.
- Fine-tuning yapılmayacak.
- Embedding ve chat modeli ayrı sorumluluklar olarak tutulacak.
- LLM yalnızca gerektiğinde yüklenecek.
- Güçlü ve kısa tek chunk varsa extractive cevap tercih edilecek.
- Alakasız sorular threshold ile engellenecek.
- PDF kaynaklarında sayfa metadata'sı tutulacak.
- Chunking overlap içerecek.
- Reindex atomik olacak; başarısız işlem mevcut indeksi bozmayacak.
- Bozuk LLM cevabı uygulamayı çökertmeyecek; fallback kullanılacak.
- Değişiklikler eval testleriyle ölçülecek.

## 11. Son doğrulanan proje durumu

Son eval ve unit test çalışmasında:

```text
11 chunk
2 kaynak dosya
6/6 eval testi başarılı
29/29 unit testi başarılı
```

Başarılı kontroller:

- Index ve embedding sağlığı
- Cevap kalite kararları
- RAG sorusunda doğru TXT kaynağı
- Embedding sorusunda doğru TXT kaynağı
- Veri madenciliği sorusunda doğru PDF kaynağı
- Hava sorusunda threshold altında kalma
- `/sources` komutu: dosya/tür/sayfa/chunk özeti, boş indeks ve eski/eksik şema güvenliği
- `/doctor` komutu: 5 sağlık kontrolü başarılı, model yüklemeden Foundry/Phi-4 cache doğrulaması
- Standart hata çıktıları: çözüm önerileri, debug ayrıntıları ve hata sonrası oturumun devam etmesi
- Rich terminal görünümü: semantik cevap renkleri, Türkçe mod/süre etiketleri, ortak sol hiza, dar terminal ve TTY spinner kontrolü
- LLM cevap temizliği: köşeli/parantezli tekli, aralıklı ve listeli parça atıflarının kaldırılması
- Yerel embedding snapshot yüklemesi: cache varken ağ isteği olmadan 384 boyut doğrulaması

## 12. Yakın roadmap

### Tamamlanan V1 özellikleri

- `/sources`: İndeksteki dosya, tür, sayfa ve chunk sayılarını listeler.
- `/doctor`: Doküman, veritabanı, embedding ve Foundry/model cache sağlığını kontrol eder.
- Standart hata mesajları: kullanıcı mesajını teknik ayrıntıdan ayırır ve çözüm gösterir.
- Rich terminal görünümü: sade banner, semantik cevap paneli, hizalı tablolar, Türkçe süreler ve uzun işlemler için spinner gösterir.

### V1'i tamamlama

1. Fallback'in gerçek model entegrasyonunu farklı hata senaryolarıyla izlemek.
2. `/model` ve `/config` bilgi komutlarını eklemek.
3. Model karşılaştırması yapmak (`phi-3.5-mini`, `phi-4-mini`, gerekirse Qwen/Mistral).
4. Varsayılan modeli test sonuçlarına göre seçmek.
5. Model alias'ını environment variable veya CLI komutuyla değiştirilebilir yapmak.
6. Ana README'yi kullanım ve portfolyo sunumu için düzenlemek.
7. Gerçek `local-rag` terminal entrypoint'i eklemek.

### V2 fikirleri

- Streamlit web arayüzü
- FastAPI endpoint'leri (`/ask`, `/reindex`, `/stats`)
- OCR desteği
- Conversation history
- Daha büyük veri setleri için vector database
- Neighbor chunk genişletme veya reranking
- Otomatik model karşılaştırma raporu

## 13. Projeye geri dönerken kısa kontrol listesi

```bash
cd /Users/erdemac/Developer/local-rag-assistant
source .venv/bin/activate
git status
python eval.py
python main.py
```

Yeni doküman eklenmişse:

```text
/reindex
/stats
/sources
```

Bir şey beklenmedik davranıyorsa:

```text
/debug on
```

Bu rehber, dosya sorumlulukları veya roadmap değiştikçe güncellenmelidir.
