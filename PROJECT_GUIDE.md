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
Token-aware chunk oluşturma (110 token, 20 token overlap)
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
├── pyproject.toml             # Paket metadata'sı ve local-rag entrypoint'i
├── app/
│   ├── __init__.py            # Paket sürümü
│   ├── benchmark.py
│   ├── cli_input.py
│   ├── cli_output.py
│   ├── config.py
│   ├── database.py
│   ├── document_manager.py
│   ├── embeddings.py
│   ├── health.py
│   ├── index_state.py
│   ├── ingest.py
│   ├── llm.py
│   ├── prompts.py
│   ├── project.py
│   ├── rag_service.py
│   ├── session.py
│   └── retrieval.py
├── data/
│   ├── rag.db                 # Üretilen yerel veritabanı, Git'e eklenmez
│   └── exports/               # İsteğe bağlı oturum dışa aktarımları
├── docs/
│   ├── example.txt
│   ├── datamining.pdf
│   └── cybersecurity.txt
├── eval.py
├── eval_cases.json
├── benchmark_cases.json
├── embedding_test.py
├── foundry_test.py
├── main.py
├── requirements.txt
├── tests/
└── PROJECT_GUIDE.md
```

## 4. Dosyalar ne yapıyor?

### `main.py`

Uygulamanın ana giriş noktasıdır. Terminal arayüzünü, proje yolu seçimini ve
komut yönlendirmeyi yönetir; RAG kararları `app/rag_service.py` içindedir.

Başlıca sorumlulukları:

- Açılış banner'ını ve sade `>` prompt'unu gösterir.
- `/help`, `/stats`, `/model`, `/config`, `/sources`, `/show`, `/filter`, `/ask`, `/history`, `/repeat`, `/export`, `/doctor`, `/add`, `/remove`, `/benchmark`, `/reindex`, `/debug on`, `/debug off` ve `/exit` komutlarını işler.
- `--project` veya `LOCAL_RAG_HOME` ile aktif docs/data kökünü seçer.
- `RAGService` tarafından döndürülen yapılandırılmış sonucu Rich bileşenlerine verir.
- Kaynakları, skorları ve süreleri ekrana yazdırır.
- Başarılı sonuçları oturum geçmişine ekler; iptal edilen veya hata alan akışları kaydetmez.
- İndeks dokümanlardan geri kaldıysa cevap öncesinde reindex uyarısı gösterir.

LLM uygulama açılır açılmaz yüklenmez. `get_llm()` fonksiyonu sayesinde yalnızca ilk generative cevap gerektiğinde yüklenir ve aynı oturumda tekrar kullanılır. Buna lazy loading denir.

`/model` aktif chat ve embedding modellerini, yerel cache durumlarını ve mevcut CLI oturumunda belleğe yüklenip yüklenmediklerini gösterir. `/config` retrieval, cevap kalitesi ve chunking ayarlarını açıklamalarıyla listeler. İki komut da salt okunurdur; model yüklemez, inference yapmaz, indeks veya ayar değiştirmez.

`answer_question()` ortak servis sonucunu terminalde gösterir. Hem interaktif
`>` döngüsü hem `local-rag ask` bu fonksiyonu çağırır; retrieval ve cevap modu
kararlarının tek gerçek kaynağı ise `RAGService.answer()` fonksiyonudur.

`cli()` argparse alt komutlarını işler. Argümansız çağrıda interaktif oturumu açar; `ask` tek sorudan sonra çıkar, diğer alt komutları ortak komut çalıştırıcısına yönlendirir. Başarı `0`, operasyonel hata `1`, geçersiz terminal kullanımı `2` exit code üretir.

### `app/rag_service.py`

Terminalden bağımsız RAG çekirdeğidir. Retrieval, threshold, context seçimi,
extractive/generative karar, LLM fallback ve süre ölçümünü yürütür. Sonucu şu
yapılandırılmış nesnelerle döndürür:

- `RAGResult`: cevap, mod, en iyi skor, kaynak filtresi ve uyarı bilgisi
- `RAGSource`: dosya, sayfa, chunk ID/metni, retrieval skoru ve eşleşme/komşu rolü
- `RAGTimings`: retrieval, generation ve toplam süre

Servis Rich veya argparse bilmez. CLI, ilerleme ve debug görünümü için genel
callback'ler verir. Üretken akış `retrieval`, `model` ve `generation` aşamalarını
ayrı bildirir; terminal bunları tek satırda gösterebilir. Bu ayrım çekirdeğin
terminal metni ayrıştırmadan doğrudan test edilmesini sağlar.

İnteraktif TTY akışında servis ayrıca bir streaming callback'i alır. LLM'den
gelen temizlenmiş kısmi cevaplar bu callback üzerinden sunum katmanına taşınır.
`KeyboardInterrupt` fallback hatası gibi yutulmaz; üst katmana çıkar. Böylece
`Ctrl+C`, yarım cevabı kaynak fallback'i sanmadan gerçekten generation'ı iptal
eder.

Retrieval sonuçları önce relevance skoruyla seçilir. Sentez gereken cevaplarda
seçilmiş chunkların bir önceki ve bir sonraki komşusu, toplam context üst sınırı
aşılmadan eklenir. Seçimde hem sabit context eşiği hem de en iyi skora göre
izin verilen fark kullanılır; context eşiğinin altındaki komşular eklenmez.
Kaynak tablosu relevance sırasını korur ve komşuları ayrı rolle gösterir; LLM
prompt'u ise aynı parçaları kaynak, sayfa ve chunk düzenine göre okur. Böylece
model sonuç paragrafını girişten önce görmez ve uzak konular prompt'u kirletmez.

Retrieval yeterli kanıt bulduğu halde model tam kapsam dışı cümlesini döndürürse
bu üretim geçerli sayılmaz. Servis, prompt context'indeki cümleleri soru
terimleriyle karşılaştırır ve en çok örtüşen kaynak cümlelerine fallback yapar.
Böylece en yüksek cosine skorlu chunk doğrudan cevabı içermese bile yanlış veya
eksik bir kaynak metni gösterme riski azalır.

### `app/project.py`

Aktif çalışma kökünü ve bunun altındaki `docs/`, `data/rag.db`, CLI geçmişi,
oturum export klasörü ile benchmark raporu yollarını çözer. Öncelik sırası:

1. Global `--project` seçeneği
2. `LOCAL_RAG_HOME` ortam değişkeni
3. Kurulu repository kökü

Bu sayede `local-rag`, shell'in hangi klasörde olduğundan bağımsız çalışır.

### `app/cli_input.py`

Gerçek interaktif terminalde `prompt-toolkit` kullanarak çerçeveli, tek satırlı
giriş alanını yönetir. `/` yazıldığı anda komutlar kısa açıklamalarıyla girişin
üstünde açılır. Ok tuşları menü seçimini veya geçmişi, Tab ise seçilen komutu,
kaynak adını, `/add` dosya yolunu ve `/debug` değerini tamamlar. Piped veya TTY
olmayan kullanımda sade `input()`/readline yolu korunur.

Yazılan slash komutunun adı ve argümanları ayrı stillerle renklendirilir.
Parametre isteyen komutlarda `/show <chunk-id>` gibi bağlamsal kullanım ipucu
çerçevenin içinde açılır. Çerçevenin altındaki durum satırı aktif model, kaynak
sayısı, indeks güncelliği ve kaynak filtresini gösterir. Bu bilgiler her tuşta
değil, yeni prompt açılırken bir kez alınır.

Geçmiş yalnızca yerelde `data/cli_history` içinde düz metin olarak ve `0600`
izniyle tutulur. Yeni giriş motoru mevcut geçmiş biçimini değiştirmez. Bu geçmiş
yalnızca terminal ergonomisi içindir; model konuşma context'i değildir.

Klavye davranışları bağlama göre düzenlenmiştir. `Esc` açık tamamlama menüsünü
kapatır. `Ctrl+L` ekranı temizleyip aktif prompt'u yeniden çizer. Girişte metin
varken `Ctrl+C` yalnızca satırı temizler; satır boşsa oturumu kapatır. Model
generation'ı sırasında aynı `Ctrl+C` üst katmanda yakalanır, streaming bağlantısı
kapatılır ve kısmi cevap oturum geçmişine yazılmaz.

### `app/session.py`

Mevcut interaktif oturumda başarıyla tamamlanan yapılandırılmış RAG sonuçlarını
bellekte tutar. Her kayda sıra numarası, zaman, soru, cevap, cevap modu, skor,
kaynak filtresi, kaynak metadata'sı ve süreler eklenir.

- `/history`: Kayıtları kısa bir tablo halinde listeler.
- `/repeat [id]`: Son veya seçilen soruyu ilk çalıştığı kaynak filtresiyle tekrarlar.
- `/export markdown [yol]`: Okunabilir bir Markdown oturum raporu üretir.
- `/export json [yol]`: Aynı veriyi araçların işleyebileceği JSON biçiminde üretir.

Varsayılan hedef `data/exports/` klasörüdür. Kullanıcı açık bir yol verirse
boşluklu yollar tırnakla kullanılabilir. Export mevcut dosyanın üzerine yazmaz.
Kaynakların ID, dosya, sayfa, parça ve skor metadata'sı saklanır; doküman
içeriğini gereksiz çoğaltmamak için tam `chunk_text` export edilmez. Bu özellik
konuşma hafızası değildir ve önceki cevapları modele context olarak göndermez.

### `app/document_manager.py`

Kullanıcının doküman ekleme ve silme işlemlerini `main.py` içindeki terminal gösteriminden bağımsız yürütür:

- `validate_document()`: Dosyanın varlığını, TXT/PDF türünü ve indekslenebilir metin içerdiğini doğrular. TXT için UTF-8 şarttır; görüntü tabanlı ve metinsiz PDF reddedilir.
- `add_document()`: Dosyayı `docs/` klasörüne özel oluşturma moduyla kopyalar. Aynı isim varsa üzerine yazmaz; kopyalama/doğrulama yarıda kalırsa eksik hedefi temizler.
- `resolve_managed_document()`: Silme hedefinin yalnızca bir dosya adı olmasını zorunlu kılar. Mutlak yollar ve `../` ile `docs/` dışına çıkış reddedilir.
- `remove_document()`: Doğrulanan dosyayı siler. Kullanıcı onayı terminal katmanında alınır.

Add/remove sonrasında reindex otomatik çalışmaz. Böylece embedding gibi pahalı bir işlem kullanıcı kararı dışında başlamaz. İndeks güncelliği sistemi değişikliği hemen gösterir ve kullanıcı `/reindex` veya `local-rag reindex` ile ne zaman güncelleyeceğini seçer.

### `pyproject.toml`

Projeyi standart bir Python paketi olarak tanımlar. `local-rag = "main:cli"` kaydı, virtual environment içindeki `local-rag` executable'ını üretir. Sürüm `app.__version__` üzerinden okunur; mevcut sürüm `0.1.0`dır.

### `app/config.py`

Uygulamanın davranış ayarlarını tek yerde tutar.

Mevcut ayarlar:

```python
SIMILARITY_THRESHOLD = 0.20
CONTEXT_SCORE_THRESHOLD = 0.35
CONTEXT_RELATIVE_SCORE_MARGIN = 0.20
TOP_K = 3
NEIGHBOR_CHUNK_RADIUS = 1
MAX_CONTEXT_CHUNKS = 5

USE_EXTRACTIVE_FALLBACK = True
EXTRACTIVE_SCORE_THRESHOLD = 0.50
MAX_EXTRACTIVE_CHARS = 500

MIN_GENERATIVE_ANSWER_CHARS = 30
```

Anlamları:

- `SIMILARITY_THRESHOLD`: En iyi skor bunun altındaysa soru dokümanla alakasız kabul edilir.
- `CONTEXT_SCORE_THRESHOLD`: LLM'e yalnızca bu skoru geçen chunklar gönderilir.
- `CONTEXT_RELATIVE_SCORE_MARGIN`: En iyi sonuçtan bundan daha uzak eşleşmeleri context dışında bırakır.
- `TOP_K`: Retrieval aşamasında en iyi kaç chunk'ın alınacağını belirler.
- `NEIGHBOR_CHUNK_RADIUS`: Eşleşmenin çevresinden kaç önceki/sonraki chunk adayının alınacağını belirler.
- `MAX_CONTEXT_CHUNKS`: Modele giden toplam eşleşme ve komşu sayısını sınırlar.
- `USE_EXTRACTIVE_FALLBACK`: Tek güçlü chunk'ın LLM kullanılmadan cevap olmasına izin verir.
- `EXTRACTIVE_SCORE_THRESHOLD`: Extractive cevap için gereken minimum skor.
- `MAX_EXTRACTIVE_CHARS`: Çok uzun chunkların doğrudan cevap olarak dönmesini engeller.
- `MIN_GENERATIVE_ANSWER_CHARS`: Bundan kısa LLM cevapları başarısız kabul edilir.

Bu değerler rastgele seçilmemiştir; mevcut küçük eval seti ve manuel testlerle başlangıç değerleri olarak belirlenmiştir. Doküman sayısı büyüdüğünde yeniden kalibre edilebilir.

### `app/cli_output.py`

Rich tabanlı terminal sunumunu tek merkezden yönetir:

- Özgün mini terminal robotu içeren açılış paneli
- Göz yormayan ortak bordo tema
- `/help`, `/stats`, `/sources` ve `/doctor` tabloları
- Cevap başlığında Türkçe cevap modu ve en iyi retrieval skoru
- Kaynak tablosu ve kompakt performans satırı
- Reindex gibi bağımsız işler için spinner
- Arama, model hazırlama ve yanıt üretimini aynı satırda ilerleten RAG göstergesi
- Standart hata, uyarı, başarı ve bilgi mesajları

Cevap renkleri dekorasyon için değil, durum bilgisini hızlı okutmak için kullanılır:

| İç mod | Kullanıcı etiketi | Renk | Anlam |
| --- | --- | --- | --- |
| `generative` | Üretken | Yumuşak bordo | Yerel LLM birden fazla kaynağı sentezledi. |
| `extractive` | Doğrudan | Yumuşak yeşil | Güçlü ve kısa kaynak metni doğrudan kullanıldı. |
| `fallback_extractive` | Kaynak metni | Yumuşak amber | LLM yerine güvenli kaynak metnine dönüldü. |
| `no_evidence` | Kanıt bulunamadı | Gri | Soru için yeterli doküman kanıtı bulunamadı. |

Teknik mod adları normal kullanıcı görünümünde gösterilmez. Panel başlığı `Cevap · Üretken · Skor 0.6174`, süre satırı ise `Arama · Yanıt · Toplam` biçimindedir. Cevap paneli, kaynak tablosu ve süreler aynı sol hizayı kullanır.

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

`source_manifest` tablosu ise her desteklenen dokümanın adını, türünü, dosya boyutunu ve SHA-256 özetini tutar. Bu tablo “şu anki `docs/` klasörü, bu indeksi üreten dosyalarla aynı mı?” sorusuna cevap verir.

Önemli fonksiyonlar:

- `init_db()`: Tabloyu oluşturur ve eksik metadata kolonlarını ekler.
- `insert_chunk()`: Bir chunk ve metadata'sını kaydeder.
- `replace_chunks()`: Eski chunk ve manifesti yenileriyle tek transaction içinde değiştirir. Herhangi bir ekleme başarısız olursa rollback ile eski indeks ve manifest birlikte korunur.
- `get_all_chunks(source_name=None)`: Retrieval için bütün chunkları veya yalnızca parametreli kaynak filtresine uyan kayıtları okur.
- `get_chunk_by_id()`: `/show` için embeddingi yüklemeden tek chunk'ın metadata ve tam metnini okur.
- `get_source_manifest()`: İndeksi üreten doküman özetlerini okur; eski şemalarda güvenli biçimde boş liste döndürür.
- `get_chunk_stats()`: `/stats` komutuna chunk ve kaynak sayısını verir.
- `get_indexed_sources()`: `/sources` komutuna dosya bazında tür, sayfa ve chunk özetini verir. Veritabanı dosyası varsa sorgu öncesi şemayı güvenli şekilde hazırlar.

`ensure_chunk_metadata_columns()` küçük bir migration görevi görür. Eski `rag.db` dosyasında yeni kolonlar yoksa tabloyu silmeden kolonları ekler.

### `app/health.py`

`/doctor` komutunun sağlık kontrollerini terminal gösteriminden bağımsız olarak yürütür:

- `docs/` klasörü ile TXT/PDF varlığını kontrol eder.
- Doküman manifestini mevcut dosyalarla karşılaştırarak indeksin güncel olup olmadığını kontrol eder.
- SQLite indeksinin okunabildiğini, kaynak ve chunk sayılarını doğrular.
- Embeddinglerin 384 boyutlu ve sonlu sayılardan oluştuğunu denetler.
- `foundry` terminal aracının ve model cache dizininin varlığını kontrol eder.
- `phi-4-mini` model dosyalarının cache içinde gerçekten bulunduğunu doğrular.

Bu kontrol model yüklemez, model indirmez ve inference yapmaz. Her sonuç `ok`, `warning` veya `error` durumuyla birlikte gerektiğinde çözüm önerisi taşır.

### `app/index_state.py`

İndeks ile `docs/` klasörünün aynı veri sürümünü temsil edip etmediğini izler. TXT ve PDF dosyalarını ada göre sıralar, dosya içeriğini bloklar halinde okuyup SHA-256 özeti üretir ve SQLite'taki `source_manifest` ile karşılaştırır.

Olası durumlar:

- `current`: Dokümanlar indeksle eşleşir.
- `stale`: En az bir dosya eklenmiş, değiştirilmiş veya silinmiştir.
- `untracked`: Eski indeks vardır ama henüz manifest kaydı yoktur.
- `missing`: Veritabanı henüz yoktur.
- `error`: Dosya veya manifest okunamamıştır.

Karşılaştırma yalnızca dosya tarihine dayanmaz. İçerik özeti kullanıldığı için dosya zamanı korunmuş olsa bile gerçek içerik değişikliği algılanır.

### `app/ingest.py`

Dokümanları RAG sisteminin arayabileceği hale getirir. Bu işleme ingestion denir.

Akış:

1. `docs/` içindeki `.txt` ve `.pdf` dosyalarının ilk manifestini üretir.
2. TXT dosyasını UTF-8 metin olarak okur.
3. PDF dosyasını `pypdf.PdfReader` ile sayfa sayfa okur.
4. Her sayfa/paragraf metnini chunklara böler.
5. Chunkları toplu olarak embedding'e çevirir.
6. Bütün yeni indeks kayıtlarını bellekte hazırlar.
7. Doküman manifestini yeniden üretir; ilk manifestten farklıysa yazmayı iptal eder.
8. Chunkları ve manifesti tek transaction ile SQLite'a yazar.

Chunk ayarları:

```python
CHUNK_SIZE = 110
CHUNK_OVERLAP = 20
```

Bu değerler karakter değil embedding tokenizer'ının token sayısıdır ve özel
tokenları da kapsar. Kullanılan embedding modelinin maksimum girişi 128
tokendır; 110 tokenlık sınır modelin göremeyeceği metin kuyruğu bırakmaz.
Overlap, iki ardışık chunk arasında en fazla 20 tokenlık ortak alan hedefler.
Chunk sınırlarında önce cümle sonu, mümkün değilse tam kelime/token sınırı
seçilir; önceki chunk tam cümlede bittiyse yeni chunk sonraki cümleden başlar.

`read_pdf_file()` her PDF sayfasını ayrı bir document kaydı olarak üretir. Bu sayede cevap kaynaklarında `page=2` gibi sayfa bilgisi gösterilebilir.

Metni başarıyla çıkarılan bozuk PDF'lerdeki bilinen `Ignoring wrong pointing object` mesajları kullanıcı terminalini kirletmemesi için filtrelenir. Gerçek okuma hataları exception olarak görünmeye devam eder.

`/reindex` komutu doğrudan `ingest_documents()` fonksiyonunu çağırır. Doküman okuma veya embedding üretme başarısız olursa veritabanına dokunulmaz. Bir dosya indeksleme sürerken değişirse tutarsız bir indeks yazılmaz. SQLite yazımı sırasında hata oluşursa chunklar ve manifest birlikte rollback edilir; önceki indeks kullanılmaya devam eder.

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
- `get_embedding_tokenizer()`: Ingestion'ın embedding modeliyle aynı token sınırını kullanmasını sağlar.
- `embed_text()`: Tek bir metni embedding'e çevirir.
- `embed_texts()`: Birden fazla metni batch halinde embedding'e çevirir.

Ingestion ve retrieval aynı model instance'ını kullanır. Yerel snapshot tercihi, model daha önce indirilmişken gereksiz ağ kontrolünü ve retry loglarını engeller. İlk çağrı yine modelin belleğe alınması nedeniyle sonraki çağrılardan yavaş olabilir.

### `app/retrieval.py`

Kullanıcı sorusuna en yakın chunkları bulur.

Akış:

1. SQLite'tan bütün chunkları veya seçilmiş kaynağın chunklarını alır.
2. Sorunun embedding'ini üretir.
3. Chunk embeddinglerini NumPy `float32` matrisine dönüştürür.
4. Bozuk, `NaN` veya sonsuz embeddingleri filtreler.
5. Soru ve chunk matrislerini scikit-learn ile L2 normalize eder; NumPy `einsum` ile normalized dot product hesaplar. Bu değer cosine similarity ile aynıdır ve mevcut NumPy/BLAS ortamındaki matmul uyarılarını önler.
6. Sonuçları yüksek skordan düşük skora sıralar.
7. En iyi `TOP_K` sonucu seçer.
8. Seçilen her sonuç için aynı kaynaktaki önceki/sonraki chunkı gerçek cosine
   skoruyla neighbor adayı olarak ekler.

Retrieval yalnızca ilgili metni bulur; cevap üretmez.
Kaynak filtresi SQL'e parametre olarak verilir; dosya adı sorgu metnine eklenmez.

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

`LOCAL_RAG_MODEL` çevre değişkeni doluysa aktif alias bu değerden alınır; aksi halde `phi-4-mini` kullanılır. `LocalLLM(model_alias=...)` benchmark gibi kontrollü akışların modeli açıkça seçebilmesini sağlar.

`LocalLLM` sınıfı:

1. Foundry Local servis durumunu kontrol eder; normal modda gerekirse servisi sessiz başlatır.
2. Seçilen modeli yerel cache'ten yükler.
3. Foundry Local'ın OpenAI-compatible endpoint'ine bağlanan client'ı oluşturur.
4. Chat completion çağrısı yapar.
5. Ham cevabı `clean_answer()` ile temizler.

TTY kullanımında `generate_answer_stream()` OpenAI-compatible streaming
cevabını parça parça okur. Her yeni parça birikmiş metne eklenir ve
`clean_streaming_preview()` ile kullanıcıya gösterilmeden önce `Cevap:` ile
`[Parça 1]` benzeri etiketlerden arındırılır. Akış başarıyla bittiğinde aynı
nihai `clean_answer()` ve kalite kontrolleri uygulanır. Normal bitişte de
`Ctrl+C` iptalinde de response kapatılır; yarım bağlantı açık bırakılmaz.

Foundry SDK normalde `foundry service start` alt sürecinin çıktısını doğrudan terminale bağlar. `create_foundry_manager()` normal modda aynı servisi `stdout` ve `stderr` kapalı başlatıp hazır olana kadar durumunu kontrol eder; böylece servis mesajı Rich spinner satırına karışmaz. Debug modunda başlangıç çıktısı görünür kalır. Servis durumuna 15 saniye, Foundry HTTP/model çağrılarına 120 saniye sınır uygulanır.

`clean_answer()` modelin ekleyebildiği `Cevap:`, `Kaynak:`, `[Parça 1-3]`, `(Parça 1)` ve `(Parça 3)` gibi istenmeyen etiketleri temizler. Etiket kaldırılırken noktalama önündeki gereksiz boşluklar da düzeltilir. Parça numaraları yalnızca modelin context'i ayırmasına yardım eder; kullanıcı kaynak bilgisini aşağıdaki kaynak tablosundan görür.

`is_valid_answer()` cevabın boş, aşırı kısa, yalnızca kaynak etiketi veya aşırı tekrar döngüsü olup olmadığını kontrol eder. Geçersiz cevaplar `main.py` tarafından fallback'e yönlendirilir. Tekrar kontrolü, Phi 3.5 benchmark'ında görülen aynı kelimenin sürekli üretilmesi sorununu yakalamak için eklenmiştir.

### `app/benchmark.py`

Modelleri aynı retrieval sonucu ve promptlarla karşılaştırır. Retrieval her model için tekrar edilmediğinden ölçüm LLM farkına odaklanır. İlk vaka iki kez çalıştırılarak model yükleme, ilk generation ve sıcak generation süreleri ayrılır; diğer vakalar cevap kalitesini genişletir. Geçerli cevap sayısı ve `benchmark_cases.json` içindeki beklenen terimlerin kapsanma oranı hesaplanır. Ayrıntılı cevaplar ve ham süreler `data/model_benchmark.json` içine yazılır.

### `eval.py`

Projenin hızlı kalite kontrol programıdır. LLM'i başlatmadan index, embedding, retrieval ve cevap kalite kurallarını test eder.

Kontroller:

- Index boş mu?
- Embeddingler 384 boyutlu mu?
- Embeddingler geçerli ve sonlu sayılardan mı oluşuyor?
- Boş/kısa/etiketli cevaplar başarısız kabul ediliyor mu?
- Bilinen sorular doğru kaynak dosyayı buluyor mu?
- En iyi chunk beklenen ana kavramları gerçekten içeriyor mu?
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
PASS  security_goals
PASS  phishing_definition
PASS  multi_factor_authentication
PASS  backup_rule
PASS  out_of_scope_weather
PASS  out_of_scope_cooking

11/11 test başarılı
```

### `eval_cases.json`

Eval senaryolarını koddan ayrı, okunabilir veri halinde tutar.

Her vaka şu bilgilerin bir kısmını içerir:

- Test adı
- Kullanıcı sorusu
- Beklenen davranış (`relevant` veya `not_found`)
- Beklenen kaynak dosya
- Minimum veya maksimum skor
- En iyi chunk içinde bulunması gereken kavramlar

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
CLI bu modu kullanıcıya `Üretken` etiketi ve yumuşak bordo çerçeveyle gösterir.

İlk generative cevapta model yükleme süresi de `generation` süresine dahildir. Aynı oturumdaki sonraki cevaplar daha hızlı olabilir.

### `fallback_extractive`

Generative cevap şu durumlarda başarısız kabul edilir:

- Model yükleme veya completion hatası
- Boş cevap
- 30 karakterden kısa cevap
- Yalnızca kaynak/parça etiketi içeren cevap

Bu durumda uygulama çökmez; en güçlü retrieval chunk'ını cevap olarak gösterir.
CLI güvenli geri dönüşü `Kaynak metni` etiketi ve yumuşak amber çerçeveyle görünür kılar.

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

Terminal komutunu editable kur:

```bash
pip install -e .
```

Uygulamayı başlat:

```bash
local-rag
```

`python main.py` aynı interaktif uygulama için desteklenmeye devam eder.

Tek seferlik terminal kullanımları:

```bash
local-rag ask "RAG nedir?"
local-rag ask --source example.txt "RAG nedir?"
local-rag add "/Users/kullanici/Documents/notlar.pdf"
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
local-rag --version
```

`local-rag --debug ask "RAG nedir?"` tek soruluk akışta teknik ayrıntıları açar.
Editable kurulumdan sonra komut repository dışında da çalışır. Başka bir docs/data
kökü seçmek için global seçenek alt komuttan önce yazılır:

```bash
local-rag --project /dosya/yolu/rag-calismasi stats
LOCAL_RAG_HOME=/dosya/yolu/rag-calismasi local-rag
```

CLI komutları:

```text
/help       Komutları gösterir
/stats      Index, model ve threshold bilgilerini gösterir
/model      Model, cache ve oturumdaki lazy-load durumunu gösterir
/config     Aktif RAG ayarlarını açıklamalarıyla salt okunur gösterir
/sources    İndeksteki dosya, tür, sayfa ve chunk sayılarını gösterir
/show <id>  Chunk metadata'sını ve tam kaynak metnini gösterir
/filter <dosya|off> Oturumdaki retrieval kaynağını sınırlar veya filtreyi kapatır
/ask [--source dosya] <soru> Tek soruluk kaynak filtresi uygular
/doctor     Doküman, indeks, embedding ve Foundry/model sağlığını kontrol eder
/add <yol>  TXT veya PDF dosyasını doğrulayıp docs/ klasörüne kopyalar
/remove <ad> Dokümanı onay alarak docs/ klasöründen siler
/benchmark [model] Model sürelerini ve cevap kalitesini karşılaştırır
/reindex    docs/ klasörünü yeniden işler
/debug on   Retrieval, context ve hata detaylarını açar
/debug off  Debug çıktısını kapatır
/exit       Uygulamadan çıkar
```

Önerilen ilk kullanım:

```text
/reindex
/stats
/model
/config
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

Uygulamanın chat modelini kod değiştirmeden seçmek için komutun başına çevre değişkeni eklenir:

```bash
LOCAL_RAG_MODEL=phi-3.5-mini local-rag ask "RAG nedir?"
```

`local-rag model` seçimin `LOCAL_RAG_MODEL` veya varsayılandan geldiğini gösterir. Chat modeli değiştiğinde `/reindex` gerekmez; embedding modeli ve veritabanı aynı kalır.

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
5. Üretken cevap akarken `Ctrl+C` ile iptal ve ardından yeni soru sorabilme
6. `/history`, `/repeat` ve iki export biçiminin beklenen kaydı üretmesi

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
- `/history` kayıtları yalnızca açık interaktif süreçte bellekte yaşar; kalıcılık
  gerekiyorsa oturum kapanmadan `/export` kullanılmalıdır.
- Oturum geçmişi önceki cevapları modele vermez; conversation memory ve takip
  sorusu çözümleme henüz yoktur.

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
24 chunk
3 kaynak dosya
11/11 eval testi başarılı
153/153 unit testi başarılı
```

Başarılı kontroller:

- Index ve embedding sağlığı
- Cevap kalite kararları
- RAG sorusunda doğru TXT kaynağı
- Embedding sorusunda doğru TXT kaynağı
- Veri madenciliği sorusunda doğru PDF kaynağı
- Hava sorusunda threshold altında kalma
- `/sources` komutu: dosya/tür/sayfa/chunk özeti, boş indeks ve eski/eksik şema güvenliği
- `/doctor` komutu: 6 sağlık kontrolü başarılı, indeks güncelliği ile Foundry/Phi-4 cache doğrulaması
- Standart hata çıktıları: çözüm önerileri, debug ayrıntıları ve hata sonrası oturumun devam etmesi
- Rich terminal görünümü: yumuşak bordo tema, özgün açılış maskotu, semantik cevap renkleri, Türkçe mod/süre etiketleri ve tek satırlı RAG aşama göstergesi
- LLM cevap temizliği: köşeli/parantezli tekli, aralıklı ve listeli parça atıflarının kaldırılması
- Çıplak parça etiketi temizliği: cevap sonundaki `Parça 1.` ve `Parça 1-3` biçimlerinin normal “parça” kelimelerine dokunmadan kaldırılması
- Yerel embedding snapshot yüklemesi: cache varken ağ isteği olmadan 384 boyut doğrulaması
- Foundry başlangıcı: normal modda alt süreç çıktısının bastırılması, debug modunda korunması ve timeout hata yolu
- `/model` ve `/config`: model yüklemeden cache/lazy-load durumu ile aktif ayarların gösterilmesi
- `local-rag` paketi: editable kurulum, Türkçe sürüm/yardım, interaktif oturum, `ask`, `add`, `remove`, `reindex` ve bilgi alt komutları
- Ortak soru akışı ve exit code'lar: interaktif/tek-komut davranış birliği, başarı `0`, operasyonel hata `1`
- İndeks güncelliği: SHA-256 manifesti, eklenen/değişen/silinen dosya ayrımı, cevap öncesi uyarı ve atomik rollback
- Güvenli dosya yönetimi: TXT/PDF doğrulama, üzerine yazma ve path traversal koruması, silme onayı, `--yes` otomasyon seçeneği
- Genişletilmiş eval: siber güvenlik dokümanı, dört yeni doğru-kaynak vakası, ikinci kapsam dışı vaka ve chunk kavram kontrolü
- Model yapılandırması: `LOCAL_RAG_MODEL` override'ı, `/model` ve `/config` görünürlüğü
- Model benchmark: Phi 4 için 3/3 geçerli ve %89 terim kapsamı; Phi 3.5 için 2/3 ve %56 kapsam
- Tekrar filtresi: Phi 3.5'in bozuk tekrarlı cevabını geçersiz sayıp normal akışta fallback'e yönlendirme
- Yapılandırılmış servis: CLI'dan bağımsız `RAGResult`, kaynak ve süre sözleşmesi
- Kaynak denetimi: ID ile chunk görüntüleme ve parametreli dosya filtresi
- Proje yolu: repository dışından çalışma, `--project` ve `LOCAL_RAG_HOME` önceliği
- Terminal girişi: çerçeveli prompt, giriş üstünde canlı slash menüsü, komut renklendirme, parametre ipucu, model/kaynak/indeks/filtre durumu, kalıcı geçmiş ve Tab tamamlama
- Terminal uyumluluğu: ok tuşu geçmişi, `0600` geçmiş izni, `Esc`/`Ctrl+L`/bağlamsal `Ctrl+C` davranışı ve TTY olmayan kullanımlarda sade giriş fallback'i
- Streaming generation: tokenlarla güncellenen geçici cevap paneli, bitişte nihai kalite kontrolü ve `Ctrl+C` ile güvenli bağlantı kapatma
- Oturum araçları: `/history`, kaynak filtresini koruyan `/repeat` ve tam chunk metnini taşımayan Markdown/JSON `/export`
- Token-aware chunking: 128 tokenlık model sınırına karşı maksimum 109 token ölçümü, cümle/kelime hizası ve kesilmeyen embedding girdisi
- Context kalitesi: skorla seçim, belge düzeninde prompt, en fazla 5 parçalık komşu genişletme ve kaynaklarda eşleşme/komşu rolü
- Context eval: doğrudan bilgi için en iyi chunk, dağıtılmış bilgi için gerçekten modele giden sınırlı context kavramlarının doğrulanması

## 12. Yakın roadmap

### Tamamlanan V1 özellikleri

- `/sources`: İndeksteki dosya, tür, sayfa ve chunk sayılarını listeler.
- `/doctor`: Doküman, veritabanı, embedding ve Foundry/model cache sağlığını kontrol eder.
- Standart hata mesajları: kullanıcı mesajını teknik ayrıntıdan ayırır ve çözüm gösterir.
- Rich terminal görünümü: sade banner, semantik cevap paneli, hizalı tablolar, Türkçe süreler ve tek satırlı aşamalı RAG ilerlemesi gösterir.
- Sessiz Foundry başlangıcı: normal kullanıcı görünümünde servis logunu gizler, debug modunda ham çıktıyı korur.
- `/model` ve `/config`: model/cache/lazy-load durumunu ve aktif ayarları salt okunur gösterir.
- Kurulabilir CLI: `local-rag` interaktif oturumunu ve tek seferlik alt komutları standart Python entrypoint'iyle sunar.
- İndeks güncelliği: dokümanların SHA-256 manifestini saklar; soru akışı, `/stats` ve `/doctor` üzerinden reindex ihtiyacını bildirir.
- Güvenli dosya yönetimi: interaktif `/add`/`/remove` ile `local-rag add`/`remove` komutlarını doğrulama ve onay kurallarıyla sunar.
- Genişletilmiş eval: `cybersecurity.txt` ile kaynak, skor ve chunk kavramlarını doğrular.
- Model benchmark ve yapılandırma: süre/kalite raporu üretir, `LOCAL_RAG_MODEL` ile kod değiştirmeden model seçer.
- Ana README: kurulum, kullanım, mimari, gerçek benchmark sonuçları, testler, sınırlamalar ve V2 yol haritasını tek giriş noktasında sunar.
- Yapılandırılmış RAG servisi: cevap kararlarını Rich terminal sunumundan ayırır.
- Kaynak denetimi: `/show`, `/filter` ve `ask --source` ile chunk ve dosya bazlı aramayı görünür kılar.
- Taşınabilir CLI: `--project` ve `LOCAL_RAG_HOME` ile çalışma dizininden bağımsız yollar kullanır.
- Terminal ergonomisi: çerçeveli giriş, açıklamalı canlı slash menüsü, durum satırı, komut renklendirme, parametre ipuçları, ok tuşu geçmişi ve bağlama duyarlı Tab tamamlama sağlar.
- Klavye ve streaming ergonomisi: `Esc`, `Ctrl+L`, bağlamsal `Ctrl+C`, token geldikçe güncellenen cevap ve güvenli generation iptali sağlar.
- Oturum geçmişi: `/history`, `/repeat` ve üzerine yazma korumalı Markdown/JSON export sunar.
- Token-aware chunking: embedding tokenizer'ının 128 token sınırına uygun, cümle odaklı 110/20 parçalama sağlar.
- Context hazırlama: relevance seçimini belge okuma sırasından ayırır ve üretken cevaplarda skorlu komşu parçaları en fazla 5 context parçasına kadar ekler.

### Sonraki CLI adımları

Web/API şu an proje hedefinde değildir. Sıradaki geliştirmeler saf CLI deneyimi
ve retrieval kalitesi üzerinde ilerleyecektir.

### Bekleyen fikirler

- Değişen dosya odaklı incremental reindex ve dosya bazlı ilerleme
- SQLite FTS5/BM25 ile dense retrieval'ı birleştiren hybrid search
- Daha geniş aday havuzu için reranking
- OCR desteği
- Conversation history
- Daha büyük veri setleri için vector database
- Otomatik model karşılaştırma raporu

## 13. Projeye geri dönerken kısa kontrol listesi

```bash
cd /Users/erdemac/Developer/local-rag-assistant
source .venv/bin/activate
git status
python eval.py
local-rag stats
local-rag
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
