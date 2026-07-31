# Öğrenme Notları — Bu Projedeki RAG Kavramları

Bu dosya, projede kullanılan teknik kavramların sıfırdan açıklamasıdır. Amacı,
ileride mimari kararlar verirken (hybrid search, reranking, conversation
history) terimlerin ne anlama geldiğini bilerek karar verebilmektir.

> Not: Bu dosya `docs/` klasöründe durur ama RAG indeksine **girmez**.
> İndeksleme yalnızca `.txt` ve `.pdf` uzantılarını alır
> (`app/index_state.py` → `SUPPORTED_DOCUMENT_EXTENSIONS`), bu yüzden `.md`
> dosyaları güvenle `docs/` altında tutulabilir.

Kod ve davranış referansı için [`../AGENTS.md`](../AGENTS.md), ayrıntılı teknik
anlatım için [`../PROJECT_GUIDE.md`](../PROJECT_GUIDE.md).

---

## 1. Temel fikir: neden RAG

Bir dil modelinin bilgisi eğitim verisinde donmuştur. Modelin senin PDF'ini
bilmesinin iki yolu var:

- **Fine-tuning:** modelin ağırlıklarını yeni veriyle güncellemek. Pahalı,
  yavaş, her yeni dokümanda tekrar eğitim gerekir ve model neyi nereden
  bildiğini söyleyemez.
- **RAG (Retrieval-Augmented Generation):** modele hiç dokunmamak. Soru
  geldiğinde ilgili doküman parçalarını *bulup* prompt'un içine koymak. Model
  onları okuyup cevaplar.

RAG'in kazandırdığı: doküman eklemek = bir dosya kopyalamak + indeksi yenilemek.
Ve **kaynak gösterebiliyorsun** — cevabın hangi dosyanın hangi parçasından
geldiğini biliyorsun. Projedeki `Kaynaklar` tablosu tam olarak bu.

Sistemin iki yarısı var: **retriever** (arama motoru) ve **generator** (LLM).
Kalite sorunlarının büyük kısmı retriever'dan gelir ama genelde LLM suçlanır.
Roadmap'in retrieval ağırlıklı olmasının sebebi budur.

---

## 2. Embedding: metni sayıya çevirmek

KNN'de noktaları uzayda konumlandırıp en yakınları buluyorsun. **Embedding,
metni tam olarak böyle bir noktaya çevirme işidir.**

Projedeki model `paraphrase-multilingual-MiniLM-L12-v2`, bir metin alıp
**384 sayıdan oluşan bir vektör** döndürüyor. Yani her chunk, 384 boyutlu
uzayda bir nokta.

Kritik nokta: bu vektör rastgele değil. Model, **anlamı yakın metinler uzayda
yakın düşecek şekilde** eğitilmiş. "Kimlik avı nedir?" ile "phishing saldırısı"
farklı kelimelerdir ama vektörleri birbirine yakındır. TF-IDF'in yapamadığı şey
budur: TF-IDF kelime örtüşmesine bakar, embedding anlama bakar.

Terimler:

- **384 = boyut sayısı (dimension).** Model mimarisinin sabiti, seçilemez.
  Daha büyük boyut genelde daha iyi ayrım, daha çok bellek ve hesap demektir.
- **multilingual** = çok dilli eğitilmiş, Türkçe çalışır. Çoğu embedding modeli
  yalnızca İngilizcedir; bu seçim projenin çalışması için zorunluydu.
- **MiniLM** = küçültülmüş (distilled) model. Laptop'ta hızlı çalışır,
  karşılığında en yüksek kalite değildir.
- **bi-encoder** = soruyu ve dokümanı **ayrı ayrı** vektöre çevirir. Bu yüzden
  chunk vektörleri önceden hesaplanıp SQLite'a kaydedilebiliyor. Karşıtı olan
  cross-encoder için bkz. bölüm 12.

---

## 3. Cosine similarity: yakınlığı ölçmek

KNN'de genelde Öklid mesafesi kullanılır; burada **cosine similarity** var:

```text
cos(θ) = (A · B) / (|A| × |B|)
```

İki vektör arasındaki **açının** kosinüsü. Neden mesafe değil açı?

Vektörün **uzunluğu** metnin uzunluğu/yoğunluğu gibi şeylerden etkilenir,
**yönü** ise anlamı taşır. Uzun bir paragraf ile kısa bir cümle aynı şeyi
söylüyorsa aralarındaki Öklid mesafesi büyük olabilir, ama açı küçüktür. Anlam
aramasında yön önemlidir, büyüklük değil.

Koddaki optimizasyon: vektörler önce **L2 normalize** ediliyor (boyları 1
yapılıyor), sonra basit **dot product** alınıyor. Boy 1 olunca formülün böleni
1 olur, yani `dot product = cosine similarity`. Aynı sonuç, daha hızlı hesap.

Skor aralığı teorik olarak -1 ile 1; pratikte bu modelde 0–1 civarı.
Eval çıktısından: alakalı sorular **0.55–0.87**, alakasız sorular **0.06–0.07**.

---

## 4. Chunking: dokümanı neden parçalıyoruz

İki sebep:

**Sebep 1 — model sınırı.** Embedding modeli en fazla **128 token** alır. Daha
uzun metin verilirse sessizce keser (truncation) ve gerisi hiç görülmez. 200
sayfalık bir PDF tek vektöre çevrilemez.

**Sebep 2 — hassasiyet.** Tüm doküman tek vektör olsaydı, o vektör dokümanın
"ortalama anlamı" olurdu ve hiçbir spesifik soruya yakın çıkmazdı. Küçük
parçalar keskin eşleşme sağlar.

**Token nedir?** Kelime değil, **kelime parçası** (subword). Model metni
sözlüğündeki birimlere böler: "yedekleme" → `yedek` + `##leme` gibi. Türkçe gibi
eklemeli dillerde bir kelime kolayca 3-4 token olur; bu yüzden token saymak
karakter saymaktan farklıdır.

Projedeki ayarlar:

```python
CHUNK_SIZE = 128      # token, karakter değil
CHUNK_OVERLAP = 20    # token
```

**110 neden?** Sınır 128, ama modelin eklediği özel tokenlar var (`[CLS]`,
`[SEP]` gibi — metnin başlangıç/bitiş işaretleri). 110 bırakmak hiçbir parçanın
kesilmeyeceğini garanti eder. Dokümanlardaki "en uzun chunk 109 token" ifadesi
bunun ölçülmüş kanıtıdır.

**Overlap 20 neden?** Parçalar keskin bitseydi, sınıra denk gelen bir cümle iki
parçada da yarım kalır ve hiçbiri soruyla eşleşmezdi. 20 tokenlık örtüşme,
sınırdaki bilginin en az bir parçada bütün olmasını sağlar. Bedeli: tekrarlanan
metin ve biraz daha fazla chunk.

Koddaki incelik: chunk'lar rastgele yerden değil, **mümkünse cümle sonundan**
bölünür. Çünkü extractive modda chunk doğrudan kullanıcıya cevap olarak
gösterilir; cümlenin ortasından başlayan bir cevap kötü görünür.

---

## 5. İki farklı "komşu" — karıştırılmaması gereken ayrım

Projede iki ayrı komşuluk mekanizması var:

### (a) Vektör uzayındaki komşular = `TOP_K = 3`

Klasik KNN'in k'sı. Soru vektörüne **anlamsal olarak** en yakın 3 chunk. Bunlar
dokümanın tamamen farklı yerlerinden, hatta farklı dosyalardan gelebilir.

`k=3` neden? Klasik precision/recall dengesi:

- k çok küçük (1) → doğru cevap 2. sıradaysa kaçırılır
- k çok büyük (10) → alakasız metin prompt'a girer, LLM'in kafası karışır
  (buna **context dilution** denir) ve token maliyeti artar

### (b) Dokümandaki komşular = `NEIGHBOR_CHUNK_RADIUS = 1`

Tamamen farklı bir şey. Eşleşen chunk'ın **aynı dosyadaki bir öncesi ve bir
sonrası**. Uzayda değil, metinde komşu.

Neden var? Bir cevap chunk sınırına bölünmüş olabilir. "Veri madenciliği
süreçleri" başlığı bir chunk'ta, adımların devamı sonraki chunk'ta olabilir.
Anlamsal arama başlığı bulur ama devamını kaçırır; komşu ekleme bunu telafi eder.

`MAX_CONTEXT_CHUNKS = 5` bu genişlemenin tavanıdır: 3 eşleşme + komşular
toplamda 5 parçayı geçemez, yoksa prompt şişer.

Kaynak tablosundaki `Eşleşme` / `Komşu` rolü tam bu ayrımı gösterir.

---

## 6. Eşikler: üç ayrı karar

Bu sayılar üç farklı soruyu cevaplar:

| Eşik | Değer | Sorduğu soru |
|---|---|---|
| `SIMILARITY_THRESHOLD` | 0.20 | Bu soru dokümanlarla **ilgili mi**? Altındaysa LLM hiç çalıştırılmaz. |
| `CONTEXT_SCORE_THRESHOLD` | 0.35 | Bu parça LLM'e **gönderilmeye değer mi**? |
| `CONTEXT_RELATIVE_SCORE_MARGIN` | 0.20 | En iyi sonuçtan çok mu geride? 0.80 varken 0.40'ı almanın anlamı yok. |
| `EXTRACTIVE_SCORE_THRESHOLD` | 0.50 | Ham metni **doğrudan cevap olarak** göstermeye yetecek kadar güvenli mi? |

İlk eşik en değerlisidir: **hallucination'ı kaynağında keser.** "Hava nasıl?"
sorusu 0.069 alır, LLM hiç çağrılmaz, sabit cevap döner. Bu, modelden prompt'ta
"uydurma" diye rica etmekten çok daha güvenilirdir.

Ancak dikkat: alakalılar 0.55+, alakasızlar 0.07. Aradaki boşluk çok büyük. Bu,
eşiklerin **doğru olduğunu değil, henüz zorlanmadığını** gösterir. Gerçek zorluk
"konusu yakın ama cevabı içermeyen" sorularda ortaya çıkar. Roadmap'teki
*hard negative* maddesinin sebebi budur.

---

## 7. Prompt ve context: LLM'e ne veriliyor

LLM'e giden iki mesaj var:

- **system prompt:** kurallar. "Sadece verilen bağlamı kullan, uydurma, bağlam
  yetmezse şu cümleyi yaz."
- **user prompt:** bulunan chunk'lar (`[Parça 1]`, `[Parça 2]`...) ve soru.

Terimler:

- **Context (bağlam):** prompt'a konan doküman metni. RAG'de "context" hep bunu
  kasteder.
- **Context window:** modelin bir seferde okuyabildiği toplam token sayısı.
  Sonsuz değildir; `MAX_CONTEXT_CHUNKS` bu bütçeyi korur.
- **Grounding:** cevabın verilen context'e dayanması. RAG'in bütün amacı.
- **Hallucination:** modelin context'te olmayan bir şeyi uydurup emin bir tonda
  söylemesi. RAG'in çözmeye çalıştığı asıl problem.

Projedeki prompt'ta iki incelik var:

1. Soru **iki kez** yazılır (bağlamın önünde ve arkasında). Modeller uzun
   prompt'ların ortasını daha zayıf hatırlar — buna **"lost in the middle"**
   denir. Soruyu iki uca koymak buna karşı basit bir önlemdir.
2. Chunk'lar LLM'e **belge sırasıyla** verilir, skor sırasıyla değil. Yoksa
   model sonuç paragrafını girişten önce okur ve anlatım bozulur. Ama *kaynak
   tablosunda* skor sırası korunur; kullanıcı için en alakalı olan üstte
   olmalıdır. İki farklı sıralama, iki farklı amaç.

---

## 8. Cevap modları: üç strateji

Soru-cevap literatüründeki standart ayrım:

**`extractive`** — cevap **alıntılanır**. Chunk skoru ≥ 0.50 ve metin kısaysa
ham metin gösterilir, LLM hiç çalışmaz.
Avantajı: sıfır süre, sıfır hallucination riski, metin bozulmaz. Küçük yerel
modeller Türkçe'de hata yaptığı için bu bilinçli bir tercihtir.

**`generative`** (literatürde *abstractive*) — LLM birden fazla parçayı
**sentezleyip** yeni cümleler yazar.
Avantajı: birden çok kaynağı birleştirebilir, akıcıdır. Dezavantajı: yavaştır ve
hallucination riski taşır.

**`fallback_extractive`** — generative denendi, başarısız oldu, güvenli alıntıya
dönüldü. Bu bir **graceful degradation** mekanizmasıdır: sistem çökmek yerine
daha düşük kalitede ama doğru bir cevap verir. Production sistemlerinde çok
yaygın bir desendir.

Fallback'i tetikleyen durumlar `is_valid_answer()` içindedir: boş cevap, 30
karakterden kısa cevap, yalnızca etiket içeren cevap, **aşırı tekrar döngüsü**
(küçük modeller bazen aynı kelimeyi sonsuz üretir — Phi-3.5 benchmark'ında
gözlendi) ve retrieval kanıt bulduğu halde modelin "dokümanlarda yok" demesi
(**yanlış ret**).

---

## 9. Performans: neden ilk soru yavaş

- **Lazy loading:** LLM uygulama açılışında değil, ilk gerektiğinde yüklenir.
  Aksi halde `/stats` yazmak için 30 saniye beklenirdi.
- **Cold start:** ilk çağrı modeli diskten RAM'e alır, yavaştır (~31 sn).
  Sonrakiler **warm**'dır, hızlıdır (~4.7 sn). Benchmark'ta ikisi ayrı ölçülür;
  karıştırmak yanıltıcı sonuç verir.
- **Streaming:** model cevabı bir bütün olarak değil, token token üretir.
  Bunları geldikçe göstermek toplam süreyi değiştirmez ama **algılanan**
  gecikmeyi büyük ölçüde azaltır.

---

## 10. Atomiklik: reindex neden bu kadar dikkatli

Risk şu: reindex sırasında hata olursa eski indeks silinmiş, yenisi
yazılamamış olur ve arama tamamen bozulur.

Çözüm: yeni indeksin **tamamı bellekte hazırlanır**, sonra tek transaction'da
değiştirilir. Hata olursa rollback yapılır, eski indeks yerinde kalır. `chunks`
ve `source_manifest` birlikte değişir; biri güncel biri eski kalamaz.

**`source_manifest` ve SHA-256:** her dokümanın içeriğinin **hash**'i saklanır.
Hash, içeriğin parmak izidir. Neden dosya tarihine bakılmıyor? Çünkü dosya
tarihi yanıltıcıdır — kopyalama tarihi koruyabilir, tarih değişmeden içerik
değişebilir. Hash yalan söylemez. Bu sayede `/doctor` "indeks dokümanlardan geri
kalmış" diyebiliyor.

---

# İleride karar vermen gereken konular

Aşağıdakiler roadmap'te karşına çıkacak kavramlardır.

## 11. Sparse vs Dense retrieval

Şu an yapılan **dense retrieval**: yoğun (dense) vektörler, anlam bazlı arama.

**Sparse retrieval** ise TF-IDF / BM25 ailesidir. **BM25**, TF-IDF'in
geliştirilmiş halidir: kelime frekansı artı doküman uzunluğu normalizasyonu.

Ayrımın önemi:

| | Güçlü olduğu yer | Zayıf olduğu yer |
|---|---|---|
| **Dense** | Eşanlamlı ve farklı ifade ediş ("kimlik avı" ↔ "phishing") | Tam terim, kısaltma, sayı, özel isim |
| **Sparse (BM25)** | Tam eşleşme — `3-2-1`, `SHA-256`, `GDPR` | Eşanlamlıyı hiç bulamaz |

**Hybrid search** ikisini birleştirmektir. Karar noktası: iki farklı skor nasıl
birleştirilir?

- **RRF (Reciprocal Rank Fusion):** skorları değil *sıralamaları* birleştirir.
  Her liste için `1/(k + sıra)` hesaplanır ve toplanır. Skor ölçekleri farklı
  olduğu için genelde daha sağlamdır.
- **Ağırlıklı normalizasyon:** skorları aynı ölçeğe çekip toplar. Ayar gerektirir.

### Bu projede ne yapıldı

Hybrid search eklendi ve RRF seçildi. Gerekçe somut: cosine 0–1 arasında,
BM25'in üst sınırı yok. İki ölçeği toplamak için normalizasyon gerekir ve
normalizasyon aynı sorgunun aday havuzu içinde yapılmak zorunda olduğundan
skorlar **sorgular arası karşılaştırılamaz** hale gelir. Bu, "skoru şu eşiğin
altındaysa reddet" mantığını bozar. RRF bu kalibrasyon sorununu tamamen ortadan
kaldırır.

Bedeli: RRF skorun büyüklük bilgisini atar. 0.90 ile 0.89 arasındaki fark, 0.90
ile 0.30 arasındaki farkla aynı sayılır. Bu yüzden ikinci bir karar verildi:
**birleşik skor yalnızca sıralamada kullanılır**, eşik karşılaştırmalarında ve
kullanıcıya gösterimde cosine skoru kalır.

Ölçülen sonuç: `Recall@1` 0.78 -> 0.98, `MRR` 0.86 -> 1.00.

RRF'in `k` sabiti sezgisel değil. İki listenin birden gördüğü chunk, her k
değerinde tek listenin gördüğünün önüne geçer. k'nın belirlediği şey, bir listede
tepe yapan chunk ile iki listede de ortalarda kalan chunk arasındaki tercihtir:
küçük k tepeyi, büyük k istikrarı ödüllendirir. Bu korpusta 1 ile 60 arası bütün
değerler aynı sonucu verdi, yani ölçüm bu parametreyi ayırt edemedi — o yüzden
veriye uydurmak yerine gelenek olan 60 seçildi. Ölçemediğin bir parametreyi
"optimize etmek" gürültüye uydurmaktır.

### Öğretici yan etki: bir iyileştirme başka bir şeyi bozabilir

Hybrid search, kelime kanıtı kapısının kör noktasını açığa çıkardı. İkisi de
kelime örtüşmesine bakıyor; retrieval kelime örtüşmesinde güçlenince, kapı da
aynı yanlış eşleşmeye kandı ve bir tuzak soru sızdı. Eval bunu yakaladı.

Çözüm, kapsamayı **IDF ile ağırlıklandırmak** oldu: `güvenlik` gibi her chunk'ta
geçen bir kelime neredeyse hiçbir şey kanıtlamaz, `duvarı` gibi hiç geçmeyen bir
kelime ise sorunun cevaplanamadığının en güçlü işaretidir. Oran ikisini eşit
sayıyordu.

Alınacak ders: bileşenler birbirinden bağımsız değildir. Bir sinyali
güçlendirmek, o sinyale güvenen başka bir mekanizmanın varsayımını bozabilir.
Ölçüm olmadan bu sessizce geçer.

## 12. Bi-encoder vs Cross-encoder (reranking)

Bu ayrım, reranking kararının tamamıdır:

- **Bi-encoder** (şimdiki): soru ve chunk ayrı ayrı vektöre çevrilir, sonra
  karşılaştırılır. Chunk vektörleri **önceden hesaplanabildiği** için çok
  hızlıdır; 10.000 chunk'ta bile anlıktır.
- **Cross-encoder:** soru ve chunk **birlikte** modele verilir, model doğrudan
  bir alaka skoru üretir. Çok daha doğrudur çünkü ikisi arasındaki etkileşimi
  görür. Ama önceden hesaplanamaz; her soru için her chunk'ı modelden geçirmek
  gerekir.

Standart desen **iki aşamalıdır**: bi-encoder ile 20–50 aday çek (hızlı, geniş
ağ), cross-encoder ile o adayları yeniden sırala (yavaş ama yalnızca 20–50
tanede). Buna **reranking** denir. Karar verilecek şey: aday sayısı ve kabul
edilebilir ek gecikme.

## 13. Retrieval metrikleri

Sınıflandırmadaki precision/recall'un retrieval karşılıkları:

- **Recall@k:** doğru chunk ilk k sonucun içinde mi? En temel metrik. "Buldu mu?"
- **MRR (Mean Reciprocal Rank):** doğru sonuç kaçıncı sıradaydı? 1. ise 1,
  2. ise 1/2, 3. ise 1/3. **Sıralama kalitesini** ölçer; Recall@k'nın göremediği
  şeydir.
- **nDCG:** birden fazla alakalı sonuç varsa ve alakalılık dereceliyse kullanılır.
  Bu proje için şimdilik gerekli değil.

Bu metrikler artık projede ölçülüyor (`app/eval_metrics.py`). İlk ölçüm sonucu:

```text
Recall@1 : 0.6667
Recall@3 : 0.9444
Recall@5 : 1.0000
MRR      : 0.8333
```

Güncel değerler (hybrid search sonrası, 23 etiketli vaka):
`Recall@1 = 0.9783`, `Recall@3 = 1.0000`, `Recall@5 = 1.0000`, `MRR = 1.0000`.

Bu tablo, eski eval'in neden yetersiz olduğunun kanıtıdır. Eski eval `11/11 PASS`
diyordu çünkü yalnızca "doğru dosya geldi mi" diye soruyordu. Recall@1'in 0.6667
olması, doğru chunk'ın üç vakada **1. sırada olmadığını** gösteriyor — eski eval
bunu hiç göremiyordu.

`Recall@5 = 1.0000` ise en faydalı bilgi: doğru chunk **her zaman** ilk 5 adayın
içinde. Yani retrieval doğru parçayı buluyor, sadece yanlış sıralıyor. Bu, tam
olarak reranking'in (bölüm 12) çözmek için var olduğu durumdur — aday havuzu
sağlam, sıralama bozuk.

### Hard negative bulgusu

Projeye konusu dokümana yakın ama cevabı dokümanda olmayan 6 soru eklendi.
Altısı da mevcut `SIMILARITY_THRESHOLD = 0.20` eşiğini geçti. En çarpıcı sonuç:

| Soru | Skor | Cevabı dokümanda var mı? |
|---|---:|---|
| "Güvenlik duvarı kuralları nasıl yapılandırılmalıdır?" | 0.5985 | **Hayır** |
| "RAG nedir?" | 0.5570 | Evet |

Cevabı hiç bulunmayan bir soru, cevabı bulunan bir sorudan yüksek skor aldı.
Sonuç: **hiçbir tek eşik değeri bu ikisini ayıramaz.** 0.5985'i eleyen bir eşik
`RAG nedir?` sorusunu da eler.

Sebebi kavramsaldır ve bölüm 3'teki cosine tanımından çıkar: cosine similarity
**konu benzerliğini** ölçer, sorunun cevabının metinde bulunup bulunmadığını
değil. "Güvenlik duvarı kuralları" sorusu siber güvenlik dokümanına konu olarak
gerçekten benzer — model yanılmıyor, biz ondan yapamayacağı bir şey istiyoruz.

Bu yüzden yanlış pozitif problemi eşik ayarıyla çözülemez.

### Çözüm: kelime kanıtı

Eklenen savunma şu basit soruyu sorar: **sorunun kelimeleri, bulunan metinde
gerçekten geçiyor mu?**

- "Güvenlik duvarı" → hiçbir parçada geçmiyor → kanıt yok → model çalıştırılmaz
- "RAG" → geçiyor → kanıt var → normal akış

Bu sinyal cosine'dan bağımsızdır ve iki grubu net ayırır: alakalı sorular
0.80-1.00, cevabı bulunmayanlar 0.00-0.33.

Kapı LLM'den **önce** çalışır ve bir bonus getirir: sistem modele ancak gerçekten
kanıt varken gidiyor. Böylece "model bilmiyorum dedi ama biz ona inanmadık"
hatası da ortadan kalkıyor — çünkü artık modele sorulan her soruda kanıt var.

### Türkçe'nin zorluğu

Soruda "bağlantılar", metinde "bağlantı" yazıyor. Aynı kelime ama bilgisayar
farklı görüyor. Türkçe eklemeli bir dil: kök başta, ekler sonda.

Çözüm önek karşılaştırması: kısa kelime uzun kelimenin **başlangıcı** mı?
`bağlantı` → `bağlantılar` ✓

**Denenip elenen kısayol:** "kelimelerin ilk 5 harfini karşılaştır". Bu,
`sayısı` ile `sayısal`'ı eşleştiriyor — biri "adet" biri "numerik" demek.
Yanlış eşleşmeler iki grup arasındaki boşluğu 0.38'den 0.05'e düşürdü. Ders:
**"kanıt yok" demek üzereyken emin olmak, çok şey yakalamaktan önemlidir.**

**Bir de dilbilgisi tuzağı:** Türkçe'de sonu p/ç/t/k ile biten kelimeler ek
alınca yumuşar — `süreç` → `süre**c**i` (süreçi değil), `kitap` → `kita**b**ı`.
Buna *ünsüz yumuşaması* denir. Düz önek karşılaştırması bunu göremez çünkü son
harf değişmiştir; ayrıca ele alınması gerekti.

Kalan iş: cross-encoder (bölüm 12) ve cevap groundedness kontrolü.

## 14. Query rewriting (conversation history)

Takip sorusu probleminin özü. Kullanıcı "Peki dezavantajları neler?" derse bu
cümlenin embedding'i hiçbir şeye benzemez; "dezavantaj" kelimesi tek başına
anlamsızdır.

Çözüm: LLM'e önce konuşma geçmişini verip soruyu **bağımsız (standalone)** hale
getirtmek → "Çok faktörlü doğrulamanın dezavantajları neler?" — ve retrieval'a
bunu göndermek. Gerçek RAG sistemlerinin en çok hata yaptığı yer burasıdır ve
göründüğünden zordur.

## 15. ANN / HNSW (vector database tartışması)

Şu an **brute force** arama yapılıyor: her chunk'la tek tek karşılaştırma.
24 chunk'ta anlık, 100.000'de bile idare eder.

**ANN (Approximate Nearest Neighbor)**, milyonlarca vektörde kesin cevaptan
biraz taviz verip çok hızlanan yöntemlerin genel adıdır. **HNSW** en yaygın
olanıdır; vektörleri katmanlı bir graf yapısında düzenler. Vector database'lerin
(Qdrant, Pinecone) sattığı şey esas olarak budur.

Tam bu yüzden şu an gereksizdir. Anlamlı öğrenme yolu: chunk sayısını sentetik
olarak artırıp brute force'un **nerede kırıldığını ölçmek**. Kütüphane kurmadan
ANN'in neden var olduğu öğrenilir.

---

## 16. Toparlarsak

Sistem kavramsal olarak şudur:

> Dokümanları token sınırına uygun parçalara böl → her parçayı 384 boyutlu bir
> noktaya çevir → soruyu da aynı uzaya koy → cosine ile en yakın 3'ünü bul →
> eşiklerle zayıf kanıtı ele → kanıt güçlü ve kısaysa alıntıla, sentez
> gerekiyorsa yerel LLM'e yalnızca o kanıtı vererek yazdır → LLM bozuk cevap
> verirse güvenli alıntıya dön.

Yani **retriever + generator + bol miktarda güvenlik ağı**. Güvenlik ağları
(eşikler, fallback, atomik reindex, cevap kalite kontrolü) projenin en olgun
kısmıdır; çoğu tutorial RAG projesinde bunlar hiç yoktur.

Zayıf kısım nettir: **ölçme yeteneği**. Retrieval'ın ne kadar iyi olduğu 11 kaba
vakayla biliniyor. Roadmap'in ilk üç adımı tam olarak bunu düzeltir.
