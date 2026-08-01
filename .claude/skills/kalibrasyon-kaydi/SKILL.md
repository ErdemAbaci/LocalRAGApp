---
name: kalibrasyon-kaydi
description: Bu projedeki eşik, chunking, hybrid search ve kelime kanıtı kapısı kararlarının ölçüm geçmişi ve gerekçeleri. Eşiklere, retrieval'a, chunking'e, eval setine veya kelime kanıtı kapısına dokunmadan önce çağır.
---

# Kalibrasyon Kaydı

Bu dosya, `AGENTS.md` Bölüm 7 (Güncel Durum) ve Bölüm 12 (Öncelikli Roadmap)
içindeki ölçüm anlatımlarının ve mimari karar gerekçelerinin tam metnini tutar.
`AGENTS.md` yalnızca özetini içerir; ayrıntı ve gerekçe burada, birebir.

## Güncel Durum — tam ölçüm anlatımı

**Bu bölüm bir ölçüm günlüğüdür, durum raporu değildir.** Maddeler yazıldıkları
andaki korpus ve eval setine aittir ve bilinçli olarak güncellenmez; bir ölçümü
sonradan düzenlemek, o ölçümün hangi koşullarda yapıldığı bilgisini yok eder.
Her maddede geçerli olduğu korpus/vaka sayısı yazılıdır.

**Güncel sayılar için tek yetkili kaynak `AGENTS.md` Bölüm 7'dir.** Bir
maddedeki rakam oradakiyle çelişiyorsa `AGENTS.md` doğrudur ve buradaki rakam
tarihsel kayıttır.

Kayda geçmiş ölçümler:

- 12 kaynak dosya ve 217 chunk bulunuyor. En uzun chunk özel tokenlar dahil 128 tokendır; bu embedding modelinin sert sınırıdır ve `split_long_text` onu hiç aşmaz.
- Retrieval, indeks ve cevap kararı değerlendirmesi `39/39` başarılı; bilinen boşluk (`GAP`) kalmadı.
- Retrieval metrikleri: `Recall@1 = 0.9565`, `Recall@3 = 1.0000`, `Recall@5 = 1.0000`, `MRR = 0.9783` (23 etiketli vaka).
  `Recall@5` uzun süredir `1.0`; yani sorun doğru parçayı **bulmak** değil
  **sıralamak**tır ve hybrid search'ün iyileştirdiği yer tam olarak orasıdır.
- **Korpus 38'den 217 chunk'a çıkarıldı ve metrikler tavandan indi.** Sekiz yeni
  konu dosyası eklendi (ağlar, işletim sistemleri, veri yapıları, veritabanı,
  makine öğrenmesi, doğal dil işleme, yazılım mimarisi, dağıtık sistemler).
  `Recall@1` `0.9783`'ten `0.9565`'e, `MRR` `1.0000`'den `0.9783`'e indi.
  **Bu bir gerileme değil, ölçme yeteneğinin geri kazanılmasıdır.** Önceki
  korpusta `MRR = 1.0` idi; doygun bir metrik hiçbir iyileştirmeyi gösteremez,
  yalnızca gerilemeyi gösterebilir. Reranking gibi sıralama iyileştirmeleri
  ancak şimdi ölçülebilir hale geldi.
  Baseline bu yüzden güncellendi: 38 chunk'lık bir baseline ile 217 chunk'lık
  bir sonucu karşılaştırmak anlamlı bir regresyon sinyali üretmez, kalıcı bir
  sahte gerileme üretir.
  Yeni dokümanlar mevcut tuzak konularına (yedekleme sıklığı, parola uzunluğu,
  fidye yazılımı aracı, k-means küme sayısı, min-max formülü, güvenlik duvarı,
  sürüm kontrol komutu, dal isimlendirme, test kütüphanesi, kapsam aracı)
  bilinçli olarak değmez; değselerdi o vakalar sessizce geçersizleşirdi.
- Hard negative ölçümü kritik bir sınırı ortaya çıkardı: cevabı dokümanda hiç
  bulunmayan `hard_negative_firewall_rules` sorusu `0.5985` alırken, cevabı
  bulunan `rag_definition` `0.5570` alıyor. Yani **hiçbir tek `SIMILARITY_THRESHOLD`
  değeri bu ikisini ayıramaz.** Cosine similarity konu benzerliğini ölçer, cevap
  içerip içermediğini değil. Bu boşluğun çözümü eşik ayarı değil; BM25 terim
  kanıtı, cross-encoder ve groundedness kontrolüdür.
- **Hybrid search eklendi ve sıralama sorununu ölçülebilir biçimde düzeltti.**
  `app/sparse_search.py` BM25 ile kelime örtüşmesini ölçer; `app/retrieval.py`
  dense ve sparse sıralamaları RRF (`1/(k + sıra)` toplamı) ile birleştirir.
  Güncel ölçüm (23 etiketli vaka, 38 chunk): `Recall@1` 0.7826 -> 0.9783,
  `MRR` 0.8551 -> 1.0000. Manuel testte
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
- **Chunking ölçüldü ve `CHUNK_SIZE` 110'dan 128'e çıkarıldı.**
  `tools/chunking_analysis.py` yedi ayarı **indekse dokunmadan** ölçer: her ayar
  için dokümanları yeniden parçalar, embeddingleri bellekte üretir ve
  `retrieval.rank_chunks()` ile uygulamanın gerçek sıralama mantığını çalıştırır.
  Bu yüzden `app/retrieval.py` içinde skorlama veri erişiminden ayrıldı; aksi
  halde araç kendi sıralama kopyasını tutmak zorunda kalır ve ölçülen şey
  uygulamanın çalıştırdığı şey olmazdı.

  | boyut/overlap | chunk | R@1 | R@3 | R@5 | MRR |
  |---|---|---|---|---|---|
  | 60/12 | 77 | 0.5000 | 0.6364 | 0.6364 | 0.5530 |
  | 80/16 | 53 | 0.6818 | 0.7727 | 0.7727 | 0.7197 |
  | 110/20 | 47 | 0.8636 | 0.9773 | 1.0000 | 0.9318 |
  | 120/20 | 43 | 0.9318 | 1.0000 | 1.0000 | 0.9773 |
  | 128/12, 128/20, 128/30 | 38 | 0.9773 | 1.0000 | 1.0000 | 1.0000 |

  Üç okuma notu:
  - **60 ve 80 satırları retrieval kalitesi değildir.** `R@5` bile 1.0'ın
    altında, çünkü eval imzaları "bu terimlerin hepsi aynı chunk'ta" der; küçük
    chunk cevabı bölünce imza matematiksel olarak karşılanamaz. Bu, içerik
    imzası yönteminin chunk boyutu karşılaştırmasındaki yanlılığıdır ve büyük
    chunk lehine çalışır. 110/120/128'in üçünde de `R@5 = 1.0` olduğu için o üçü
    arasındaki fark gerçek sıralama farkıdır.
  - **Overlap'in ölçülebilir etkisi yok** (12/20/30 birebir aynı sonuç ve aynı
    chunk sayısı). Bölme paragraf bazlı olduğu için overlap nadiren devreye
    giriyor. 20 korundu.
  - 128, embedding modelinin sert sınırının kendisidir ve pay bırakmaz.
    `split_long_text` bütçeyi özel tokenlar dahil ölçtüğü için sınır aşılmaz;
    başka bir embedding modeline geçilirse bu değer yeniden kontrol edilmelidir.

  Yan etki: chunk büyümesi yalnızca sıralamayı değil **kapıyı da** iyileştirdi.
  Kelime kanıtı ayrım boşluğu `0.09`'dan `0.22`'ye çıktı, çünkü her parça kendi
  konusunun kelimelerini birlikte taşıyor ve tuzak sorular parçalı eşleşme
  toplayamıyor. Eşik aralığın ortasına, `0.67`'den `0.61`'e alındı.
- **Geçersiz bir hard negative bulundu ve alakalı vakaya çevrildi.**
  `hard_negative_coverage_target` ("Kod kapsamı yüzde kaç olmalıdır?") yanlış
  etiketlenmişti: doküman soruyu **yanıtlıyor**, bir yüzde vermiyor ama kapsamın
  hedef sayı olarak kullanılmamasını söylüyor. Doğru davranış reddetmek değil bu
  ifadeyi bulmaktır; vaka `coverage_not_a_target` adıyla alakalı vakaya çevrildi
  ve yerine gerçekten karşılığı olmayan `hard_negative_coverage_tool` eklendi.
  Ders: hard negative yazarken "doküman bu soruya kısmen bile cevap veriyor mu"
  diye kontrol et; vermiyorsa değil, veriyorsa vaka bozuktur.
- **Ölçülen eşleştirici sınırı: `yüzde` ~ `yüzden`.** İkisi de `yüz` kökünün
  çekimidir (bulunma ve ayrılma hali) ve ortak kökleri tam 5 karakterdir, yani
  `terms_match()` bunları eşleştirir. `min_prefix` 6'ya çıkarılarak
  kapatılamaz: ölçümde 6, `korunulur` ~ `korunmak`, `süreç` ~ `süreci` ve
  `aşamasında` ~ `aşama` gibi meşru eşleşmeleri kaybettiriyor. `sayısı` ~
  `sayısal` ile aynı sınıftır ve kabul edilmiş maliyettir; kapıyı taşıyan şey
  tek bir kelime değil IDF ağırlıklı toplamdır.
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
- ~~**Açık sorun: context kirlenmesi.**~~ **Çözüldü.** "Çok faktörlü doğrulama
  neden önemli?" sorusunda doğru chunk hem cosine hem BM25'te açık ara birinci
  olmasına rağmen `datamining.pdf`'ten üç parça da mutlak cosine eşiğini geçtiği
  için context'e giriyordu; model konuları karıştırıyor, doğrulama yakalayıp
  `fallback_extractive`e düşüyordu.
  Ölçüm ayırt edici sinyali gösterdi: context'e sızan yedi parçanın cosine'i
  `0.36`-`0.53` arasındayken IDF ağırlıklı kelime kanıtı en fazla `0.267`'ydi.
  `CONTEXT_TERM_EVIDENCE_MIN = 0.30` eklendi ve **yalnızca ikinci ve sonraki
  sıralara** uygulanır; birinci sıra koşulsuz girer, çünkü elimizdeki en iyi
  cevap adayı odur ve onu kanıt şartıyla elemek retrieval kararını ikinci kez
  sorgulamak olurdu.
  Sonuç: beklenen kaynak dışından gelen parça sayısı `7 -> 0`. Retrieval
  metrikleri değişmedi (filtre sıralamaya değil context'e dokunur), eval
  `39/39` kaldı ve kelime kanıtı ayrım boşluğu `0.09`'da sabit kaldı.
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
- Unit testler `242/242` başarılı. Kapsamın ayrıntısı testlerin kendisinden
  okunur; burada tekrarlanmaz.
- Gerçek benchmark'ta `phi-4-mini` 3/3 geçerli cevap ve %89 terim kapsamı;
  `phi-3.5-mini` 2/3 geçerli cevap ve %56 kapsam verdi. Varsayılan model bu
  nedenle `phi-4-mini` olarak korundu.
- `phi-4-mini` ile yapılan gerçek generative test doğru ve kaynakla uyumlu cevap
  verdi. İlk model yüklemeli generation yaklaşık 39 saniye sürdü; bu beklenen bir
  cold-start davranışıdır.
- Retrieval bazı hard negative'lerde tamamen alakasız context seçiyor:
  "Parola en az kaç karakter olmalıdır?" sorusunun en iyi eşleşmesi, kategorik
  verilerin 0/1'e dönüştürülmesini anlatan `datamining.pdf` chunk'ı. Kapı bu
  vakaları yakalar, ama sıralamanın kendisi hâlâ yanlış.

## Kelime kanıtı kapısından groundedness'a geçiş — tam kayıt

**Kapı bir kalibrasyon sorunu değil yöntem sorunuymuş.** Eval seti 36 vakadan
125 vakaya (23 -> 112 etiketli) çıkarılınca `tools/term_evidence_analysis.py`
şunu gösterdi:

    EŞLEŞTİRİCİ  oran alk  oran tzk  BOŞLUK   ağr alk  ağr tzk  BOŞLUK
    prefix5         0.25      0.60    -0.35     0.12     0.58    -0.46
    common4         0.33      0.75    -0.42     0.27     0.58    -0.31
    common5         0.33      0.67    -0.33     0.24     0.59    -0.35
    common6         0.33      0.60    -0.27     0.09     0.56    -0.47
    kök5-3          0.33      0.75    -0.42     0.27     0.65    -0.39   <- mevcut
    kök5-4          0.33      0.75    -0.42     0.26     0.58    -0.32

    Hiçbir ayar iki grubu ayırmıyor.

Boşluk daralmadı, **işaret değiştirdi**. Meşru soru 0.27, tuzak 0.65. Daha önce
raporlanan 0.02, yalnızca kapının kolay tarafını ölçen 36 vakalık setin ürettiği
bir yanılsamaymış. Ders: bir eşiğin "dar ama çalışıyor" görünmesi, ölçüm setinin
zor vakaları içermediği anlamına gelebilir. Eşiği savunmadan önce seti sorgula.

**Seçilen mimari: (a) kapıyı cevaba taşı.** (b) embedding kapısı elendi çünkü
cosine'in hard negative'leri ayıramadığı iki kez ölçülmüştü
(`firewall_rules` 0.5985 > `rag_definition` 0.5570). (c) cross-encoder elendi
çünkü aynı model reranking'de de gerekecek; ölçmeden bağımlılık eklemek yerine
ikisini birlikte ele almak daha verimli. Gerekçe: cevap kaynağın kelimeleriyle
yazılır, soru kullanıcının kelimeleriyle; `önlenir` ~ `önler` sınıfındaki
uyuşmazlık tanım gereği ortadan kalkar.

### Ölçüm yöntemi ve ilk hatası

Model çıktısı deterministik olmadığı için üç vekil grup ölçüldü: DAYANAKLI
(context'in kendi cümleleri), PARAFRAZ (vakanın sorusu — aynı içeriğin
kullanıcı kelimeleriyle yazılmış hâli, yani meşru cevabın en kötü durumu),
DAYANAKSIZ (başka dokümandan cümleler).

İlk ölçüm "ayrım yok, cross-encoder gerekir" dedi ve **yanlıştı**. Sebep:
vekil metinler tek cümleydi ve tek cümlelik bir metinde desteklenen cümle oranı
yalnızca 0.00 veya 1.00 olabilir. Uç değerler bu yüzden anlamsızdı; asıl sinyal
ortalamalardaydı (parafraz 0.97, dayanaksız 0.046). **Genel ders: bir oranı
kalibre ederken vekil metnin, oranın çözünürlüğünü taşıyacak kadar uzun
olduğundan emin ol.**

Cümle seviyesine geçilince ayrım net çıktı:

    DAYANAKLI  n=1179  min 1.00  ort 1.00
    PARAFRAZ   n=104   min 0.38  ort 0.91
    DAYANAKSIZ n=324   min 0.00  ort 0.20  max 0.67

    eşik   parafraz geçen   dayanaksız geçen
    0.34      100.0%             18.2%
    0.50       98.1%              4.6%
    0.60       96.2%              0.9%   <- seçilen
    0.67       92.3%              0.0%
    1.00       63.5%              0.0%

Cevap seviyesinde: dayanaklı cevap min 1.0000, uydurma cevap ort 0.0463
max 0.3333. `GROUNDEDNESS_THRESHOLD = 0.50` seçildi.

Aralığın ortası (0.67) **alınmadı**, ki bu projenin başka eşiklerdeki kuralından
bilinçli bir sapmadır. Sebep: buradaki üst sınır 1.0, birebir kopyalanmış
metinden gelen dejenere bir değerdir. Ona göre ortalamak gerçek parafraz
cevaplara değil kopyaya göre kalibre etmek olurdu. Ortadaki kural, iki ucu da
gerçek veriyle ölçülmüş aralıklar için geçerlidir.

Sıralamanın yanlış dokümanı getirdiği 4 vaka ölçümden dışlandı; orada düşük
groundedness doğru davranıştır ve eşiği yapay olarak sıfıra çeker.

### Ön kapının yeni görevi

`TERM_EVIDENCE_THRESHOLD` 0.675 -> 0.21. Kapı silinmedi, "cevap var mı"
sorusundan "bu soru bu korpusun konusu mu" sorusuna indirildi:

    alakalı vakaların en düşüğü  : 0.2418  (phrasing_horizontal_scaling_choice)
    kapsam dışının en yükseği    : 0.1755  (out_of_scope_cooking)
    boşluk                       : +0.0663

11 hard negative (0.23-0.65) artık bu kapıyı geçip modele ulaşır. Bu kasıtlıdır.
Uyarı: eşik yalnızca iki kapsam dışı vakayla kalibre edildi; boşluk dar.

### Birlikte kaldırılması ZORUNLU olan iki bağlantı

Bunlar tek başına yapılırsa sistem bozulur:

1. **`false_no_evidence` koruması.** `app/llm.get_answer_validation_error()`
   modelin `NO_EVIDENCE_ANSWER` üretmesini geçersiz sayıp kaynak metne
   dönüyordu. Varsayımı "arama doğru, LLM inatçı"ydı ve kapı modelin önünde
   durduğu sürece savunulabilirdi. Kapı gevşeyince hard negative'ler modele
   ulaşır ve varsayım tersine döner: arama yanlış, model haklı. Kaldırılmasaydı
   modelin doğru reddi silinip alakasız metin gösterilecekti — bu hata
   `hard_negative_ransomware_tool` üzerinde daha önce ölçülmüştü.
2. **Extractive kısayolunun kanıtsız çalışması.** `should_use_extractive_answer()`
   chunk metnini doğrudan cevap yapar; ne modelden ne groundedness'tan geçer.
   Ön kapı indirilince iki hard negative tam buradan sızdı
   (`hard_negative_git_revert_command` kapsama 0.34,
   `hard_negative_coverage_tool` 0.49) ve alakasız chunk metni cevap oldu.
   `EXTRACTIVE_TERM_EVIDENCE_MIN = 0.675` eklendi — ön kapının ESKİ değeri,
   burada hâlâ savunulabilir çünkü ödünleşim yön değiştirir: kısayol "bu chunk
   cevabın kendisidir" gibi güçlü bir iddiadır ve yanlış reddin bedeli
   cevapsızlık değil, yalnızca üretken yola düşmektir.

3. **`fallback_extractive` yolunun kanıtsız çalışması.** Bu üçüncüsü ilk turda
   ATLANDI ve gerçek modelle yapılan manuel testte sızdı: "Fidye yazılımının
   şifrelediği dosyaları çözmek için hangi araç kullanılır?" sorusuna model
   geçersiz bir üretim yaptı, akış kaynak metnine döndü ve alakasız bir
   yedekleme cümlesi cevap olarak gösterildi (`fallback_extractive`,
   kapsama 0.32).
   Atlama gerekçesi hatalıydı: "fallback kaynak metnini döndürür, inşası gereği
   dayanaklıdır" doğru ama yetersiz. **Dayanaklı olmak ALAKALI olmak değildir**
   — kabul edilen kör noktanın ta kendisi. `extractive` ile aynı iddiayı yapan
   bu yol aynı kanıt şartına bağlandı; kanıt yetersizse düşecek başka yol
   olmadığı için `no_evidence` döner.
   Eval'e üçüncü bir dal eklendi (`FailingLLM`): model bozuk üretirse sonuç
   `no_evidence` olmalı. Bu dal o ana kadar hiç sınanmamıştı.

**Genel ders:** bir kapıyı gevşetmeden önce, o kapının varlığına sessizce
güvenen başka yolları ara. Üçü de kapı yerindeyken zararsız görünüyordu ve
üçüncüsü ancak gerçek modelle çalışınca ortaya çıktı — deterministik eval
modelin bozuk üretim yapacağını kendiliğinden denemiyordu.

**İkinci ders:** manuel test bir formalite değil. Kabul edilmiş bir sınırın
("dayanaklı ≠ alakalı") hangi somut yolda patlayacağını ölçüm değil kullanım
gösterdi.

### Sonuç

Eval `118/128 -> 124/128`. Kapının reddettiği 6 meşru vakanın 6'sı geçti;
13 `not_found` vakasının 13'ü hâlâ reddediliyor. Unit test `242 -> 262`.
Retrieval metrikleri değişmedi (bu değişiklik sıralamaya dokunmaz).
Kalan 4 başarısız vaka sıralama hatasıdır ve reranking'in konusudur.

### Kabul edilen sınır

Groundedness "cevap context'e dayanıyor mu" diye sorar, "cevap soruyu
yanıtlıyor mu" diye değil. Retrieval alakasız ama gerçek bir metin getirir ve
model onu özetlerse cevap dayanaklı çıkar. O durumda tek savunma modelin kendi
reddidir; yani kararın bir kısmı deterministik olmaktan çıktı. Eval bunu
ölçemez, bu yüzden bizim tarafımızın sözleşmesini iki dalda sınar (model
reddederse red nihai, model uydurursa `ungrounded`). Cross-encoder'a geçilirse
bu boşluk da kapanır.

Ölçüm aracı: `tools/groundedness_analysis.py`.

## Roadmap madde 2 — Yanlış pozitif savunması, tam gerekçe

**Yanlış pozitif savunması — büyük kısmı tamamlandı.** Kelime kanıtı kapısı
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
- **Kelime kanıtı eşiği kalibre edilebilir olmaktan çıktı — sıradaki iş
  budur.** `TERM_EVIDENCE_THRESHOLD = 0.675` ve ayrım boşluğu `0.02`.
  217 chunk'lık korpusta tek bir turda şu zincir yaşandı: tuzak sızıntısı
  eşiği `0.61`'den `0.675`'e çıkarttı, aynı yükseltme iki meşru soruyu kesti
  (`Kilitlenme nedir ve nasıl önlenir?`, `Aşırı öğrenme nasıl anlaşılır?`),
  yanlış retleri düzeltmek boşluğu `0.05`'ten `0.02`'ye indirdi. Kapıyı bir
  uçtan sıkıştırmak diğer ucu açıyor. Stopword listesi de üç kez aynı sebeple
  büyüdü (`önemli`, `arasındaki`, `önlenir`) ve her seferinde tetikleyen şey
  ölçüm seti değil gerçek bir kullanıcı sorusu oldu.
  **Eşiği bir daha kovalama.** Sıradaki adım groundedness kontrolüdür:
  "soru kelimeleri context'te geçiyor mu" yerine "üretilen cevap context'e
  dayanıyor mu" sorusuna geç. Ölçüm: `tools/term_evidence_analysis.py`.

  *Mimari karar — bir sonraki oturumda yüksek effort ile ele alınacak.*
  Değerlendirilen üç seçenek:
  (a) **Groundedness, kapıyı cevaba taşı — öne çıkan.** Kelime kapısı
      silinmez, ucuz ön eleme olarak kalır ve eşiği DÜŞÜRÜLÜR; asıl karar
      üretilen cevabın context'le örtüşmesine bakılarak verilir. Gerekçe:
      cevap kaynağın kelimeleriyle yazılır, soru kullanıcının kelimeleriyle;
      `önlenir` ~ `önler` sınıfındaki uyuşmazlık tanım gereği ortadan kalkar.
      Ayrıca şu anki kapının hiç göremediği uydurma cevabı da yakalar.
      Bedeli: kapsam dışı sorularda da LLM çalışır, o yol yavaşlar.
  (b) Kapıyı embedding benzerliğine taşımak. Kelime uyuşmazlığını çözer ama
      cosine'in hard negative'leri ayıramadığı zaten ölçüldü; tek başına
      yetersiz.
  (c) Cross-encoder ile kapı. En doğru, en pahalı, ek model indirmesi
      gerektirir.
  Karar verilmesi gerekenler: kapının akışta nereye gireceği, latency
  bütçesi, örtüşme ölçütü (n-gram mı, cümle bazlı mı) ve eşiğin nasıl
  kalibre edileceği. Kalibrasyonun kendisi de artık 71 etiketli vaka ve
  11 hard negative üzerinden ölçülebilir.

## Roadmap madde 3 — Hybrid search, tam gerekçe

**Hybrid search.** ~~Yapıldı.~~ `app/sparse_search.py` (BM25) +
`app/retrieval.py` (RRF). Karar: birleşik skor yalnızca sıralamada kullanılır,
kapı skoru cosine kalır. Sonuç: `Recall@1` 0.7826 -> 0.9783, `MRR` 0.8551 -> 1.0000.
Yan etkisi kelime kanıtı kapısını IDF ağırlıklarına taşımayı zorunlu kıldı;
ayrıntı yukarıdaki Güncel Durum bölümünde. SQLite FTS5 tercih edilmedi:
`unicode61` tokenizer'ı Türkçe stemming yapmaz ve `remove_diacritics`
seçenekleri `ı/i`, `ş/s` ayrımını bozar; ayrıca `normalize_text()`'ten sapan
ikinci bir normalizasyon yolu açardı.

## Roadmap madde 4 — Reranking, tam kayıt (NEGATİF SONUÇ)

**Sonuç: ölçüldü, kötüleştirdi, kapatıldı.** `USE_RERANKER = False`.
Kod (`app/reranker.py`), testleri (`tests/test_reranker.py`) ve ölçüm aracı
(`tools/reranker_analysis.py`) repoda duruyor. Silme; negatif sonucun değeri
ölçümün tekrar edilebilmesindedir.

### Ön şart nasıl karşılandı

İlk planda not düşülmüştü: 23 etiketli vakada tek bir vaka `1/23 = 0.043`
oynatır ve bu gürültü beklenen reranker kazancından büyüktür. Bu yüzden önce
eval seti 112 etiketli vakaya çıkarıldı; tek vakanın etkisi `0.009`'a indi.
Ancak bundan sonra ölçüm anlamlı hale geldi.

### Ölçüm

`tools/reranker_analysis.py`, 112 etiketli vaka, 217 chunk,
`BAAI/bge-reranker-base` (XLM-RoBERTa base, 278M parametre):

| ayar | R@1 | R@3 | R@5 | MRR | sn/soru |
|---|---|---|---|---|---|
| kapalı | **0.8973** | 0.9911 | 0.9911 | **0.9464** | 0.060 |
| havuz=5 | 0.9018 | 0.9821 | 0.9911 | 0.9457 | 0.130 |
| havuz=10 | 0.8393 | 0.9732 | 0.9821 | 0.9085 | 0.193 |
| havuz=15 | 0.8393 | 0.9554 | 0.9911 | 0.9068 | 0.260 |
| havuz=20 | 0.8393 | 0.9554 | 0.9821 | 0.9046 | 0.325 |
| havuz=30 | 0.8214 | 0.9375 | 0.9732 | 0.8894 | 0.465 |

8 vaka iyileşti, 14 vaka kötüleşti.

**Monotonluk kritik bir ayrımdır.** Tek bir havuz değerinde bozulma olsaydı
"ayar sorunu" denebilir ve tarama sürdürülebilirdi. Eğrinin tamamı düşüyorsa
sinyalin kendisi zayıf demektir ve tarama sürdürmek zaman kaybıdır.
`havuz=5`'te `R@1`'in hafifçe yükselmesi (0.8973 -> 0.9018) tek başına gürültü
sınırındadır ve `MRR` aynı noktada düşmüştür.

### Modelin bozuk olmadığı nasıl doğrulandı

Negatif sonucu raporlamadan önce "acaba bende bir hata mı var" sorusu
kapatılmalıydı. Model doğrudan test edildi:

| soru | metin | skor |
|---|---|---|
| Kilitlenme nedir? | doğru tanım cümlesi | 0.9992 |
| Kilitlenme nedir? | muzun potasyum içerdiği | 0.0000 |
| What is a deadlock? | doğru İngilizce tanım | 0.8308 |
| What is a deadlock? | muz cümlesi | 0.0000 |

Yani model Türkçede ayırt ediyor ve entegrasyon doğru. Bozulma yalnızca gerçek
chunk'larda oluyor. `ml_train_val_test_split` sorusunda doğru chunk 0.8392
alırken, dağıtık sistemlerdeki uzlaşıyla ilgili tamamen alakasız bir chunk
0.9981 alıyor. `phrasing_db_query_optimization_myth` sorusunda doğru chunk
0.0028 alıyor.

### Teşhis

Chunk'larımız 128 tokenda kesilen **kırıntılar**, kendi başına ayakta duran
paragraflar değil. 128 embedding modelinin sert sınırıdır; cross-encoder 512
token okuyabilirdi, yani ona gereksiz yere elimizin körü veriliyor.
Cross-encoder'lar paragraf seviyesinde alaka için eğitilir ve cümlenin
ortasında başlayan bir parça yeterli malzeme vermiyor.

### Denenmemiş iki iyileştirme

Teşhisin işaret ettiği yönde, yapılmadı:

1. **Reranker'a chunk+komşu penceresi ver.** Komşular zaten
   `NEIGHBOR_CHUNK_RADIUS` ile alınıyor, sadece modele geçirilmiyor.
2. **Reranking'i RRF ile birleştir.** Şu an cross-encoder ilk aşamanın sırasını
   **tamamen siliyor**; bu yüzden `phrasing_db_query_optimization_myth` 1.
   sıradan ilk 5'in dışına düştü. Hybrid'in kararı iyi (`MRR` 0.9464), onu
   silmek yerine birleştirmek felaket düşüşleri engellerdi — tıpkı BM25 ile
   cosine'i birleştirdiğimiz gibi.

Yapılmama gerekçesi: iyileştirilecek alan çok dar. `Recall@3` kapalıyken
0.9911, yani reranking'in kazanabileceği en fazla şey eval'de kalan 4 vakadır
(%3) ve gerçekçi beklenti bunun yarısıdır. Bu kazanç için 1-2 GB model, her
soruya +0.2-1 saniye ve bakımı olan bir bileşen taşınıyor.

### Genel dersler

- **Pahalı bileşen her zaman iyileştirmez.** Bu, RAG'de en sık yapılan
  hatalardan biridir ve burada okuyarak değil ölçerek görüldü.
- **Sıra kendini doğruladı.** Eval önce güçlendirildiği için bu kötüleşme
  görülebildi. Ölçme yeteneği kurulmasaydı cross-encoder eklenir ve
  iyileştirdiği varsayılırdı — literatür de öyle söylüyor.
- **Negatif sonucu tek modele dayandırmamaya çalış, ama maliyetini de gör.**
  `bge-reranker-v2-m3` (568M, 2.1 GB) ikinci kanıt için indirildi ve ölçüm
  başlatıldı; makine ısındığı için durduruldu. Ölçüm yükü gerçek kullanım
  yükünün yüzlerce katıdır (112 soru × 5 ayar, kesintisiz), bunu planlarken
  hesaba kat. İkinci model diskte duruyor; küçük ölçekli bir doğrulama
  (tek havuz, 30 vaka) hâlâ yapılabilir.
- **Bayrağın varsayılanına bağlı test yazma.** `USE_RERANKER` kapatıldığında
  6 test kırıldı; hepsi bayrağın açık olduğunu varsayıyordu. Testler artık
  `use_reranker=True`'yu açıkça geçiyor.
- **Kapalı olmak arıza değildir.** `/doctor` içinde `USE_RERANKER = False`
  durumu `ok` olarak raporlanır; ölçülmüş bir kararı her çalıştırmada uyarı
  olarak basmak gürültüdür. Uyarı yalnızca "açık ama model indirilmemiş"
  durumuna saklanır.
