SIMILARITY_THRESHOLD = 0.20
CONTEXT_SCORE_THRESHOLD = 0.35
CONTEXT_RELATIVE_SCORE_MARGIN = 0.20
TOP_K = 3
NEIGHBOR_CHUNK_RADIUS = 1
MAX_CONTEXT_CHUNKS = 5

NO_EVIDENCE_ANSWER = "Bu bilgi verilen dokümanlarda yok."

# Kelime kanıtı eşikleri. Değerler `tools/term_evidence_analysis.py` ölçümünden
# gelir. Kapsama kelime sayısına göre değil IDF ağırlığına göre hesaplanır;
# ölçülen ayrım boşlukları (alakalı min - tuzak max, pozitif olan ayırıyor):
#
#                       oran    ağırlıklı
#   ortak kök 4        -0.08        -0.16
#   ortak kök 5         0.00         0.02
#   ortak kök 6        -0.17        -0.17
#   kök 5 + kısa 3      0.05         0.21   <- seçilen
#   kök 5 + kısa 4     -0.08        -0.13
#
# İki bulgu:
#   - Ağırlık tek başına yetmedi. `common5 / ağırlıklı` boşluğu yalnızca 0.02
#     çıktı, çünkü ayırt edici kelimenin eksik olması ile eşleştiricinin onu
#     kaçırması aynı görünüyordu: "avından" korpusta hiç eşleşmediği için
#     "duvarı" kadar eksik sayılıyordu, oysa "kimlik avı" dokümanda var.
#   - Eşleştiriciye kısa kök kuralı eklenince ("avı", "avından"ın tükenen
#     öneki) yanlış eksiklik ortadan kalktı ve ağırlık asıl işini yaptı:
#     boşluk 0.21.
#
# Kalibrasyon üç kez yenilendi. IDF ağırlıkları korpustan geldiği için hem
# doküman eklemek hem chunk boyutunu değiştirmek bu eşiği doğrudan etkiler:
#
#   24 chunk (110 token), 20 vaka : tuzak max 0.60, alakalı min 0.82, boşluk 0.21
#   47 chunk (110 token), 35 vaka : tuzak max 0.63, alakalı min 0.72, boşluk 0.09
#   38 chunk (128 token), 36 vaka : tuzak max 0.50, alakalı min 0.72, boşluk 0.22
#  217 chunk (128 token), 36 vaka : tuzak max 0.65, alakalı min 0.70, boşluk 0.05
#
# İkinci satırda boşluk daraldı çünkü korpus büyüdükçe soru kelimelerinin bir
# kısmı kaçınılmaz olarak başka dokümanlarda da geçiyor. Üçüncü satırda yeniden
# açıldı: chunk büyüyünce her parça kendi konusunun kelimelerini birlikte
# taşıyor ve tuzak sorular parçalı eşleşme toplayamıyor. Chunking kararının
# yalnızca sıralamayı değil kapıyı da iyileştirdiği burada görülüyor.
#
# Dördüncü satır eğilimi doğruladı: korpus 38'den 217 chunk'a çıkınca boşluk
# 0.22'den 0.05'e düştü. Sebep mekanizmada: korpus büyüdükçe herhangi bir soru
# kelimesinin metinlerden birinde tesadüfen geçme olasılığı artıyor.
# `hard_negative_branch_naming` bu yüzden sızdırdı; üç harfli `dal` kelimesi
# kısa kök kuralıyla `dalgaları`na eşleşti (`sayısı` ~ `sayısal` ile aynı sınıf)
# ve kapsamayı 0.65'e taşıdı. Sorunun asıl ayırt edici kelimesi
# `isimlendirmesinde` hiçbir yerde geçmiyor ama tek başına kapıyı tutamadı.
#
# 0.675 seçildi; aralığın ortasıdır.
# Eşiği tuzak maksimumuna EŞİT seçmek daha önce iki kez sızdırdı (0.50 ve 0.60);
# eşitlik geçer, bu yüzden daima aralığın ortası alınır.
#
# DURDURMA NOKTASI: boşluk 0.02'ye indi (güvenli aralık 0.65-0.68).
# Bu eşik artık ölçüm gürültüsü kadar dar bir şeride sıkışmıştır ve
# KALİBRE EDİLEBİLİR OLMAKTAN ÇIKMIŞTIR. Bir sonraki doküman eklemesinde
# eşiği kovalama; groundedness kontrolüne geç.
#
# Kanıt zinciri, eşiğin neden kovalanamayacağını gösteriyor:
#   1. 217 chunk'a çıkınca `hard_negative_branch_naming` sızdı -> eşik
#      0.61'den 0.675'e çıkarıldı.
#   2. Aynı yükseltme iki MEŞRU soruyu kesti ("Kilitlenme nedir ve nasıl
#      önlenir?", "Aşırı öğrenme nasıl anlaşılır?").
#   3. Yanlış retler soru kalıbı kelimeleri stopword'e eklenerek düzeltildi,
#      ama bu düzeltme boşluğu 0.05'ten 0.02'ye indirdi.
# Yani kapıyı bir uçtan sıkıştırmak diğer ucu açıyor. Bu bir ayar sorunu
# değil, yöntemin sınırıdır: oran tabanlı kapı, ayırt edici kelimenin
# yokluğunu ayırt edici OLMAYAN kelimelerin varlığıyla dengeleyebilir ve
# korpus büyüdükçe tesadüfi eşleşme olasılığı arttığı için zayıflar.
#
# Ayrıca stopword listesi üç kez aynı sebeple büyüdü (`önemli`, `arasındaki`,
# `önlenir`). Her seferinde tetikleyen şey ölçüm seti değil gerçek bir
# kullanıcı sorusu oldu; yani liste, bilinmeyen bir kuyruğu elle kovalıyor.
#
# SONUÇ: kapı görev değiştirdi. 112 etiketli vakayla yapılan ölçüm yukarıdaki
# öngörüyü doğruladı ve daha kötüsünü gösterdi — ayrım boşluğu yalnızca daralmadı,
# İŞARET DEĞİŞTİRDİ:
#
#   EŞLEŞTİRİCİ  oran alk  oran tzk  BOŞLUK   ağr alk  ağr tzk  BOŞLUK
#   prefix5         0.25      0.60    -0.35     0.12     0.58    -0.46
#   common5         0.33      0.67    -0.33     0.24     0.59    -0.35
#   kök5-3          0.33      0.75    -0.42     0.27     0.65    -0.39   <- mevcut
#   kök5-4          0.33      0.75    -0.42     0.26     0.58    -0.32
#
# Meşru bir soru 0.27, bir tuzak 0.65 alıyor. Hiçbir eşik bu iki grubu ayıramaz;
# 0.02'lik boşluk 36 vakalık setin ürettiği bir yanılsamaymış. Bu, kapıyı bir
# kalibrasyon sorunu olmaktan çıkarıp yöntem sorunu yapar.
#
# Karar: kapı SİLİNMEDİ, görevi daraltıldı. Artık "cevap var mı" değil yalnızca
# "bu soru bu korpusun konusu mu" sorusuna bakar; asıl karar üretilen cevabın
# context'e dayanıp dayanmadığına bakan `app/groundedness.py`e taşındı.
#
# Yeni eşik bu daraltılmış göreve göre ölçüldü (`tools/groundedness_analysis.py`):
#
#   alakalı vakaların en düşüğü  : 0.2418  (phrasing_horizontal_scaling_choice)
#   kapsam dışının en yükseği    : 0.1755  (out_of_scope_cooking)
#   boşluk                       : +0.0663 -> güvenli aralık 0.1755-0.2418
#
# 0.21 seçildi; aralığın ortasıdır. 11 hard negative vakanın hepsi (0.23-0.65)
# artık bu kapıyı geçip LLM'e gidecek — bu kasıtlıdır, onları groundedness
# kontrolü reddedecek. Bedeli kapsam dışı soruda latency: 0.1 saniye yerine
# generation süresi kadar.
#
# UYARI: bu eşik yalnızca iki kapsam dışı vakayla kalibre edildi ve boşluk
# 0.066 ile dardır. Kapsam dışı vaka ekledikçe yeniden ölç. Ama bu kapının
# yanlış kararı artık cevabı belirlemiyor; yanlış geçirdiğini groundedness
# yakalar, yanlış reddettiğinde ise soru gerçekten korpusun konusu dışındadır.
#
# Doküman eklendiğinde veya eval seti büyüdüğünde yeniden ölç; ölçmeden
# değiştirme.
TERM_EVIDENCE_THRESHOLD = 0.21
TERM_EVIDENCE_MIN_PREFIX = 5
TERM_EVIDENCE_MIN_SHORT_ROOT = 3
TERM_EVIDENCE_MIN_TERM_LENGTH = 3

# Hybrid search. Dense (cosine) sıralaması tek başına yetersiz kaldı: ölçümde
# Recall@1 = 0.60 iken Recall@5 = 1.00 çıktı, yani doğru parça aday havuzunda
# var ama en üstte değil. Sparse (BM25) sinyali kelime örtüşmesini ölçer ve
# cosine'in kaçırdığı birebir terim eşleşmesini yakalar. Güncel ölçüm
# (23 etiketli vaka, 38 chunk): Recall@1 0.7826 -> 0.9783, MRR 0.8551 -> 1.0000.
# "Kimlik avından nasıl korunulur?" sorusunda cevabı içeren chunk 4. sıradan
# 1. sıraya çıktı.
#
# Birleşik skor YALNIZCA sıralama için kullanılır. Kapı ve kullanıcıya gösterilen
# skor cosine olarak kalır; aksi halde SIMILARITY_THRESHOLD,
# CONTEXT_SCORE_THRESHOLD, CONTEXT_RELATIVE_SCORE_MARGIN ve
# EXTRACTIVE_SCORE_THRESHOLD'un tamamı yeni bir ölçeğe göre yeniden kalibre
# edilmek zorunda kalırdı. Tek değişkeni izole tutmak ölçümü mümkün kılıyor.

# İkinci ve sonraki sıraların context'e girebilmesi için gereken en düşük
# IDF ağırlıklı kelime kanıtı. Cosine eşiği konu benzerliğini ölçer ve alakasız
# dokümandan gelen parçayı da geçirir; ölçümde context'e sızan yedi parçanın
# cosine'i 0.36-0.53 arasındayken kelime kanıtı en fazla 0.267'ydi.
# Birinci sıra bu kuralın dışındadır.
CONTEXT_TERM_EVIDENCE_MIN = 0.30

USE_HYBRID_SEARCH = True

# BM25 doygunluk ve uzunluk normalizasyonu sabitleri. Başlangıçta literatür
# geleneği olarak alındı (k1 genelde 1.2-2.0, b genelde 0.75); 38 chunk'lık
# korpus bu iki parametreyi ayırt edemiyordu. 217 chunk / 71 etiketli vakada
# `tools/hybrid_search_analysis.py` ile grid tarandı (MRR, RRF_K=2 sabit):
#
#   k1 \ b    0.00    0.25    0.50    0.75    1.00
#   0.9      0.9624  0.9648  0.9648  0.9718  0.9648
#   1.2      0.9624  0.9648  0.9648  0.9718  0.9648
#   1.5      0.9624  0.9648  0.9648  0.9718  0.9648   <- mevcut
#   1.8      0.9624  0.9648  0.9718  0.9718  0.9648
#   2.0      0.9624  0.9648  0.9718  0.9718  0.9648
#
# Sonuç: `k1` ekseni 0.9-2.0 arasında hiçbir fark üretmiyor. Tek ayırt edici
# eksen `b` ve mevcut 0.75 zaten en iyi grupta. Bu beklenen davranış: `k1`
# terim tekrarının doygunluk hızını ayarlar, ama 128 tokenlık chunk'larda bir
# terim zaten birkaç kez geçiyor, dolayısıyla doygunluk eğrisinin şekli
# sıralamayı değiştirmiyor. `b` uzunluk cezasını ayarlar ve chunk uzunlukları
# tokenizer sınırına göre değiştiği için etkisi ölçülebilir kalıyor.
# DEĞİŞTİRİLMEDİ. Chunk boyutu değişirse yeniden ölç.
BM25_K1 = 1.5
BM25_B = 0.75

# RRF sabiti. `tools/hybrid_search_analysis.py`, 23 etiketli vaka, 38 chunk:
#
#   dense           R@1 0.7826  R@3 0.9565  R@5 0.9565  MRR 0.8551
#   hybrid k=1,2,3  R@1 0.9783  R@3 1.0000  R@5 1.0000  MRR 1.0000   <- 2 seçildi
#   hybrid k=4      R@1 0.9348  R@3 1.0000  R@5 1.0000  MRR 0.9783
#   hybrid k=5..60  R@1 0.9348  R@3 1.0000  R@5 1.0000  MRR 0.9710
#
# İlk ölçüm (11 vaka, 24 chunk) k=1..60 arasını ayırt edememişti ve gelenek olan
# 60 seçilmişti. Korpus ve set büyüyünce fark ortaya çıktı: k büyüdükçe
# sonuç monoton kötüleşiyor. Sebep mekanizmada: büyük k iki listede de ortalarda
# kalanı, küçük k tek listede tepe yapanı ödüllendirir. Bu korpusta BM25'in
# birebir terim eşleşmesi cosine'den daha güvenilir bir sinyal, çünkü çok dilli
# embedding modeli Türkçe'de zayıf kalıyor ("Yedekleme neden gereklidir?"
# sorusunda doğru chunk cosine 0.1972 alıyor ama BM25'te 1. sırada).
#
# Kazanan plato k=1..3; ortadaki 2 seçildi. Bu bir kalibrasyondur, gelenek
# değil; korpus veya set her değiştiğinde yeniden ölç.
#
# Korpus 217 chunk'a, set 71 etiketli vakaya çıkınca yeniden ölçüldü:
#
#   dense         R@1 0.7183  R@3 0.8873  R@5 0.9155  MRR 0.7998
#   hybrid k=1    R@1 0.9437  R@3 1.0000  R@5 1.0000  MRR 0.9718
#   hybrid k=2    R@1 0.9437  R@3 1.0000  R@5 1.0000  MRR 0.9718   <- korundu
#   hybrid k=3..5 R@1 0.9437  R@3 1.0000  R@5 1.0000  MRR 0.9695
#   hybrid k=10   R@1 0.9296  R@3 0.9859  R@5 1.0000  MRR 0.9613
#   hybrid k=20   R@1 0.9296  R@3 0.9718  R@5 1.0000  MRR 0.9570
#   hybrid k=60   R@1 0.9296  R@3 0.9718  R@5 0.9859  MRR 0.9542
#
# Küçük k'nın üstünlüğü altı kat büyük bir korpusta da sürdü, yani bu bulgu
# küçük set artefaktı değilmiş. Plato daraldı (1..3 -> 1..2) ama mevcut değeri
# hâlâ içeriyor. DEĞİŞTİRİLMEDİ.
#
# Aynı ölçüm hybrid search'ün asıl değerini de büyüttü: salt dense R@1 bu
# korpusta 0.7183'e düşüyor (38 chunk'ta 0.7826'ydı), hybrid ise 0.9437'de
# kalıyor. Korpus büyüdükçe cosine'in tek başına yetmediği daha da belirginleşti.
RRF_K = 2

# Groundedness kontrolü. Üretilen cevabın verilen context'e dayanıp dayanmadığını
# cümle bazlı ölçer; mekanizma ve gerekçe `app/groundedness.py` içinde.
# Ölçüm: `tools/groundedness_analysis.py`, 108 vaka, 217 chunk.
#
# CÜMLE SEVİYESİ (`GROUNDEDNESS_SENTENCE_SUPPORT`). Bir cümlenin içerik
# kelimelerinin kaçta kaçı context'te geçiyor. Ölçülen gruplar:
#
#   DAYANAKLI  (context'in kendi cümleleri)  n=1179  min 1.00  ort 1.00
#   PARAFRAZ   (vakanın sorusu)              n=104   min 0.38  ort 0.91
#   DAYANAKSIZ (başka dokümandan cümle)      n=324   min 0.00  ort 0.20  max 0.67
#
#   eşik   parafraz geçen   dayanaksız geçen
#   0.34      100.0%             18.2%
#   0.50       98.1%              4.6%
#   0.60       96.2%              0.9%   <- seçilen
#   0.67       92.3%              0.0%
#   1.00       63.5%              0.0%
#
# 0.60 seçildi. 0.67 cümle bazında sıfır yanlış geçiş verir ama meşru
# cümlelerin %7.7'sini keser; yanlış ret bu değişikliğin ortadan kaldırmak
# için yapıldığı hatadır ve onu geri getirmek amacı baltalar. Kalan %0.9'luk
# yanlış geçiş cevap seviyesinde zaten yutuluyor (aşağıya bak).
#
# CEVAP SEVİYESİ (`GROUNDEDNESS_THRESHOLD`). Desteklenen cümlelerin oranı:
#
#   DAYANAKLI cevap  n=108  min 1.0000
#   UYDURMA cevap    n=108  ort 0.0463  max 0.3333
#
# 0.50 seçildi: ölçülen uydurma maksimumunun 0.167 üstünde ve "cümlelerin
# çoğunluğu dayanaklı olmalı" kuralına karşılık gelir. Aralığın ortası (0.67)
# alınmadı, çünkü buradaki üst sınır 1.0 birebir kopyalanmış metinden gelen
# dejenere bir değerdir; ona göre ortalamak gerçek parafraz cevaplara değil
# kopyaya göre kalibre etmek olurdu.
#
# Neden tek cümlelik ölçüm yeterli değildi: tek cümlede oran yalnızca 0.00 veya
# 1.00 olabilir. İlk ölçüm bu yüzden "ayrım yok" gibi göründü, oysa cümle
# seviyesinde ayrım 0.91'e karşı 0.20'ydi. Cevap eşiği daima çok cümleli
# vekillerle ölçülmelidir.
#
# Sıralamanın yanlış dokümanı getirdiği 4 vaka ölçümden dışlandı; orada düşük
# groundedness DOĞRU davranıştır ve eşiği yapay olarak sıfıra çeker.
GROUNDEDNESS_THRESHOLD = 0.50
GROUNDEDNESS_SENTENCE_SUPPORT = 0.60

# Bir cümle hakkında dayanaklılık iddiasında bulunmak için gereken en az içerik
# kelimesi. Tek kelimelik cümlede oran 0 veya 1'dir ve gürültüdür.
GROUNDEDNESS_MIN_SENTENCE_TERMS = 2

USE_EXTRACTIVE_FALLBACK = True
EXTRACTIVE_SCORE_THRESHOLD = 0.50

# Extractive kısayolu için gereken en az kelime kanıtı. Bu yol chunk metnini
# doğrudan cevap olarak döndürür, yani ne modelden ne groundedness kontrolünden
# geçer. Ön kapı 0.675'ten 0.21'e indirilince ölçümde iki hard negative
# (`hard_negative_git_revert_command` 0.34, `hard_negative_coverage_tool` 0.49)
# tam buradan sızdı: yüksek cosine skoru aldıkları için alakasız bir chunk
# metni doğrudan cevap oluyordu.
#
# Değer, ön kapının ESKİ kalibrasyonudur ve burada hâlâ savunulabilir. Sebep
# ödünleşimin yönünde: extractive kısayolu "bu chunk cevabın kendisidir" gibi
# güçlü bir iddiadır ve güçlü kanıt isteyebilir. Yanlış reddin bedeli de düşük,
# çünkü kanıt yetersizse soru REDDEDİLMEZ, yalnızca normal üretken yola düşer;
# kararı orada model ve groundedness verir.
EXTRACTIVE_TERM_EVIDENCE_MIN = 0.675
MAX_EXTRACTIVE_CHARS = 500

MIN_GENERATIVE_ANSWER_CHARS = 30
