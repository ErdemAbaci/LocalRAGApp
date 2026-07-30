SIMILARITY_THRESHOLD = 0.20
CONTEXT_SCORE_THRESHOLD = 0.35
CONTEXT_RELATIVE_SCORE_MARGIN = 0.20
TOP_K = 3
NEIGHBOR_CHUNK_RADIUS = 1
MAX_CONTEXT_CHUNKS = 5

NO_EVIDENCE_ANSWER = "Bu bilgi verilen dokümanlarda yok."

# Kelime kanıtı eşikleri. Değerler `tools/term_evidence_analysis.py` ölçümünden
# gelir (20 eval sorusu, 24 chunk). Kapsama artık kelime sayısına göre değil
# IDF ağırlığına göre hesaplanır; ölçülen ayrım boşlukları (alakalı min - tuzak
# max, pozitif olan ayırıyor demektir):
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
# Korpus 24'ten 47 chunk'a çıkınca kalibrasyon yeniden yapıldı. IDF ağırlıkları
# korpustan geldiği için doküman eklemek bu eşiği doğrudan etkiler:
#
#   24 chunk, 20 vaka : tuzak max 0.60, alakalı min 0.82, boşluk 0.21
#   47 chunk, 35 vaka : tuzak max 0.63, alakalı min 0.72, boşluk 0.09
#
# Boşluk daraldı çünkü korpus büyüdükçe soru kelimelerinin bir kısmı kaçınılmaz
# olarak başka dokümanlarda da geçiyor. 0.67 seçildi; aralığın ortasıdır.
# Eşiği tuzak maksimumuna EŞİT seçmek daha önce iki kez sızdırdı (0.50 ve 0.60);
# eşitlik geçer, bu yüzden daima aralığın ortası alınır.
#
# Doküman eklendiğinde veya eval seti büyüdüğünde yeniden ölç; ölçmeden
# değiştirme. Boşluk daralmaya devam ederse oran tabanlı kapı yerine
# groundedness kontrolüne geçmek gerekecek.
TERM_EVIDENCE_THRESHOLD = 0.67
TERM_EVIDENCE_MIN_PREFIX = 5
TERM_EVIDENCE_MIN_SHORT_ROOT = 3
TERM_EVIDENCE_MIN_TERM_LENGTH = 3

# Hybrid search. Dense (cosine) sıralaması tek başına yetersiz kaldı: ölçümde
# Recall@1 = 0.60 iken Recall@5 = 1.00 çıktı, yani doğru parça aday havuzunda
# var ama en üstte değil. Sparse (BM25) sinyali kelime örtüşmesini ölçer ve
# cosine'in kaçırdığı birebir terim eşleşmesini yakalar. Ölçülen sonuç
# (11 etiketli vaka): Recall@1 0.6364 -> 0.8182, MRR 0.7955 -> 0.9091.
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

# BM25 doygunluk ve uzunluk normalizasyonu sabitleri. Bunlar literatür
# geleneğidir (k1 genelde 1.2-2.0, b genelde 0.75), bizim setimizde ölçülmedi;
# 24 chunk'lık bir korpus bu iki parametreyi ayırt edecek kadar büyük değil.
# Korpus büyüdüğünde `tools/hybrid_search_analysis.py` ile ölç.
BM25_K1 = 1.5
BM25_B = 0.75

# RRF sabiti. `tools/hybrid_search_analysis.py`, 22 etiketli vaka, 47 chunk:
#
#   dense           R@1 0.7273  R@3 0.8864  R@5 0.9545  MRR 0.8220
#   hybrid k=1      R@1 0.8636  R@3 0.9773  R@5 1.0000  MRR 0.9318
#   hybrid k=2      R@1 0.8636  R@3 0.9773  R@5 1.0000  MRR 0.9318   <- seçilen
#   hybrid k=3,4    R@1 0.8182  R@3 0.9773  R@5 1.0000  MRR 0.9091
#   hybrid k=5..60  R@1 0.8182  R@3 0.9773  R@5 1.0000  MRR 0.9015
#
# İlk ölçüm (11 vaka, 24 chunk) k=1..60 arasını ayırt edememişti ve gelenek olan
# 60 seçilmişti. Korpus ve set iki katına çıkınca fark ortaya çıktı: k büyüdükçe
# sonuç monoton kötüleşiyor. Sebep mekanizmada: büyük k iki listede de ortalarda
# kalanı, küçük k tek listede tepe yapanı ödüllendirir. Bu korpusta BM25'in
# birebir terim eşleşmesi cosine'den daha güvenilir bir sinyal, çünkü çok dilli
# embedding modeli Türkçe'de zayıf kalıyor ("Yedekleme neden gereklidir?"
# sorusunda doğru chunk cosine 0.1972 alıyor ama BM25'te 1. sırada).
#
# k=1 ile k=2 birebir aynı; daha az uç olan 2 seçildi. Bu bir kalibrasyondur,
# gelenek değil; set her büyüdüğünde yeniden ölç.
RRF_K = 2

USE_EXTRACTIVE_FALLBACK = True
EXTRACTIVE_SCORE_THRESHOLD = 0.50
MAX_EXTRACTIVE_CHARS = 500

MIN_GENERATIVE_ANSWER_CHARS = 30
