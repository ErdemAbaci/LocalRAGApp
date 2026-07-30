"""BM25 ile sparse (kelime örtüşmesi) skoru.

Dense retrieval sorunun ve chunk'ın **anlamını** vektöre çevirip karşılaştırır.
Bu güçlü ama körlüğü var: birebir geçen ayırt edici bir terimi özel olarak
ödüllendirmez. Ölçümde bunun sonucu görüldü — "Kimlik avından nasıl korunulur?"
sorusunda cevabı içeren chunk cosine sıralamasında 4. sıradaydı, oysa `kimlik`
ve `avı` kelimeleri tam olarak o chunk'ta geçiyor.

BM25 tam bunu ölçer ve üç şeyi hesaba katar:

1. **Terim frekansı (tf), doygunlukla.** Bir kelimenin 10 kez geçmesi 1 kez
   geçmesinden iyidir ama 10 kat iyi değildir. `k1` bu doygunluğu ayarlar.
2. **Ters doküman frekansı (idf).** Her chunk'ta geçen bir kelime ayırt edici
   değildir; yalnızca birkaç chunk'ta geçen kelime değerlidir. Kelime kanıtı
   oranının eksiği tam buydu: "sıklıkla" ile "yedekleme"yi eşit sayıyordu.
3. **Doküman uzunluğu.** Uzun bir chunk'ta kelimenin geçmesi tesadüfe daha
   yakındır. `b` bu cezayı ayarlar.

Neden hazır bir kütüphane veya SQLite FTS5 değil? FTS5'in `unicode61`
tokenizer'ı Türkçe bilmez; stemming yok, `remove_diacritics` seçenekleri ise
`ı/i` ve `ş/s` ayrımını bozar. Daha önemlisi ikinci bir normalizasyon yolu
doğardı ve `app/term_evidence.normalize_text` ile zamanla ayrışırdı. Burada
tokenizasyon ve morfoloji toleransı uygulamanın geri kalanıyla aynı koddan
gelir.

Sorgu tarafında stopword'ler atılır. Teorik olarak idf bunu kendi yapmalı, ama
24 chunk'lık bir korpusta "nasıl" kelimesi tesadüfen tek bir chunk'ta geçerse
idf'i yükselir ve soru kalıbı ayırt edici bir sinyale dönüşür. Küçük korpusta
idf istatistiği güvenilir değildir; korpus büyüyünce bu filtre gevşetilebilir.

Chunk tarafında stopword atılmaz: doküman uzunluğu gerçek uzunluk olmalıdır,
yoksa normalizasyon bozulur.
"""

import math

from app.config import BM25_B, BM25_K1, TERM_EVIDENCE_MIN_PREFIX
from app.term_evidence import extract_question_terms, terms_match, tokenize


def build_document_terms(texts):
    return [tokenize(text) for text in texts]


def term_frequency(term, document_terms, min_prefix):
    """Kelimenin dokümanda kaç kez geçtiği; morfoloji toleranslı.

    Birebir eşitlik yerine `terms_match` kullanılır, çünkü Türkçe'de `kimlik`
    sorusu metinde `kimliğin` olarak geçer. Maliyet sorgu kelimesi x doküman
    kelimesi; 24 chunk'lık korpusta ölçülemeyecek kadar küçük, ama korpus
    büyürse burası ilk optimize edilecek yerdir (ters indeks).
    """
    return sum(
        1
        for document_term in document_terms
        if terms_match(term, document_term, min_prefix=min_prefix)
    )


def inverse_document_frequency(document_count, matching_document_count):
    """Olasılıksal (BM25) idf.

    +0.5 düzeltmeleri bütün dokümanlarda geçen bir kelimede skorun negatife
    dönmesini engeller; 1 + ... sarmalayıcısı sonucun daima pozitif kalmasını
    garanti eder.
    """
    numerator = document_count - matching_document_count + 0.5
    denominator = matching_document_count + 0.5

    return math.log(1 + numerator / denominator)


def corpus_term_weights(
    question,
    document_terms,
    min_prefix=TERM_EVIDENCE_MIN_PREFIX,
):
    """Soru kelimelerinin ayırt edicilik ağırlıkları: kelime -> idf.

    Kelime kanıtı oranının kör noktası için var. Oran bütün kelimeleri eşit
    sayıyordu ve ölçümde bu somut bir sızıntı üretti: "Güvenlik duvarı kuralları
    nasıl yapılandırılmalıdır?" sorusunda `güvenlik` neredeyse her chunk'ta
    geçiyor (hiçbir şey kanıtlamaz), `kuralları` alakasız bir chunk'taki
    "3-2-1 kuralı" ile eşleşiyor, ayırt edici olan `duvarı` ise dokümanlarda hiç
    yok. Eşit sayınca kapsama 0.67 çıkıp eşiği geçiyordu.

    Hiç geçmeyen kelime df=0 alır ve en yüksek ağırlığa çıkar. Bu kasıtlıdır:
    dokümanlarda hiç bulunmayan bir kelime, sorunun cevaplanamadığının en güçlü
    işaretidir.

    Ağırlıklar korpusun tamamı üzerinden hesaplanır, seçilen context üzerinden
    değil. Beş chunk'lık bir örneklemde df istatistiği anlamsızdır; ayrıca bir
    kelimenin context'in tamamında geçmesi onu ağırlıksız bırakırdı.

    tf hesabı `bm25_scores` ile aynı işi tekrar yapar. 24 chunk'ta ölçülemeyecek
    kadar küçük bir maliyet; korpus büyürse iki fonksiyon ortak bir frekans
    matrisi paylaşmalı.
    """
    terms = extract_question_terms(question)
    document_count = len(document_terms)

    if not document_count:
        return {term: 1.0 for term in terms}

    weights = {}

    for term in terms:
        matching_document_count = sum(
            1
            for terms_in_document in document_terms
            if term_frequency(term, terms_in_document, min_prefix)
        )
        weights[term] = inverse_document_frequency(
            document_count,
            matching_document_count,
        )

    return weights


def bm25_scores(
    question,
    document_terms,
    k1=BM25_K1,
    b=BM25_B,
    min_prefix=TERM_EVIDENCE_MIN_PREFIX,
):
    """Her doküman için BM25 skoru; sıra girişle aynıdır.

    Skorun üst sınırı yoktur ve sorgular arasında karşılaştırılamaz. Bu yüzden
    bir eşikle kullanılamaz; yalnızca aynı sorgu içinde sıralama üretir.
    """
    document_count = len(document_terms)

    if not document_count:
        return []

    query_terms = extract_question_terms(question)
    scores = [0.0] * document_count

    if not query_terms:
        return scores

    lengths = [len(terms) for terms in document_terms]
    average_length = sum(lengths) / document_count

    if average_length == 0:
        return scores

    for term in query_terms:
        frequencies = [
            term_frequency(term, terms, min_prefix)
            for terms in document_terms
        ]
        matching_document_count = sum(1 for frequency in frequencies if frequency)

        if not matching_document_count:
            continue

        idf = inverse_document_frequency(document_count, matching_document_count)

        for index, frequency in enumerate(frequencies):
            if not frequency:
                continue

            normalization = 1 - b + b * lengths[index] / average_length
            scores[index] += idf * (
                frequency * (k1 + 1) / (frequency + k1 * normalization)
            )

    return scores
