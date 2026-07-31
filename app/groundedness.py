"""Üretilen cevabın verilen context'e dayanıp dayanmadığını ölçer.

Kelime kanıtı kapısı (`app/term_evidence.py`) **sorunun** kelimelerini context'te
arar. Bu yöntem 112 etiketli vakada ölçüldüğünde ayrım boşluğu negatife döndü:
meşru bir soru 0.27, tuzak bir soru 0.65 alıyordu. Yani hiçbir eşik değeri iki
grubu ayıramaz. Sebep yapısaldır: kullanıcı soruyu kendi kelimeleriyle sorar,
doküman konuyu kendi kelimeleriyle anlatır ve ikisinin örtüşmesi cevabın var
olup olmadığıyla ilgili güvenilir bir sinyal değildir. Ölçülen örnek:
soru "nasıl önlenir" der, doküman "önler" der.

Bu modül aynı soruyu **cevabın** üstünden sorar: model bu cümleyi yazdı, karşılığı
verilen metinde var mı? Karşılaştırılan iki metin de artık kaynağın dilindedir,
çünkü model cevabı context'ten okuyarak üretir. Kullanıcının kelime seçimi
denklemden çıkar.

Ölçüm cümle bazlıdır. Cevabın tamamını tek blok saymak, beş dayanaklı cümlenin
arasına sıkışmış tek bir uydurma cümleyi geçirir; uydurma pratikte tam olarak
böyle görünür.

Ağırlık kullanılmaz. IDF ağırlıkları `corpus_term_weights()` tarafından yalnızca
**soru** kelimeleri için üretilir; cevabın kelimeleri o sözlükte yoktur. Eşit
sayma burada kapıdaki kadar riskli değildir: kapı korpusun tamamına karşı
ölçerken bu kontrol yalnızca modele gerçekten verilen metne karşı ölçer, yani
tesadüfi eşleşme yüzeyi çok daha dardır.

Ölçüm aracı: `tools/groundedness_analysis.py`.
"""

import re

from app.config import (
    GROUNDEDNESS_MIN_SENTENCE_TERMS,
    GROUNDEDNESS_SENTENCE_SUPPORT,
    GROUNDEDNESS_THRESHOLD,
)
from app.term_evidence import (
    build_context_terms,
    extract_question_terms,
    terms_match,
)

# Cümle sonu noktalaması. Kısaltma ve ondalık sayı ayrımı yapılmaz; yanlış
# bölünme bu kontrolde zararsızdır, çünkü parça da kendi kelimeleriyle ölçülür
# ve destek oranı parçanın uzunluğuna göre normalize edilir.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text):
    return [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY.split(str(text).strip())
        if sentence.strip()
    ]


def sentence_support(sentence, context_terms):
    """Cümlenin içerik kelimelerinin kaçta kaçı context'te geçiyor?

    Ölçülemiyorsa None döner. Bir cümle yalnızca bağlaç ve soru kalıbı
    kelimelerinden oluşuyorsa ("Yani şöyle.") o cümle hakkında dayanaklılık
    iddiasında bulunamayız; onu dayanaksız saymak cevabı haksız yere reddeder.
    """
    terms = extract_question_terms(sentence)

    if len(terms) < GROUNDEDNESS_MIN_SENTENCE_TERMS:
        return None

    matched = sum(
        1
        for term in terms
        if any(
            terms_match(term, context_term)
            for context_term in context_terms
        )
    )

    return matched / len(terms)


def groundedness_score(answer, chunks):
    """Cevabın dayanaklı cümlelerinin oranı.

    Ölçülemiyorsa None döner ve çağıran taraf bunu "dayanaksız" olarak değil
    "değerlendirilemedi" olarak ele almalıdır; ölçemediğimiz bir gerekçeyle
    kullanıcıyı reddetmek yanlış olur. Aynı kural `term_coverage()` içinde de
    geçerlidir.
    """
    if not chunks:
        return None

    sentences = split_sentences(answer)

    if not sentences:
        return None

    context_terms = build_context_terms(chunks)
    scored = [
        support
        for support in (
            sentence_support(sentence, context_terms)
            for sentence in sentences
        )
        if support is not None
    ]

    if not scored:
        return None

    supported = sum(
        1
        for support in scored
        if support >= GROUNDEDNESS_SENTENCE_SUPPORT
    )

    return supported / len(scored)


def is_grounded(answer, chunks, threshold=GROUNDEDNESS_THRESHOLD):
    score = groundedness_score(answer, chunks)

    if score is None:
        return True

    return score >= threshold
