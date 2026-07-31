"""Sorunun kelimelerinin seçilen context'te gerçekten geçip geçmediğini ölçer.

Cosine similarity **konu benzerliğini** ölçer, sorunun cevabının metinde bulunup
bulunmadığını değil. Ölçüm bunu somut biçimde gösterdi: cevabı dokümanda hiç
bulunmayan "Güvenlik duvarı kuralları nasıl yapılandırılmalıdır?" sorusu 0.5985
alırken, cevabı bulunan "RAG nedir?" 0.5570 alıyordu. Skorlar iç içe geçtiği için
tek bir similarity eşiği bu iki grubu ayıramaz.

Kelime kanıtı bağımsız bir sinyaldir. 19 vakalık ölçümde alakalı sorular
0.67-1.00, cevabı dokümanda bulunmayan sorular 0.00-0.50 kapsama aldı.

Türkçe eklemeli bir dildir; kök baştadır, ekler sona gelir. Bu yüzden eşleştirme
**ortak kök** temellidir: iki kelime yeterince uzun bir ortak öneki paylaşıyorsa
aynı kökten sayılır.

Kural neden "biri diğerinin öneki" değil? `korunulur` ve `korunmak` aynı kökten
türer ama hiçbiri diğerinin öneki değildir. Önek kuralı denendi ve gerçek
kullanımda "Kimlik avından nasıl korunulur?" sorusunu haksız yere reddetti.

Yöntemin sınırı: ortak kök kuralı `sayısı` ile `sayısal`ı da eşleştirir. İkisi
de `sayı` kökünden gelir; aradaki anlam farkı türetme ekindedir ve saf morfoloji
bunu ayıramaz. Ölçümde bu ödünleşmeye rağmen ortak kök 5, iki grubu ayıran tek
seçenek çıktı.

Aynı sınırın ölçülen ikinci örneği: `yüzde` (oran) ile `yüzden` (bu yüzden).
İkisi de `yüz` kökünün çekimidir — biri bulunma, diğeri ayrılma hali — ve ortak
kökleri tam 5 karakterdir. Anlamları alakasız olmasına rağmen morfoloji bunları
ayıramaz. `min_prefix` 6'ya çıkarılarak kapatılamaz: ölçümde 6, `korunulur` ~
`korunmak`, `süreç` ~ `süreci` ve `aşamasında` ~ `aşama` gibi meşru eşleşmeleri
kaybettiriyor. Bu tür çakışmalar kabul edilmiş bir maliyettir; kapıyı taşıyan
şey tek bir kelime değil, IDF ağırlıklı toplamdır.

Kalıcı çözüm terim ağırlıklandırmasıdır (BM25/IDF): oran bütün kelimeleri eşit
sayar, oysa "sıklıkla" ile "yedekleme" aynı ağırlıkta olmamalıdır.

Ölçüm aracı: `tools/term_evidence_analysis.py`.
"""

import re

from app.config import (
    TERM_EVIDENCE_MIN_PREFIX,
    TERM_EVIDENCE_MIN_SHORT_ROOT,
    TERM_EVIDENCE_MIN_TERM_LENGTH,
    TERM_EVIDENCE_THRESHOLD,
)


# Python'un casefold'u Türkçe'yi doğru küçültmez: "İ".casefold() sonucu "i"
# değil, "i" + U+0307 (birleşen nokta) olur ve "I".casefold() "ı" yerine "i"
# verir. Türkçe eşlemeyi önce elle yapıp artakalan birleşen noktayı temizliyoruz.
TURKISH_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı"})
COMBINING_DOT_ABOVE = "̇"

WORD_PATTERN = re.compile(r"[0-9a-zçğıöşü\-]+")

# Türkçe ünsüz yumuşaması. Yalnızca `min_prefix` uzunluğunu geçen kelimelere
# uygulandığı için kısa kelimelerde yanlış eşleşme üretmez.
CONSONANT_MUTATIONS = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}

# Soru kalıbı ve genel dilbilgisi kelimeleri. Bunlar her soruda geçtiği için
# ayırt edici değildir; sinyale dahil edilirlerse cevabı bulunmayan sorular da
# yüksek kapsama alır ve sinyal ayırt etme gücünü kaybeder.
#
# DİKKAT: bu küme normalize_text() çıktısıyla karşılaştırılır ve normalize_text
# Türkçe karakterleri korur. Kelimeleri ASCII yazmak listeyi sessizce etkisiz
# bırakır; `tests/test_term_evidence.py` bunu kontrol eder.
QUESTION_STOPWORDS = frozenset({
    "ne", "nedir", "nasıl", "neden", "niçin", "hangi", "kaç", "kim", "nerede",
    "nelerdir", "midir", "mi", "mu", "mü", "mı", "ve", "veya", "ile", "için",
    "bir", "bu", "şu", "da", "de", "ki", "en", "çok", "az", "olan", "olarak",
    "gibi", "sonra", "önce", "üzere", "göre", "kadar", "daha", "ise", "ama",
    "fakat", "ancak", "yani", "eğer", "her", "hiç", "bazı", "tüm", "olmalıdır",
    "olmalı", "kullanılır", "yapılır", "edilir", "edilmelidir", "seçilir",
    "alınmalıdır", "yapılandırılmalıdır", "kullanılmalıdır", "önerir",
    "izlenir", "var", "yok", "eder", "olur", "çalışır", "gerekir",
    # "neden önemli?" kalıbı. Bir şeyin önemini soran cevabın, dokümanda
    # "önemli" kelimesinin birebir geçmesini gerektirmesi yanlıştır; manuel
    # testte "Çok faktörlü doğrulama neden önemli?" bu yüzden reddedildi.
    "önemli", "önemlidir", "gerekli", "gereklidir",
    # "X ile Y arasındaki fark nedir?" kalıbı ve `-malıdır` çekimleri. Korpus
    # 24'ten 47 chunk'a çıkınca bunlar ölçümde yanlış ret üretti: hiçbiri
    # dokümanda geçmediği için en yüksek IDF ağırlığını alıyor ve asıl içerik
    # kelimeleri eşleşse bile kapsamayı eşiğin altına çekiyorlardı.
    "arasındaki", "arasında", "fark", "farkı", "farkları",
    "yazılmalıdır", "uygulanmalıdır", "tutulmalıdır", "seçilmelidir",
    # "nasıl önlenir?" / "nasıl anlaşılır?" kalıbı. Manuel testte 217 chunk'lık
    # korpusta iki meşru soru bu yüzden reddedildi: "Kilitlenme nedir ve nasıl
    # önlenir?" (kapsama 0.42) ve "Aşırı öğrenme nasıl anlaşılır?" (0.60).
    # Doküman "kilitlenmeyi önler" diyor ama `önler` ile `önlenir`in ortak öneki
    # 4 karakter ve `min_prefix` 5; yani kelime hiçbir yerde eşleşmiyor ve en
    # yüksek IDF ağırlığını (4.47) alıyor. Bir şeyin nasıl önlendiğini soran
    # cevabın metinde "önlenir" kelimesini birebir içermesi gerekmez.
    "önlenir", "önlenebilir", "anlaşılır", "anlaşılabilir",
})


def normalize_text(text):
    lowered = str(text).translate(TURKISH_LOWER_MAP).casefold()
    lowered = lowered.replace(COMBINING_DOT_ABOVE, "")

    return " ".join(lowered.split())


def tokenize(text):
    return WORD_PATTERN.findall(normalize_text(text))


def extract_question_terms(question, min_length=TERM_EVIDENCE_MIN_TERM_LENGTH):
    """Sorunun ayırt edici kelimelerini döndürür.

    Sıra ve tekrar korunmaz; her kelime bir kez sayılır, böylece aynı kelimenin
    tekrarı kapsama oranını şişirmez.
    """
    seen = []

    for token in tokenize(question):
        if len(token) < min_length:
            continue

        if token in QUESTION_STOPWORDS:
            continue

        if token not in seen:
            seen.append(token)

    return seen


def common_prefix_length(first, second):
    """Ortak önek uzunluğu; ünsüz yumuşamasına toleranslıdır.

    Türkçe'de sonu p/ç/t/k ile biten kelimeler ünlüyle başlayan ek aldığında
    son ünsüzlerini yumuşatır: `süreç` -> `süreci`, `kitap` -> `kitabı`. Harf
    harf karşılaştırma bu çifti eşdeğer sayar, aksi halde kök ortada kopar.
    """
    length = 0

    for left, right in zip(first, second):
        equivalent = (
            left == right
            or CONSONANT_MUTATIONS.get(left) == right
            or CONSONANT_MUTATIONS.get(right) == left
        )

        if not equivalent:
            break

        length += 1

    return length


def terms_match(
    term,
    context_term,
    min_prefix=TERM_EVIDENCE_MIN_PREFIX,
    min_short=TERM_EVIDENCE_MIN_SHORT_ROOT,
):
    """İki kelime aynı kökten mi?

    Üç kural sırayla denenir:

    1. Tam eşleşme. Aksi halde `rag` gibi minimum kökten kısa kelimeler hiç
       eşleşemez.
    2. Ortak önek en az `min_prefix`. Kural "biri diğerinin öneki" değildir:
       `korunulur` ve `korunmak` aynı kökten türer ama hiçbiri diğerinin öneki
       değildir; önek kuralı bu meşru eşleşmeyi kaçırıyordu.
    3. Kısa kelime tamamen tükendi (`min_short` karakterden uzunsa). Ölçüm bunun
       eksikliğini yakaladı: `avından` kelimesi korpusta hiçbir şeyle
       eşleşmiyordu, çünkü metindeki karşılığı `avı` yalnızca 3 karakter ve
       ortak önek şartı 5. Kökün kendisi kısa olduğunda şart kökü tamamen
       kapsamak olmalıdır, sabit bir uzunluğa ulaşmak değil.

    Ölçüm: 3. kural ağırlıklı kapsamada ayrım boşluğunu 0.02'den 0.21'e çıkardı
    (`tools/term_evidence_analysis.py`, `kök5-3` sütunu).
    """
    if term == context_term:
        return True

    length = common_prefix_length(term, context_term)

    if length >= min_prefix:
        return True

    shorter = min(len(term), len(context_term))

    return shorter >= min_short and length == shorter


def build_context_terms(chunks):
    context_text = "\n".join(chunk["chunk_text"] for chunk in chunks)

    return set(tokenize(context_text))


def term_coverage(
    question,
    chunks,
    min_prefix=TERM_EVIDENCE_MIN_PREFIX,
    weights=None,
):
    """Soru kelimelerinin context'te bulunma oranını döndürür.

    `weights` verilmezse bütün kelimeler eşit sayılır. Bu ölçümde sızdırdı:
    ayırt edici olmayan bir kelimenin eşleşmesi, ayırt edici bir kelimenin
    eksikliğini dengeleyebiliyordu. Ağırlık verildiğinde oran kelime sayısı
    yerine ayırt edicilik toplamı üzerinden hesaplanır
    (`app/sparse_search.corpus_term_weights`).

    Ölçülemiyorsa (soru yalnızca stopword içeriyorsa) None döner. Çağıran taraf
    bunu "kanıt yok" olarak değil "değerlendirilemedi" olarak ele almalıdır;
    ölçemediğimiz bir gerekçeyle kullanıcıyı reddetmek yanlış olur.
    """
    terms = extract_question_terms(question)

    if not terms:
        return None

    def weight_of(term):
        if weights is None:
            return 1.0

        return weights.get(term, 1.0)

    total_weight = sum(weight_of(term) for term in terms)

    if total_weight <= 0:
        return None

    context_terms = build_context_terms(chunks)
    matched_weight = sum(
        weight_of(term)
        for term in terms
        if any(
            terms_match(term, context_term, min_prefix=min_prefix)
            for context_term in context_terms
        )
    )

    return matched_weight / total_weight


def has_term_evidence(
    question,
    chunks,
    threshold=TERM_EVIDENCE_THRESHOLD,
    min_prefix=TERM_EVIDENCE_MIN_PREFIX,
    weights=None,
):
    if not chunks:
        return False

    coverage = term_coverage(
        question,
        chunks,
        min_prefix=min_prefix,
        weights=weights,
    )

    if coverage is None:
        return True

    return coverage >= threshold
