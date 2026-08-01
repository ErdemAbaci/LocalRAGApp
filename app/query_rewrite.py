"""Takip sorularını kendi başına anlaşılır hale getirir.

Sorun: retrieval her soruyu sıfırdan görür. "Kilitlenme nedir?" sorusundan
sonra gelen "Peki nasıl önlenir?" sorusunda hiçbir konu kelimesi yoktur;
embedding de BM25 de tutunacak bir şey bulamaz ve sistem alakasız bir chunk
getirir. Kelime kanıtı ve groundedness kapıları bunu yakalayıp reddeder, yani
kullanıcı yanlış cevap değil **hiç cevap** alamaz. Eksik olan retrieval'ın
kendisi değil, sorunun bağlamıdır.

Neden LLM ile yeniden yazmıyoruz? Yaygın çözüm geçmişi ve soruyu modele verip
"bunu tek başına anlaşılır bir soru haline getir" demektir. Bu, her soruya
ikinci bir model çağrısı ekler; bu projede tek bir generation 5-40 saniye
sürüyor, yani en ucuz sorular en çok yavaşlayanlar olurdu. Dahası çıktı
deterministik olmadığı için testle sabitlenemez ve yanlış yeniden yazım
sessizce yanlış retrieval üretir.

Buradaki yöntem bunun yerine mevcut kelime makinesini yeniden kullanır:
`extract_question_terms()` zaten sorunun ayırt edici kelimelerini biliyor. Bir
soru kendi başına yeterli konu kelimesi taşımıyorsa, bir önceki sorunun konu
kelimeleri sorunun sonuna eklenir. Kullanıcının yazdığı kelimeler silinmez,
yalnızca eksik bağlam tamamlanır; böylece yeniden yazım sorunun anlamını
değiştiremez.

Kapılar bu yüzden yeniden yazılmış soruyu görür. Görmeseydi retrieval konuyu
bulur, kelime kanıtı kapısı ise hâlâ bağlamsız soruya bakıp doğru sonucu
reddederdi.

Sınır: konu takibi son soruya bakar, konu **değişimini** tespit etmez. Kullanıcı
konu değiştirip yine de kısa bir soru sorarsa (örn. "Peki maliyeti?") önceki
konunun kelimeleri eklenir. Bu kabul edilmiş bir maliyettir: alternatif olan
konu değişimi sezgisi, tam da bu modülün kaçındığı belirsiz tahmin işidir.
Kullanıcıya hangi kelimelerin eklendiği gösterilir, yani yanlış bağlam görünür
kalır ve soruyu açıkça yazarak düzeltilebilir.
"""

from app.term_evidence import (
    extract_question_terms,
    terms_match,
    tokenize,
)


# İşaret ve bağlaç kelimeleri: kendileri bir konu taşımaz, bir öncekine
# gönderme yapar. `QUESTION_STOPWORDS` bunları içermez çünkü orada amaç ayırt
# edici olmayan kelimeleri elemektir; burada amaç sorunun bir **önceki soruya
# bağlı** olduğunu tanımaktır. İki liste farklı soruları cevapladığı için
# ayrıdır.
#
# DİKKAT: `normalize_text()` çıktısıyla karşılaştırılır, yani Türkçe
# karakterlerle yazılmalıdır. ASCII yazılmış bir kelime hiç eşleşmez.
FOLLOW_UP_MARKERS = frozenset({
    "peki", "ya", "yani", "o", "onu", "onun", "ona", "onda", "ondan",
    "bu", "bunu", "bunun", "buna", "bunda", "bundan", "bunlar", "bunları",
    "şu", "şunu", "şunun", "şuna", "şunlar",
    "aynı", "aynısı", "orada", "oradaki", "burada", "buradaki",
    "hepsi", "hangisi",
})

# Konu kelimesi taşımayan soru her zaman takip sorusudur ("Nasıl önlenir?").
# İşaret kelimesi olan soru ise ancak tek bir konu kelimesi kaldıysa takip
# sayılır: "Peki fidye yazılımı nedir?" iki konu kelimesiyle kendi başına
# yeterlidir ve önceki konunun kelimeleri eklenirse retrieval bozulur.
FOLLOW_UP_MAX_TERMS_WITH_MARKER = 1

# Bir önceki sorudan taşınacak en fazla kelime sayısı. Sınırsız taşımak,
# zincirin her adımında soruyu büyütür ve birkaç turdan sonra soru artık
# kullanıcının sorduğu şey olmaz. Konu genelde 1-3 kelimedir.
MAX_CARRIED_TERMS = 3


def content_terms(question):
    """Sorunun işaret kelimesi olmayan konu kelimeleri."""
    return [
        term
        for term in extract_question_terms(question)
        if term not in FOLLOW_UP_MARKERS
    ]


def has_follow_up_marker(question):
    return any(token in FOLLOW_UP_MARKERS for token in tokenize(question))


def is_follow_up(question):
    terms = content_terms(question)

    if not terms:
        return True

    return (
        has_follow_up_marker(question)
        and len(terms) <= FOLLOW_UP_MAX_TERMS_WITH_MARKER
    )


def carried_terms(question, topic, max_terms=MAX_CARRIED_TERMS):
    """Konudan alınıp soruya eklenecek kelimeler.

    Soruda zaten aynı kökten bir kelime varsa taşınmaz; `terms_match` kullanılır
    çünkü `kilitlenme` ile `kilitlenmeyi` aynı kelimedir ve birebir karşılaştırma
    bunu göremez.
    """
    existing = content_terms(question)
    carried = []

    for term in content_terms(topic):
        if any(terms_match(term, present) for present in existing):
            continue

        if term in carried:
            continue

        carried.append(term)

        if len(carried) >= max_terms:
            break

    return tuple(carried)


class FollowUpContext:
    """Son sorunun konusunu tutar ve takip sorularını tamamlar.

    Konu olarak **yeniden yazılmış** soru saklanır, kullanıcının yazdığı ham
    soru değil. Aksi halde zincir ikinci adımda kopardı: "Kilitlenme nedir?" ->
    "Peki nasıl önlenir?" -> "Ya maliyeti?" üçüncü soruda ikinci sorunun ham
    haline bakar ve orada `kilitlenme` yoktur.
    """

    def __init__(self, max_terms=MAX_CARRIED_TERMS):
        self.max_terms = max_terms
        self.topic = None

    def clear(self):
        self.topic = None

    def remember(self, question):
        clean = question.strip()

        if clean:
            self.topic = clean

    def resolve(self, question):
        """(yeniden_yazılmış_soru, eklenen_kelimeler) döndürür."""
        clean = question.strip()

        if not clean or not self.topic or not is_follow_up(clean):
            return clean, ()

        carried = carried_terms(clean, self.topic, max_terms=self.max_terms)

        if not carried:
            return clean, ()

        return f"{clean} {' '.join(carried)}", carried
