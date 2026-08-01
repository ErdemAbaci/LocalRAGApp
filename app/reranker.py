"""Aday chunkları cross-encoder ile yeniden sıralar.

Bi-encoder (mevcut embedding modeli) soruyu ve chunk'ı **ayrı ayrı** vektöre
çevirir, sonra iki vektörü karşılaştırır. Bu, chunk vektörlerinin önceden
hesaplanıp saklanabilmesini sağlar — 217 chunk için arama 0.1 saniye sürüyor.
Bedeli, sorunun chunk'ı okurken görülememesidir: model "kilitlenme" vektörünü
üretirken sorunun ne olduğunu bilmez.

Cross-encoder soruyu ve chunk'ı **birlikte** okur ve tek bir alaka puanı verir.
Daha isabetlidir ama önceden hesaplanamaz: her soru için her aday yeniden
işlenmek zorundadır. Bu yüzden korpusun tamamına uygulanamaz, yalnızca ilk
aşamanın seçtiği küçük havuza uygulanır.

İki aşamalı tasarımın nedeni budur: ucuz aşama 217 chunk'ı birkaç adaya indirir,
pahalı aşama o adayları doğru sıraya koyar.

Yeniden sıralama skoru **yalnızca sıralamada** kullanılır. Kapı skoru cosine
kalır (`retrieval.gate_score()`), tıpkı hybrid search'ün RRF skorunda olduğu
gibi. Aksi halde `SIMILARITY_THRESHOLD`, `CONTEXT_SCORE_THRESHOLD`,
`EXTRACTIVE_SCORE_THRESHOLD` ve eval'deki hard negative `max_score` kontrolü
yeni bir ölçeğe göre yeniden kalibre edilmek zorunda kalırdı. Tek değişkeni
izole tutmak, değişikliğin ölçülebilmesinin ön şartıdır.

Model yüklenemezse (ağ yok, model indirilmemiş) sıralama sessizce ilk aşamanın
sonucunda kalır. Reranking bir iyileştirmedir, bir ön şart değil; yokluğunda
uygulama çalışmaya devam etmelidir.
"""

from app.config import (
    RERANK_MAX_LENGTH,
    RERANKER_MODEL,
)


class RerankerUnavailableError(RuntimeError):
    pass


_cross_encoder = None
_load_error = None


def load_cross_encoder(model_name=RERANKER_MODEL, max_length=RERANK_MAX_LENGTH):
    """Cross-encoder'ı lazy yükler ve süreç boyunca tekrar kullanır.

    Yükleme hatası bir kez saklanır ve sonraki çağrılarda yeniden denenmez.
    Aksi halde model indirilmemişse her soru bir ağ zaman aşımı bekler.
    """
    global _cross_encoder, _load_error

    if _cross_encoder is not None:
        return _cross_encoder

    if _load_error is not None:
        raise RerankerUnavailableError(_load_error)

    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(model_name, max_length=max_length)
    except Exception as error:
        _load_error = f"Yeniden sıralama modeli yüklenemedi: {error}"
        raise RerankerUnavailableError(_load_error) from error

    return _cross_encoder


def reset_cross_encoder():
    """Testler ve ölçüm araçları için yükleme durumunu sıfırlar."""
    global _cross_encoder, _load_error

    _cross_encoder = None
    _load_error = None


def score_pairs(question, texts, model=None):
    encoder = model or load_cross_encoder()
    pairs = [(question, text) for text in texts]

    return [float(score) for score in encoder.predict(pairs)]


def rerank(question, results, score_func=None):
    """Adayları yeniden sıralar ve her sonuca `rerank_score` ekler.

    Sıralama kararlılığı için eşitlikte önceki sıra korunur: cross-encoder iki
    adaya aynı puanı verirse ilk aşamanın kararı geçerli kalır.
    """
    if not results:
        return []

    scorer = score_func or score_pairs
    scores = scorer(question, [result["chunk_text"] for result in results])

    if len(scores) != len(results):
        raise ValueError("Yeniden sıralama skor sayısı aday sayısıyla eşleşmiyor.")

    scored = [
        (dict(result, rerank_score=score), order)
        for order, (result, score) in enumerate(zip(results, scores))
    ]
    scored.sort(key=lambda item: (-item[0]["rerank_score"], item[1]))

    return [result for result, _order in scored]
