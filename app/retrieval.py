import numpy as np
from sklearn.preprocessing import normalize

from app.config import (
    BM25_B,
    BM25_K1,
    RERANK_CANDIDATE_POOL,
    RRF_K,
    USE_HYBRID_SEARCH,
    USE_RERANKER,
)
from app.database import get_all_chunks
from app.embeddings import embed_texts
from app.sparse_search import (
    bm25_scores,
    build_document_terms,
    corpus_term_weights,
)


def calculate_cosine_similarities(question_embedding, chunk_embeddings):
    normalized_question = normalize(question_embedding, norm="l2")
    normalized_chunks = normalize(chunk_embeddings, norm="l2")
    return np.einsum(
        "ij,kj->ik",
        normalized_question,
        normalized_chunks,
    )[0]


def rank_positions(scores, only_positive=False):
    """Skorları 1'den başlayan sıra numaralarına çevirir.

    Eşitlikte indeks sırası kullanılır; böylece sonuç deterministiktir.
    `only_positive` verildiğinde sıfır skorlu dokümanlar sıralanmaz ve None
    alır. BM25'te sıfır "hiçbir sorgu kelimesi geçmiyor" demektir; bunları
    sıralamaya sokmak, aralarındaki keyfi eşitlik sırasını sinyal sanıp
    alakasız dokümanlara puan dağıtmak olur.
    """
    order = sorted(
        range(len(scores)),
        key=lambda index: (-scores[index], index),
    )

    positions = [None] * len(scores)

    for rank, index in enumerate(order, start=1):
        if only_positive and scores[index] <= 0:
            continue

        positions[index] = rank

    return positions


def reciprocal_rank_fusion(dense_scores, sparse_scores, rrf_k=RRF_K):
    """İki sıralamayı birleştirir: skor = toplam 1/(k + sıra).

    Neden skor değil sıra? Cosine 0-1 arasında, BM25 üst sınırsız. İki ölçeği
    doğrudan toplamak için normalizasyon gerekir; normalizasyon aynı sorgunun
    aday havuzu içinde min-max yapılmak zorunda olduğu için skorlar sorgular
    arası karşılaştırılamaz hale gelir ve hard negative kapısı bozulur. Sıra
    kullanmak bu kalibrasyon sorununu tamamen ortadan kaldırır.

    Bedeli: skorun büyüklük bilgisi atılır. 0.90 ile 0.89 arasındaki fark,
    0.90 ile 0.30 arasındaki farkla aynı sayılır. Bu yüzden füzyon skoru
    yalnızca sıralamada kullanılır, kapıda kullanılmaz.

    `rrf_k`'nın ne yaptığı sezgisel değil: iki sinyalin birden gördüğü chunk her
    k değerinde tek sinyalin gördüğünün önüne geçer. k'nın belirlediği şey,
    listelerden birinde tepe yapan chunk ile ikisinde de ortalarda kalan chunk
    arasındaki tercihtir. Küçük k tepeyi, büyük k istikrarı ödüllendirir.
    `tests/test_retrieval.py` bu davranışı sabitler.
    """
    dense_positions = rank_positions(dense_scores)
    sparse_positions = rank_positions(sparse_scores, only_positive=True)

    fused = []

    for dense_rank, sparse_rank in zip(dense_positions, sparse_positions):
        score = 1 / (rrf_k + dense_rank)

        if sparse_rank is not None:
            score += 1 / (rrf_k + sparse_rank)

        fused.append(score)

    return fused


def gate_score(results):
    """Eşik karşılaştırmalarında kullanılacak dense skor.

    Hybrid search sıralamayı değiştirdiği için `results[0]` artık en yüksek
    cosine skoruna sahip chunk olmak zorunda değil. Kapı skorunu sıralamadan
    okumak, füzyonun aday sırasını değiştirdiği her sorguda eşiği sessizce
    gevşetir veya sıkar. Bu yüzden retrieval en yüksek cosine skorunu ayrıca
    taşır ve kapı onu kullanır.
    """
    if not results:
        return 0.0

    return max(
        float(result.get("dense_best_score", result["score"]))
        for result in results
    )


def document_order_key(chunk):
    page_number = chunk.get("page_number")
    chunk_index = chunk.get("chunk_index")
    return (
        chunk["source_name"].casefold(),
        page_number if page_number is not None else 0,
        chunk_index if chunk_index is not None else chunk["id"],
        chunk["id"],
    )


def attach_neighbor_chunks(ranked_results, selected_results, radius=1):
    if radius <= 0:
        return [dict(result, neighbors=[]) for result in selected_results]

    chunks_by_source = {}
    for result in ranked_results:
        chunks_by_source.setdefault(result["source_name"], []).append(result)

    positions = {}
    for source_chunks in chunks_by_source.values():
        source_chunks.sort(key=document_order_key)
        positions.update({chunk["id"]: index for index, chunk in enumerate(source_chunks)})

    enriched_results = []
    for result in selected_results:
        source_chunks = chunks_by_source[result["source_name"]]
        position = positions[result["id"]]
        start = max(0, position - radius)
        end = min(len(source_chunks), position + radius + 1)
        neighbors = [
            dict(chunk)
            for chunk in source_chunks[start:end]
            if chunk["id"] != result["id"]
        ]
        enriched_results.append(dict(result, neighbors=neighbors))

    return enriched_results


def apply_reranking(
    question,
    ranked_results,
    top_k,
    use_reranker=USE_RERANKER,
    candidate_pool=RERANK_CANDIDATE_POOL,
    rerank_func=None,
):
    """İlk aşamanın havuzunu cross-encoder ile yeniden sıralar.

    Havuz `top_k`'dan küçük olamaz; aksi halde reranking seçilecek sonuç sayısını
    kısar ve ölçüm "reranking mi kötüledi, havuz mu daraldı" sorusuna cevap
    veremez hale gelir.

    Model yüklenemezse ilk aşamanın sırası aynen döner. Reranking bir
    iyileştirmedir, ön şart değil.
    """
    if not use_reranker or not ranked_results:
        return ranked_results[:top_k]

    pool = ranked_results[:max(top_k, candidate_pool)]

    if rerank_func is None:
        from app.reranker import RerankerUnavailableError, rerank

        try:
            reranked = rerank(question, pool)
        except RerankerUnavailableError:
            return ranked_results[:top_k]
    else:
        reranked = rerank_func(question, pool)

    return reranked[:top_k]


def rank_chunks(
    question,
    chunks,
    top_k=3,
    neighbor_radius=1,
    use_hybrid=USE_HYBRID_SEARCH,
    rrf_k=RRF_K,
    bm25_k1=BM25_K1,
    bm25_b=BM25_B,
    use_reranker=USE_RERANKER,
    candidate_pool=RERANK_CANDIDATE_POOL,
    rerank_func=None,
):
    """Verilen chunk listesini sıralar; veri erişimi yapmaz.

    Skorlama, veritabanından ayrıldı çünkü chunking deneyi aynı sıralama
    mantığını **indekse dokunmadan** farklı parçalama ayarlarıyla ölçmek
    zorunda. Aynı kodu araç içinde tekrar yazmak, ölçülen şeyin uygulamanın
    gerçekte çalıştırdığı şey olmadığı anlamına gelirdi.
    """
    if not chunks:
        return []

    question_embedding = np.asarray(embed_texts([question]), dtype=np.float32)

    if not np.isfinite(question_embedding).all():
        return []

    chunk_embeddings = []
    valid_chunks = []

    for chunk in chunks:
        embedding = np.asarray(chunk["embedding"], dtype=np.float32)

        if embedding.ndim != 1:
            continue

        if not np.isfinite(embedding).all():
            continue

        chunk_embeddings.append(embedding)
        valid_chunks.append(chunk)

    if not chunk_embeddings:
        return []

    chunk_embeddings = np.vstack(chunk_embeddings)
    similarities = calculate_cosine_similarities(
        question_embedding,
        chunk_embeddings,
    )
    similarities = np.nan_to_num(similarities, nan=-1.0, posinf=-1.0, neginf=-1.0)

    dense_scores = [float(score) for score in similarities]
    document_terms = build_document_terms(
        chunk["chunk_text"] for chunk in valid_chunks
    )

    # Ağırlıklar korpusun tamamı üzerinden hesaplanır ve kelime kanıtı kapısına
    # taşınır. Kapı yalnızca seçilen context'i görür; ayırt ediciliği ise ancak
    # korpusun tamamı söyleyebilir. Bu bilgiyi üretebilecek tek katman burası.
    question_term_weights = corpus_term_weights(question, document_terms)

    if use_hybrid:
        sparse_scores = bm25_scores(question, document_terms, k1=bm25_k1, b=bm25_b)
        fusion_scores = reciprocal_rank_fusion(
            dense_scores,
            sparse_scores,
            rrf_k=rrf_k,
        )
    else:
        sparse_scores = [0.0] * len(valid_chunks)
        fusion_scores = list(dense_scores)

    dense_best_score = max(dense_scores)
    results = []

    for index, chunk in enumerate(valid_chunks):
        results.append({
            "id": chunk["id"],
            "source_name": chunk["source_name"],
            "source_type": chunk["source_type"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "chunk_text": chunk["chunk_text"],
            "score": dense_scores[index],
            "sparse_score": sparse_scores[index],
            "fusion_score": fusion_scores[index],
            "dense_best_score": dense_best_score,
            "question_term_weights": question_term_weights,
        })

    # Eşitlikte cosine'e düşülür; sparse sinyal hiç eşleşme bulamadığında
    # sıralama birebir eski dense davranışına döner.
    results.sort(
        key=lambda item: (item["fusion_score"], item["score"]),
        reverse=True,
    )
    selected_results = apply_reranking(
        question,
        results,
        top_k=top_k,
        use_reranker=use_reranker,
        candidate_pool=candidate_pool,
        rerank_func=rerank_func,
    )

    return attach_neighbor_chunks(
        results,
        selected_results,
        radius=neighbor_radius,
    )


def get_top_chunks(
    question,
    top_k=3,
    source_name=None,
    neighbor_radius=1,
    use_hybrid=USE_HYBRID_SEARCH,
    rrf_k=RRF_K,
    bm25_k1=BM25_K1,
    bm25_b=BM25_B,
    use_reranker=USE_RERANKER,
    candidate_pool=RERANK_CANDIDATE_POOL,
    rerank_func=None,
):
    return rank_chunks(
        question,
        get_all_chunks(source_name=source_name),
        top_k=top_k,
        neighbor_radius=neighbor_radius,
        use_hybrid=use_hybrid,
        rrf_k=rrf_k,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        use_reranker=use_reranker,
        candidate_pool=candidate_pool,
        rerank_func=rerank_func,
    )
