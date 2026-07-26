import numpy as np
from sklearn.preprocessing import normalize

from app.database import get_all_chunks
from app.embeddings import embed_texts


def calculate_cosine_similarities(question_embedding, chunk_embeddings):
    normalized_question = normalize(question_embedding, norm="l2")
    normalized_chunks = normalize(chunk_embeddings, norm="l2")
    return np.einsum(
        "ij,kj->ik",
        normalized_question,
        normalized_chunks,
    )[0]


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


def get_top_chunks(question, top_k=3, source_name=None, neighbor_radius=1):
    chunks = get_all_chunks(source_name=source_name)

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

    results = []

    for index, score in enumerate(similarities):
        chunk = valid_chunks[index]

        results.append({
            "id": chunk["id"],
            "source_name": chunk["source_name"],
            "source_type": chunk["source_type"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "chunk_text": chunk["chunk_text"],
            "score": float(score)
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    selected_results = results[:top_k]

    return attach_neighbor_chunks(
        results,
        selected_results,
        radius=neighbor_radius,
    )
