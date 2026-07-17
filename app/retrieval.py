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


def get_top_chunks(question, top_k=3):
    chunks = get_all_chunks()

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

    return results[:top_k]
