from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.database import get_all_chunks


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_top_chunks(question, top_k=3):
    model = SentenceTransformer(MODEL_NAME)

    chunks = get_all_chunks()

    if not chunks:
        return []

    question_embedding = model.encode([question])

    chunk_embeddings = []

    for chunk in chunks:
        chunk_embeddings.append(chunk["embedding"])

    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]

    results = []

    for index, score in enumerate(similarities):
        chunk = chunks[index]

        results.append({
            "id": chunk["id"],
            "source_name": chunk["source_name"],
            "chunk_text": chunk["chunk_text"],
            "score": float(score)
        })

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]