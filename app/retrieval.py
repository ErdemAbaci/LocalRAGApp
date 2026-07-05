from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.database import get_all_chunks


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_embedding_model = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        print("Embedding modeli yükleniyor...")
        _embedding_model = SentenceTransformer(MODEL_NAME)

    return _embedding_model


def get_top_chunks(question, top_k=3):
    model = get_embedding_model()

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