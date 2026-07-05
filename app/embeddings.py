from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_embedding_model = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        print("Embedding modeli yükleniyor...")
        _embedding_model = SentenceTransformer(MODEL_NAME)

    return _embedding_model


def embed_text(text):
    model = get_embedding_model()

    return model.encode(text).tolist()


def embed_texts(texts):
    model = get_embedding_model()

    return model.encode(texts).tolist()
