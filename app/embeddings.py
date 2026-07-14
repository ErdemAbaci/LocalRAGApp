from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_embedding_model = None


def get_local_model_path():
    try:
        return snapshot_download(repo_id=MODEL_NAME, local_files_only=True)
    except LocalEntryNotFoundError:
        return None


def is_embedding_model_loaded():
    return _embedding_model is not None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        model_source = get_local_model_path() or MODEL_NAME
        _embedding_model = SentenceTransformer(model_source)

    return _embedding_model


def embed_text(text):
    model = get_embedding_model()

    return model.encode(text).tolist()


def embed_texts(texts):
    model = get_embedding_model()

    return model.encode(texts).tolist()
