import unittest
from unittest.mock import patch

from huggingface_hub.errors import LocalEntryNotFoundError

from app import embeddings


class EmbeddingLoadingTests(unittest.TestCase):
    def tearDown(self):
        embeddings._embedding_model = None

    def test_cached_snapshot_path_is_preferred(self):
        fake_model = object()

        with patch("app.embeddings.snapshot_download", return_value="/cache/model"):
            with patch("app.embeddings.SentenceTransformer", return_value=fake_model) as loader:
                model = embeddings.get_embedding_model()

        self.assertIs(model, fake_model)
        loader.assert_called_once_with("/cache/model")

    def test_model_name_is_used_when_snapshot_is_not_cached(self):
        fake_model = object()
        cache_miss = LocalEntryNotFoundError("cache boş")

        with patch("app.embeddings.snapshot_download", side_effect=cache_miss):
            with patch("app.embeddings.SentenceTransformer", return_value=fake_model) as loader:
                model = embeddings.get_embedding_model()

        self.assertIs(model, fake_model)
        loader.assert_called_once_with(embeddings.MODEL_NAME)


if __name__ == "__main__":
    unittest.main()
