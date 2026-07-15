import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import database
from app.index_state import build_source_manifest, get_index_freshness


def sample_chunk(source_name):
    return {
        "source_name": source_name,
        "source_type": "txt",
        "page_number": None,
        "chunk_index": 0,
        "chunk_text": "Örnek içerik",
        "embedding": [0.1] * 384,
    }


class IndexFreshnessTests(unittest.TestCase):
    def test_matching_manifest_is_current(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "notes.txt").write_text("RAG notları", encoding="utf-8")
            db_path = root / "rag.db"

            with patch.object(database, "DB_PATH", db_path):
                database.init_db()
                database.replace_chunks(
                    [sample_chunk("notes.txt")],
                    source_manifest=build_source_manifest(docs_dir),
                )
                freshness = get_index_freshness(docs_dir, db_path)

        self.assertTrue(freshness.is_current)
        self.assertEqual(freshness.display_status(), "güncel")

    def test_added_modified_and_deleted_sources_make_index_stale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            changed_path = docs_dir / "changed.txt"
            deleted_path = docs_dir / "deleted.txt"
            changed_path.write_text("eski içerik", encoding="utf-8")
            deleted_path.write_text("silinecek içerik", encoding="utf-8")
            db_path = root / "rag.db"

            with patch.object(database, "DB_PATH", db_path):
                database.init_db()
                database.replace_chunks(
                    [sample_chunk("changed.txt"), sample_chunk("deleted.txt")],
                    source_manifest=build_source_manifest(docs_dir),
                )
                changed_path.write_text("yeni ve daha uzun içerik", encoding="utf-8")
                deleted_path.unlink()
                (docs_dir / "added.pdf").write_bytes(b"new pdf")
                freshness = get_index_freshness(docs_dir, db_path)

        self.assertEqual(freshness.status, "stale")
        self.assertEqual(freshness.added, ("added.pdf",))
        self.assertEqual(freshness.modified, ("changed.txt",))
        self.assertEqual(freshness.deleted, ("deleted.txt",))
        self.assertIn("Eklenen: added.pdf", freshness.change_summary())
        self.assertIn("Değişen: changed.txt", freshness.change_summary())
        self.assertIn("Silinen: deleted.txt", freshness.change_summary())

    def test_legacy_index_without_manifest_is_untracked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "notes.txt").write_text("RAG notları", encoding="utf-8")
            db_path = root / "rag.db"

            with patch.object(database, "DB_PATH", db_path):
                database.init_db()
                database.replace_chunks([sample_chunk("notes.txt")])
                freshness = get_index_freshness(docs_dir, db_path)

        self.assertEqual(freshness.status, "untracked")
        self.assertIn("yeniden indeks", freshness.display_status())

    def test_missing_database_has_distinct_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            freshness = get_index_freshness(root / "docs", root / "missing.db")

        self.assertEqual(freshness.status, "missing")
        self.assertEqual(freshness.display_status(), "indeks bulunamadı")


if __name__ == "__main__":
    unittest.main()
