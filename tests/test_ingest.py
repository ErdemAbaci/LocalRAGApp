import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from app import database
from app.ingest import ingest_documents, split_long_text


class ChunkingTests(unittest.TestCase):
    def test_chunks_prefer_sentence_boundaries(self):
        text = (
            "Birinci cümle temel kavramı açıklar. "
            "İkinci cümle ayrıntılı bir örnek sunar. "
            "Üçüncü cümle sonucu açık biçimde özetler."
        )

        chunks = split_long_text(text, chunk_size=70, chunk_overlap=25)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk[0].isupper() for chunk in chunks))
        self.assertTrue(all(len(chunk) <= 70 for chunk in chunks))

    def test_chunks_do_not_start_inside_words(self):
        text = " ".join(f"kelime{i}" for i in range(40))

        chunks = split_long_text(text, chunk_size=80, chunk_overlap=17)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.startswith("kelime") for chunk in chunks))


class AtomicReindexTests(unittest.TestCase):
    def test_replace_chunks_rolls_back_on_insert_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "rag.db"

            with patch.object(database, "DB_PATH", db_path):
                database.init_db()
                old_manifest = [{
                    "source_name": "old.txt",
                    "source_type": "txt",
                    "file_size": 12,
                    "sha256": "old-hash",
                }]
                database.replace_chunks(
                    [{
                        "source_name": "old.txt",
                        "chunk_text": "eski içerik",
                        "embedding": [0.1, 0.2],
                    }],
                    source_manifest=old_manifest,
                )

                invalid_chunks = [{
                    "source_name": "new.txt",
                    "chunk_text": None,
                    "embedding": [0.3, 0.4],
                }]

                with self.assertRaises(sqlite3.IntegrityError):
                    database.replace_chunks(
                        invalid_chunks,
                        source_manifest=[{
                            "source_name": "new.txt",
                            "source_type": "txt",
                            "file_size": 15,
                            "sha256": "new-hash",
                        }],
                    )

                chunks = database.get_all_chunks()
                manifest = database.get_source_manifest()

            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["source_name"], "old.txt")
            self.assertEqual(chunks[0]["chunk_text"], "eski içerik")
            self.assertEqual(manifest, old_manifest)

    def test_embedding_error_does_not_replace_index(self):
        documents = [{
            "source_name": "new.txt",
            "source_type": "txt",
            "page_number": None,
            "text": "Yeni doküman içeriği.",
        }]

        with (
            patch("app.ingest.init_db"),
            patch("app.ingest.build_source_manifest", return_value=[]),
            patch("app.ingest.read_documents", return_value=documents),
            patch("app.ingest.embed_texts", side_effect=RuntimeError("embedding hatası")),
            patch("app.ingest.replace_chunks") as replace_mock,
        ):
            with self.assertRaises(RuntimeError):
                ingest_documents()

        replace_mock.assert_not_called()

    def test_ingest_replaces_chunks_and_manifest_together(self):
        manifest = [{
            "source_name": "notes.txt",
            "source_type": "txt",
            "file_size": 24,
            "sha256": "stable-hash",
        }]
        documents = [{
            "source_name": "notes.txt",
            "source_type": "txt",
            "page_number": None,
            "text": "RAG dokümanlardan cevap üretir.",
        }]

        with (
            patch("app.ingest.init_db"),
            patch("app.ingest.build_source_manifest", side_effect=[manifest, manifest]),
            patch("app.ingest.read_documents", return_value=documents),
            patch("app.ingest.embed_texts", return_value=[[0.1] * 384]),
            patch("app.ingest.replace_chunks") as replace_mock,
        ):
            chunk_count = ingest_documents()

        self.assertEqual(chunk_count, 1)
        self.assertEqual(replace_mock.call_args.kwargs["source_manifest"], manifest)

    def test_documents_changed_during_ingest_preserve_old_index(self):
        original_manifest = [{
            "source_name": "notes.txt",
            "source_type": "txt",
            "file_size": 24,
            "sha256": "before-hash",
        }]
        changed_manifest = [{
            "source_name": "notes.txt",
            "source_type": "txt",
            "file_size": 31,
            "sha256": "after-hash",
        }]
        documents = [{
            "source_name": "notes.txt",
            "source_type": "txt",
            "page_number": None,
            "text": "RAG dokümanlardan cevap üretir.",
        }]

        with (
            patch("app.ingest.init_db"),
            patch(
                "app.ingest.build_source_manifest",
                side_effect=[original_manifest, changed_manifest],
            ),
            patch("app.ingest.read_documents", return_value=documents),
            patch("app.ingest.embed_texts", return_value=[[0.1] * 384]),
            patch("app.ingest.replace_chunks") as replace_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "indeksleme sırasında değişti"):
                ingest_documents()

        replace_mock.assert_not_called()


class IndexedSourcesTests(unittest.TestCase):
    def test_get_indexed_sources_returns_empty_when_db_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "missing.db"

            with patch.object(database, "DB_PATH", db_path):
                sources = database.get_indexed_sources()

        self.assertEqual(sources, [])

    def test_get_indexed_sources_prepares_schema_when_table_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "rag.db"
            sqlite3.connect(db_path).close()

            with patch.object(database, "DB_PATH", db_path):
                sources = database.get_indexed_sources()

                conn = database.get_connection()
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name='chunks'"
                    )
                    table_row = cursor.fetchone()
                finally:
                    conn.close()

        self.assertEqual(sources, [])
        self.assertIsNotNone(table_row)

    def test_get_indexed_sources_works_with_legacy_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "rag.db"

            conn = sqlite3.connect(db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """)
                cursor.execute(
                    """
                    INSERT INTO chunks (source_name, chunk_text, embedding)
                    VALUES (?, ?, ?)
                    """,
                    ("legacy.txt", "eski şema içeriği", json.dumps([0.1, 0.2])),
                )
                conn.commit()
            finally:
                conn.close()

            with patch.object(database, "DB_PATH", db_path):
                sources = database.get_indexed_sources()

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source_name"], "legacy.txt")
        self.assertIsNone(sources[0]["source_type"])
        self.assertEqual(sources[0]["chunk_count"], 1)
        self.assertEqual(sources[0]["page_count"], 0)

    def test_get_indexed_sources_groups_by_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "rag.db"

            with patch.object(database, "DB_PATH", db_path):
                database.init_db()
                database.insert_chunk(
                    "notes.txt",
                    "txt chunk 1",
                    [0.1, 0.2],
                    source_type="txt",
                    page_number=None,
                    chunk_index=0,
                )
                database.insert_chunk(
                    "notes.txt",
                    "txt chunk 2",
                    [0.3, 0.4],
                    source_type="txt",
                    page_number=None,
                    chunk_index=1,
                )
                database.insert_chunk(
                    "guide.pdf",
                    "pdf page 1",
                    [0.5, 0.6],
                    source_type="pdf",
                    page_number=1,
                    chunk_index=0,
                )
                database.insert_chunk(
                    "guide.pdf",
                    "pdf page 2 a",
                    [0.7, 0.8],
                    source_type="pdf",
                    page_number=2,
                    chunk_index=1,
                )
                database.insert_chunk(
                    "guide.pdf",
                    "pdf page 2 b",
                    [0.9, 1.0],
                    source_type="pdf",
                    page_number=2,
                    chunk_index=2,
                )

                sources = database.get_indexed_sources()

        self.assertEqual(len(sources), 2)

        by_name = {source["source_name"]: source for source in sources}

        self.assertEqual(by_name["guide.pdf"]["source_type"], "pdf")
        self.assertEqual(by_name["guide.pdf"]["chunk_count"], 3)
        self.assertEqual(by_name["guide.pdf"]["page_count"], 2)

        self.assertEqual(by_name["notes.txt"]["source_type"], "txt")
        self.assertEqual(by_name["notes.txt"]["chunk_count"], 2)
        self.assertEqual(by_name["notes.txt"]["page_count"], 0)

        self.assertEqual(
            [source["source_name"] for source in sources],
            ["guide.pdf", "notes.txt"],
        )


class SourcesCommandTests(unittest.TestCase):
    def test_handle_command_sources_empty_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "missing.db"
            buffer = io.StringIO()

            with patch.object(database, "DB_PATH", db_path):
                with redirect_stdout(buffer):
                    result = main.handle_command("/sources")

        output = buffer.getvalue()

        self.assertEqual(result, "handled")
        self.assertIn("İndekste kaynak dosya yok", output)
        self.assertIn("/reindex", output)

    def test_handle_command_sources_prints_index_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "rag.db"
            buffer = io.StringIO()

            with patch.object(database, "DB_PATH", db_path):
                database.init_db()
                database.insert_chunk(
                    "example.txt",
                    "örnek chunk",
                    [0.1, 0.2],
                    source_type="txt",
                    page_number=None,
                    chunk_index=0,
                )
                database.insert_chunk(
                    "guide.pdf",
                    "pdf chunk",
                    [0.3, 0.4],
                    source_type="pdf",
                    page_number=1,
                    chunk_index=0,
                )

                with redirect_stdout(buffer):
                    result = main.handle_command("/sources")

        output = buffer.getvalue()

        self.assertEqual(result, "handled")
        self.assertIn("İndeksteki kaynaklar", output)
        self.assertIn("example.txt", output)
        self.assertIn("guide.pdf", output)
        self.assertIn("txt", output)
        self.assertIn("pdf", output)
        self.assertIn("Toplam: 2 dosya, 2 chunk", output)


if __name__ == "__main__":
    unittest.main()
