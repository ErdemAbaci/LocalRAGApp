import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from app import database
from app.retrieval import get_top_chunks


class SourceDatabaseTests(unittest.TestCase):
    def test_chunks_can_be_filtered_and_read_by_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "rag.db"

            with patch.object(database, "DB_PATH", db_path):
                database.init_db()
                database.insert_chunk(
                    "first.txt",
                    "Birinci kaynak",
                    [1.0, 0.0],
                    source_type="txt",
                    chunk_index=1,
                )
                database.insert_chunk(
                    "second.txt",
                    "Ikinci kaynak",
                    [0.0, 1.0],
                    source_type="txt",
                    chunk_index=1,
                )

                filtered = database.get_all_chunks(source_name="second.txt")
                chunk = database.get_chunk_by_id(filtered[0]["id"])

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["source_name"], "second.txt")
        self.assertEqual(chunk["chunk_text"], "Ikinci kaynak")
        self.assertNotIn("embedding", chunk)

    def test_missing_database_or_chunk_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "missing.db"

            with patch.object(database, "DB_PATH", db_path):
                self.assertIsNone(database.get_chunk_by_id(999))


class FilteredRetrievalTests(unittest.TestCase):
    def test_retrieval_loads_only_selected_source(self):
        chunks = [{
            "id": 3,
            "source_name": "notes.txt",
            "source_type": "txt",
            "page_number": None,
            "chunk_index": 1,
            "chunk_text": "Guvenlik notu",
            "embedding": [1.0, 0.0],
        }]

        with patch("app.retrieval.get_all_chunks", return_value=chunks) as get_chunks:
            with patch("app.retrieval.embed_texts", return_value=[[1.0, 0.0]]):
                results = get_top_chunks(
                    "Guvenlik nedir?",
                    top_k=1,
                    source_name="notes.txt",
                )

        get_chunks.assert_called_once_with(source_name="notes.txt")
        self.assertEqual(results[0]["source_name"], "notes.txt")


class SourceCommandTests(unittest.TestCase):
    def tearDown(self):
        main._source_filter = None

    def test_show_command_prints_full_chunk(self):
        chunk = {
            "id": 42,
            "source_name": "datamining.pdf",
            "source_type": "pdf",
            "page_number": 2,
            "chunk_index": 3,
            "chunk_text": "Veri temizleme eksik ve hatali verileri duzeltir.",
        }
        buffer = io.StringIO()

        with patch("main.get_chunk_by_id", return_value=chunk):
            with redirect_stdout(buffer):
                result = main.handle_command("/show 42")

        output = buffer.getvalue()
        self.assertEqual(result, "handled")
        self.assertIn("datamining.pdf", output)
        self.assertIn("Sayfa 2", output)
        self.assertIn("Veri temizleme", output)

    def test_filter_can_be_enabled_inspected_and_disabled(self):
        sources = [{"source_name": "DataMining.pdf"}]
        buffer = io.StringIO()

        with patch("main.get_indexed_sources", return_value=sources):
            with redirect_stdout(buffer):
                main.handle_command("/filter datamining.pdf")
                main.handle_command("/filter")
                main.handle_command("/filter off")

        output = buffer.getvalue()
        self.assertIsNone(main._source_filter)
        self.assertIn("DataMining.pdf", output)
        self.assertIn("filtresi kapatıldı", output)

    def test_interactive_ask_forwards_one_time_source_filter(self):
        sources = [{"source_name": "datamining.pdf"}]

        with patch("main.get_indexed_sources", return_value=sources):
            with patch("main.answer_question", return_value=True) as answer:
                result = main.handle_command(
                    '/ask --source "datamining.pdf" Veri temizleme nedir?'
                )

        self.assertEqual(result, "handled")
        answer.assert_called_once_with(
            "Veri temizleme nedir?",
            source_name="datamining.pdf",
        )

    def test_persistent_filter_is_used_by_normal_question(self):
        chunks = [{
            "id": 1,
            "source_name": "example.txt",
            "source_type": "txt",
            "page_number": None,
            "chunk_index": 1,
            "chunk_text": "RAG aciklamasi",
            "score": 0.05,
        }]
        main._source_filter = "example.txt"

        with patch("main.warn_if_index_is_stale"):
            with patch("main.get_top_chunks", return_value=chunks) as retrieval:
                with redirect_stdout(io.StringIO()):
                    success = main.answer_question("RAG nedir?")

        self.assertTrue(success)
        retrieval.assert_called_once_with(
            "RAG nedir?",
            top_k=3,
            neighbor_radius=1,
            source_name="example.txt",
        )

    def test_cli_show_and_filtered_ask_have_operational_exit_codes(self):
        sources = [{"source_name": "example.txt"}]

        with patch("main.show_chunk_command", return_value=True) as show:
            self.assertEqual(main.cli(["show", "12"]), 0)
        show.assert_called_once_with(12)

        with patch("main.get_indexed_sources", return_value=sources):
            with patch("main.answer_question", return_value=True) as answer:
                exit_code = main.cli([
                    "ask",
                    "--source",
                    "example.txt",
                    "RAG",
                    "nedir?",
                ])

        self.assertEqual(exit_code, 0)
        answer.assert_called_once_with("RAG nedir?", source_name="example.txt")


if __name__ == "__main__":
    unittest.main()
